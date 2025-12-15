import gymnasium as gym
from gymnasium import spaces
import cityflow
import numpy as np
import json
import os


class CityFlowMultiEnv(gym.Env):
    """
    Multi-Agent CityFlow Environment with Pressure-based Reward
    """

    def __init__(self, config_path, steps_per_action=10, reward_type='pressure'):
        super().__init__()

        self.config_path = config_path
        self.steps_per_action = steps_per_action
        self.reward_type = reward_type  # 'pressure', 'waiting', or 'mixed'
        self.current_step = 0

        # Load config
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.dir = cfg["dir"]
        self.roadnet_file = os.path.join(self.dir, cfg["roadnetFile"])
        self.flow_file = os.path.join(self.dir, cfg["flowFile"])

        # Parse roadnet
        with open(self.roadnet_file, "r") as f:
            roadnet = json.load(f)

        # Get real intersections
        self.intersection_ids = [
            inter["id"] for inter in roadnet["intersections"]
            if not inter.get("virtual", False)
        ]
        self.num_agents = len(self.intersection_ids)

        print(f"✅ Detected {self.num_agents} intersections: {self.intersection_ids}")

        # Phase configuration
        self.phase_dict = {}
        for inter in roadnet["intersections"]:
            if not inter.get("virtual", False):
                iid = inter["id"]
                if "trafficLight" in inter and inter["trafficLight"]:
                    phases = inter["trafficLight"].get("lightphases", [])
                    self.phase_dict[iid] = len(phases) if len(phases) > 0 else 4
                else:
                    self.phase_dict[iid] = 4

        # Action space
        self.action_space = spaces.MultiDiscrete(
            [self.phase_dict[iid] for iid in self.intersection_ids]
        )

        # Observation space: [waiting_vehicles, total_vehicles] per intersection
        # Shape: (num_agents, 2)
        self.observation_space = spaces.Box(
            low=0, high=500, shape=(self.num_agents, 2), dtype=np.float32
        )

        # Initialize engine
        self.eng = cityflow.Engine(config_path, thread_num=1)
        print("✅ CityFlow engine initialized")

        # Build lane mapping
        self._build_lane_mapping(roadnet)

        # Previous waiting time for reward shaping
        self.prev_waiting = None

    def _build_lane_mapping(self, roadnet):
        """Build intersection -> incoming lanes mapping"""
        self.incoming_lanes = {iid: [] for iid in self.intersection_ids}

        for road in roadnet["roads"]:
            road_id = road["id"]
            num_lanes = len(road["lanes"])
            lane_ids = [f"{road_id}_{i}" for i in range(num_lanes)]

            end_inter = road["endIntersection"]
            if end_inter in self.incoming_lanes:
                self.incoming_lanes[end_inter].extend(lane_ids)

        print(f"🔍 Lane mapping: {sum(len(v) for v in self.incoming_lanes.values())} total lanes")

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        self.eng.reset()
        self.current_step = 0
        self.prev_waiting = None
        return self._get_state(), {}

    def step(self, actions):
        """Execute actions"""
        # Set phases
        for idx, iid in enumerate(self.intersection_ids):
            try:
                self.eng.set_tl_phase(iid, int(actions[idx]))
            except Exception as e:
                print(f"Warning: Phase setting failed for {iid}: {e}")

        # Simulate
        for _ in range(self.steps_per_action):
            self.eng.next_step()
            self.current_step += 1

        obs = self._get_state()
        reward = self._get_reward()
        done = self.current_step >= 3600
        truncated = False

        info = {
            'total_waiting': self._get_total_waiting(),
            'total_vehicles': self._get_total_vehicles(),
            'avg_pressure': np.mean([abs(r) for r in self._get_pressure_values()]),
        }

        return obs, reward, done, truncated, info

    def _get_state(self):
        """
        Get state for all agents.
        State = [waiting_vehicles, total_vehicles] for each intersection
        Shape: (num_agents, 2)
        """
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        lane_waiting_count = self.eng.get_lane_waiting_vehicle_count()

        states = []
        for iid in self.intersection_ids:
            waiting = sum(lane_waiting_count.get(lane_id, 0)
                          for lane_id in self.incoming_lanes[iid])
            total = sum(lane_vehicle_count.get(lane_id, 0)
                        for lane_id in self.incoming_lanes[iid])
            states.append([waiting, total])

        return np.array(states, dtype=np.float32)

    def _get_reward(self):
        """
        Calculate reward based on reward_type
        """
        if self.reward_type == 'pressure':
            return self._get_pressure_reward()
        elif self.reward_type == 'waiting':
            return self._get_waiting_reward()
        elif self.reward_type == 'mixed':
            return self._get_mixed_reward()
        else:
            return self._get_pressure_reward()

    def _get_pressure_reward(self):
        """Pressure-based reward"""
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        lane_waiting_count = self.eng.get_lane_waiting_vehicle_count()

        for idx, iid in enumerate(self.intersection_ids):
            in_flow = sum(lane_vehicle_count.get(lane_id, 0)
                          for lane_id in self.incoming_lanes[iid])
            out_flow = sum(lane_waiting_count.get(lane_id, 0)
                           for lane_id in self.incoming_lanes[iid])

            pressure = in_flow - out_flow
            rewards[idx] = -abs(pressure) / 10.0

        return rewards

    def _get_waiting_reward(self):
        """Waiting-based reward with delta"""
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        lane_waiting_count = self.eng.get_lane_waiting_vehicle_count()

        current_waiting = []
        for idx, iid in enumerate(self.intersection_ids):
            waiting = sum(lane_waiting_count.get(lane_id, 0)
                          for lane_id in self.incoming_lanes[iid])
            current_waiting.append(waiting)

        if self.prev_waiting is not None:
            # Reward = improvement in waiting time
            for idx in range(self.num_agents):
                delta = self.prev_waiting[idx] - current_waiting[idx]
                rewards[idx] = delta  # Positive if waiting decreased
        else:
            # First step: penalize current waiting
            rewards = -np.array(current_waiting, dtype=np.float32) / 10.0

        self.prev_waiting = current_waiting
        return rewards

    def _get_mixed_reward(self):
        """Mixed reward: pressure + waiting improvement"""
        pressure_reward = self._get_pressure_reward()
        waiting_reward = self._get_waiting_reward()
        return 0.7 * pressure_reward + 0.3 * waiting_reward

    def _get_pressure_values(self):
        """Get pressure values for all intersections"""
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        lane_waiting_count = self.eng.get_lane_waiting_vehicle_count()

        pressures = []
        for iid in self.intersection_ids:
            in_flow = sum(lane_vehicle_count.get(lane_id, 0)
                          for lane_id in self.incoming_lanes[iid])
            out_flow = sum(lane_waiting_count.get(lane_id, 0)
                           for lane_id in self.incoming_lanes[iid])
            pressures.append(in_flow - out_flow)

        return pressures

    def _get_total_waiting(self):
        """Total waiting vehicles"""
        lane_wait = self.eng.get_lane_waiting_vehicle_count()
        return sum(lane_wait.get(lane_id, 0)
                   for iid in self.intersection_ids
                   for lane_id in self.incoming_lanes[iid])

    def _get_total_vehicles(self):
        """Total vehicles"""
        lane_vehicles = self.eng.get_lane_vehicle_count()
        return sum(lane_vehicles.get(lane_id, 0)
                   for iid in self.intersection_ids
                   for lane_id in self.incoming_lanes[iid])

    def render(self):
        pass

    def close(self):
        pass