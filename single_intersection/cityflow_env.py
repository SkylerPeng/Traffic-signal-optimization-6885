import gymnasium as gym
from gymnasium import spaces
import cityflow
import numpy as np
import os

class CityFlowEnv(gym.Env):
    def __init__(self, config_path, steps_per_action=10):
        super(CityFlowEnv, self).__init__()

        self.config_path = config_path
        self.steps_per_action = steps_per_action
        self.current_step = 0

        # Initialize CityFlow engine
        print(f"Loading file: {config_path}")
        self.eng = cityflow.Engine(config_path, thread_num=1)

        # 1. Crossing ID
        self.intersection_id = "intersection_1_1"

        # 2. Phase number (Phase)
        self.num_phases = 8
        # 3. Import track IDs (State Space)
        self.incoming_roads = [
            "road_0_1_0",  # W to E
            "road_2_1_2",  # E to W
            "road_1_0_1",  # S to N
            "road_1_2_3"   # N to S
        ]

        # 4. Track number of each road
        self.lanes_per_road = 7

        # Automatically generate every track IDs needed to be observed
        self.start_lane_ids = []
        for road_id in self.incoming_roads:
            for lane_idx in range(self.lanes_per_road):
                # Concatenate lane IDs, e.g., road_0_1_0_0, road_0_1_0_1, ...
                self.start_lane_ids.append(f"{road_id}_{lane_idx}")

        print(f"Successfully initialize environment: watch the crossing {self.intersection_id}, focus on {len(self.start_lane_ids)}")

        # Define action space: 0-7 choose phase
        self.action_space = spaces.Discrete(self.num_phases)
        # Define watching space: shape for (roads x tracks, )i.e.,(4 x 7 = 28, )
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(len(self.start_lane_ids),),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.eng.reset()
        self.current_step = 0
        return self._get_state(), {}

    def step(self, action):
        # Setup Phase
        try:
            self.eng.set_tl_phase(self.intersection_id, int(action))
        except Exception as e:
            # if error, phase ID out of range
            print(f"Setup Phase Fail (Action {action}): {e}")
            raise e

        # 2. Execute multi-step simulation
        for _ in range(self.steps_per_action):
            self.eng.next_step()
            self.current_step += 1

        # 3. Feedback
        observation = self._get_state()
        reward = self._get_reward()

        # Assume 3600s for a round
        terminated = self.current_step >= 3600
        truncated = False

        return observation, reward, terminated, truncated, {}

    def _get_state(self):
        """Gain the number of cars on the track"""
        lane_vehicle_count = self.eng.get_lane_vehicle_count()

        state = []
        for lane_id in self.start_lane_ids:
            # if ID error,return 0
            cnt = lane_vehicle_count.get(lane_id, 0)
            state.append(cnt)

        return np.array(state, dtype=np.float32)

    
    def _get_reward(self):
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        lane_waiting_count = self.eng.get_lane_waiting_vehicle_count()

        # incoming lanes vehicles
        in_flow = sum(lane_vehicle_count.get(lane, 0)
                      for lane in self.start_lane_ids)

        # waiting as proxy
        out_flow = sum(lane_waiting_count.get(lane, 0)
                       for lane in self.start_lane_ids)

        # pressure
        pressure = in_flow - out_flow

        reward = -abs(pressure) / 10.0

        return reward

