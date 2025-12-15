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

        # 2. Phase number 
        self.num_phases = 8

        # 3. Incoming track IDs (State Space)
        self.incoming_roads = [
            "road_0_1_0",  # W to E
            "road_2_1_2",  # E to W
            "road_1_0_1",  # S to N
            "road_1_2_3"   # N to S
        ]
        # 4. Outgoing track IDs
        self.outgoing_roads = [
            "road_1_1_0",  # E
            "road_1_1_2",  # W
            "road_1_1_1",  # N
            "road_1_1_3"   # S
        ] 

        # 5. Track number of each road
        self.lanes_per_road = 7
        

        # Automatically generate every track IDs needed to be observed
        self.start_lane_ids = [] # for observation
        self.end_lane_ids = []   # for reward
        
        for lane_idx in range(self.lanes_per_road):
            for r_in in self.incoming_roads:
                self.start_lane_ids.append(f"{r_in}_{lane_idx}")
            for r_out in self.outgoing_roads:
                self.end_lane_ids.append(f"{r_out}_{lane_idx}")
        print(f"Successfully initialize environment: watch the crossing {self.intersection_id}, focus on {len(self.start_lane_ids)} incoming tracks, {len(self.end_lane_ids)} outgoing tracks")
        # ===================================================

        # Define action space: 0-7 choose phase
        self.action_space = spaces.Discrete(self.num_phases)
        # Define watching space: shape for (roads x tracks, )，即 (4 x 7 = 28, )
        self.observation_space = spaces.Box(
            low=0,
            high=200,
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
        # Pressure = Incflow - Outflow, Reward = -Pressure
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        
        # 1. Inflow
        total_incoming = 0
        for lane_id in self.start_lane_ids:
            total_incoming += lane_vehicle_count.get(lane_id, 0)
            
        # 2. Outflow
        total_outgoing = 0
        for lane_id in self.end_lane_ids:
            total_outgoing += lane_vehicle_count.get(lane_id, 0)
            
        # 3. Pressure
        pressure = total_incoming - total_outgoing
        return -np.abs(pressure)
    
    """def _get_reward(self):
        Calculate the reward: Minimize queuing
        lane_waiting_count = self.eng.get_lane_waiting_vehicle_count()
        total_waiting = 0
        for lane_id in self.start_lane_ids:
            total_waiting += lane_waiting_count.get(lane_id, 0)
        return -total_waiting"""

    """def _get_reward(self):
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
    
        in_flow = sum(lane_vehicle_count.get(lane, 0) for lane in self.start_lane_ids)
        # 出口车道：对应 incoming roads 的对侧出口
        # 简化：我们用等待数作为代理
        out_flow = sum(self.eng.get_lane_waiting_vehicle_count().get(lane, 0) for lane in self.start_lane_ids)

        pressure = in_flow - out_flow
        return -abs(pressure)/10"""
 