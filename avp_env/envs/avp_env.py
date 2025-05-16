import numpy as np
import gymnasium as gym
from gymnasium import spaces
from transformers import AutoTokenizer
from avp_env.dataLoder import ImageLoader, DataReader
import random
from avp_env.agents.park_match import load_prefect_park

class AutonomousParkingEnv(gym.Env):
    def __init__(self, env_type='raw', args=[]):
        super(AutonomousParkingEnv, self).__init__()
        # self.env_type = 'train'
        self.env_type = env_type
        self.image_raw_shape = (270, 480, 3)
        self.max_string_length = 512
        # self.image_shape = (4, 270, 480, 3)
        self.image_shape = (270, 480, 12)

        # Initialize helpers
        self.image_loader = ImageLoader(self.env_type, self.image_raw_shape)
        self.data_reader = DataReader(self.env_type)
        self.tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased")

        # Initialize environment data
        self.image_data = self.image_loader.image_data
        # self.render_image = self.image_loader.render_image
        self.parking_slots = self.data_reader.load_parking_slots()
        # self.trajectories = self.data_reader.load_trajectories()
        self.metrics_instructions = self.data_reader.load_metrics_instructions(self.env_type)
        self.park_id, self.experiment_id, self.path_num = self.data_reader.load_vision_path()


        # Define observation space
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8),  # 4张图
            spaces.Box(low=0, high=99999, shape=(self.max_string_length,), dtype=np.int64)  # 指令token
        ))

        # Define action space
        self.action_space = spaces.Discrete(3)

        # Initialize state
        self.current_observation = (
            np.zeros(self.image_shape, dtype=np.uint8), np.zeros(self.max_string_length, dtype=np.uint8))

    def get_parking_slots(self, loc_id, path_id):
        return [slot for slot in self.parking_slots if slot.LocID == loc_id and slot.PathID == path_id]

    def get_perfect_trajectory(self, trajectory):
        if hasattr(trajectory, "path_id"):
            perfect_trajectory = [0] * (trajectory.path_id - 1)
            perfect_trajectory.append(trajectory.loc_id)
        else:
            perfect_trajectory = [0] * int(trajectory.path_num)
        return perfect_trajectory

    def reset(self, InsIndex=None):
        self.current_position = 1
        self.target_instruction = random.choice(self.metrics_instructions)
        instruction_tokens = self.tokenizer.encode(
            self.target_instruction.instruction, add_special_tokens=True,
            max_length=self.max_string_length, padding='max_length',
            truncation=True
        )

        self.inital_instruction = np.array(instruction_tokens)

        # self.perfect_trajectory = self.get_perfect_trajectory(self.target_instruction)

        key = f"{self.park_id}/{self.experiment_id}/{(self.current_position-1):06d}.jpg"
        # self.render_observation = self.render_image[key]
        self.current_observation = (
            self.image_data[key], self.inital_instruction
        )

        self.matching_slots = load_prefect_park(self.target_instruction, self.parking_slots)

        return self.current_observation

    # def getPerfectTraj(self):
    #     return self.perfect_trajectory

    def getPosition(self):
        return self.current_position

    def getCurrentParkingSlot(self):
        return self.CurrentParkingSlot

    def getParkingSlotInfo(self):
        return self.parking_slots

    def getTargetInstruction(self):
        return self.target_instruction

    def getMatchingSlots(self):
        return self.matching_slots

    def getParkInfo(self):
        return self.park_id, self.experiment_id, self.path_num

    def step(self, action):

        # Execute action and return reward, next observation, whether to terminate, debugging information
        if self.current_position > int(self.path_num):
            reward = -2
            done = True
            self.CurrentParkingSlot = []
        elif action == 0 and self.current_position != int(self.path_num):
            reward = 0
            self.current_position += 1
            done = False
        elif action == 0 and self.current_position == int(self.path_num):
            reward = -1
            done = True
            self.CurrentParkingSlot = []
        else:
            self.CurrentParkingSlot = self.get_parking_slots(action, self.current_position)
            done = True
            # if hasattr(self.target_instruction, 'ParkingID'):
            reward = self.getReward(self.CurrentParkingSlot)
            # else:
            #     reward = 0

        key = f"{self.park_id}/{self.experiment_id}/{(self.current_position-1):06d}.jpg"

        # self.render_observation = self.render_image[key]
        self.current_observation = (
            self.image_data[key], self.inital_instruction
        )

        info = {}
        # print(f"Step: done={done}, reward={reward}")

        return self.current_observation, reward, done, info

    def getReward(self, CurrentParkingSlot):
        if CurrentParkingSlot:
            for slot in CurrentParkingSlot:
                if slot.ParkingID in self.matching_slots:
                    reward = 3  # Give a big reward if the current parking space is the same as the target parking slot

                elif slot.Occupied != 0:
                    reward = -1  # Give a negative punitive reward for not having an empty parking slot

                elif slot.Disabled != self.target_instruction.tags['Disabled']:
                    reward = -0.3  # Give a negative punitive reward for parking in the wrong disabled slot

                elif slot.Charging != self.target_instruction.tags['Charging']:
                    reward = -0.2  # Give a negative punitive reward for parking in the wrong charging slot

                else:
                    reward = 1
                    for key, value in self.target_instruction.tags.items():
                        if getattr(slot, key, None) == value:
                            reward += 0.2  # If the attribute in slot is the same as the target attribute, give a medium reward

        else:
            reward = -1  # Give a negative punitive reward for parking in a non-existent parking space

        return reward

    # def render(self, mode='human'):
    #     self.render_observation[:, :, [0, 2]] = self.render_observation[:, :, [2, 0]]
    #     # Optional rendering method for visualising the state of the environment
    #     img, command = self.render_observation, self.target_instruction.instruction
    #     return img, command

    def close(self):

        pass


class RllibEnv(AutonomousParkingEnv):
    def __init__(self, config=None):
        config = config or {}
        view = config.get("view", "multi")
        super().__init__()
        if view == "side":
            self.image_shape = (270, 480, 6)
            self.image_data = self.image_loader.image_side_data
            self.observation_space = spaces.Tuple((
                spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8),  # 4张图
                spaces.Box(low=0, high=99999, shape=(self.max_string_length,), dtype=np.int64)  # 指令token
            ))
        elif view == "right":
            self.image_shape = (270, 480, 3)
            self.image_data = self.image_loader.image_right_data
            self.observation_space = spaces.Tuple((
                spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8),  # 4张图
                spaces.Box(low=0, high=99999, shape=(self.max_string_length,), dtype=np.int64)  # 指令token
            ))


class MetricsVLLMEnv(AutonomousParkingEnv):
    def __init__(self, env_type='raw', args = []):
        super(MetricsVLLMEnv, self).__init__(env_type=env_type, args=args)

    def reset(self, InsIndex=None):
        self.current_position = 1
        if InsIndex == None:
            # Select the next trajectory in sequence
            self.target_instruction = self.metrics_instructions[self.trajectory_index]
            self.trajectory_index = (self.trajectory_index + 1) % len(self.metrics_instructions)
        else:
            self.target_instruction = self.metrics_instructions[InsIndex]

        self.inital_instruction = self.target_instruction.instruction
        print(self.inital_instruction)
        key = f"{self.park_id}/{self.experiment_id}/{(self.current_position-1):06d}.jpg"
        # self.render_observation = self.render_image[key]
        self.current_observation = (
            self.image_data[key], self.inital_instruction
        )
        self.matching_slots = load_prefect_park(self.target_instruction, self.parking_slots)

        return self.current_observation

        
class MetricsEnv(AutonomousParkingEnv):
    def __init__(self, env_type='test'):
        super(MetricsEnv, self).__init__(env_type)
        self.env_type = env_type
        self.trajectory_index = 0  # Initialize trajectory index
        self.traj_len = len(self.metrics_instructions)
        # Initialize helpers
        self.image_loader = ImageLoader(self.env_type, self.image_raw_shape)
        self.data_reader = DataReader(self.env_type)
        self.tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased")

        # Initialize environment data
        self.image_data = self.image_loader.image_data
        # self.render_image = self.image_loader.render_image
        self.parking_slots = self.data_reader.load_parking_slots()
        # self.trajectories = self.data_reader.load_trajectories()
        self.metrics_instructions = self.data_reader.load_metrics_instructions(self.env_type)

    def getScan(self):
        return self.experiment_id

    def reset(self, InsIndex=None):

        self.current_position = 1
        if InsIndex == None:
            # Select the next trajectory in sequence
            self.target_instruction = self.metrics_instructions[self.trajectory_index]
            self.trajectory_index = (self.trajectory_index + 1) % len(self.metrics_instructions)
        else:
            self.target_instruction = self.metrics_instructions[InsIndex]

        instruction_tokens = self.tokenizer.encode(
            self.target_instruction.instruction, add_special_tokens=True,
            max_length=self.max_string_length, pad_to_max_length=True,
            truncation=True
        )

        self.inital_instruction = np.array(instruction_tokens)

        # self.perfect_trajectory = self.get_perfect_trajectory(self.target_instruction)

        key = f"{self.park_id}/{self.experiment_id}/{self.current_position:06d}.jpg"
        # self.render_observation = self.render_image[key]
        self.current_observation = (
            self.image_data[key], self.inital_instruction
        )

        return self.current_observation
