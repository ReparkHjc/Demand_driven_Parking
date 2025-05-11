import os
import json
from avp_env.dataLoder.path import PathLoader
from avp_env.common import Trajectory, Instruction, ParkingSlot


class DataReader:
    def __init__(self, env_type):
        self.path_loader = PathLoader(env_type)
        self.experiment_paths = self.path_loader.load_path()

    def _load_json(self, json_path, filename):
        json_path = os.path.join(json_path, filename)
        with open(json_path, 'r') as f:
            return json.load(f)

    def load_parking_slots(self):
        combined_parking_data = []

        for experiment_path in self.experiment_paths:
            parking_data = self._load_json(experiment_path, 'parking_slots.json')
            combined_parking_data.extend(parking_data)

            # parking_data_sorted = sorted(parking_data, key=lambda x: x["ParkingID"])
        combined_parking_data_sorted = sorted(combined_parking_data, key=lambda x: x["ParkingID"])

        return [ParkingSlot(slot_data) for slot_data in combined_parking_data_sorted]

    def load_trajectories(self):
        combined_traj_data = []
        for experiment_path in self.experiment_paths:
            traj_data = self._load_json(experiment_path, 'Traj.json')
            combined_traj_data.extend(traj_data)

        return [Trajectory(traj_entry) for traj_entry in combined_traj_data]

    def load_metrics_instructions(self, env_type):
        # if env_type == 'train':
        #     instruction_name = 'target_command.json'
        # elif env_type == 'test':
        #     instruction_name = 'target_command.json'
        # elif env_type == 'raw':
        #     instruction_name = 'raw_command.json'
        # else:
        #     raise ValueError(f"Invalid env_type '{env_type}'. Please check your input.")
        instruction_name = f"{env_type}_command.json"
        # instruction_data = self._load_json('../data/commands', instruction_name)
        instruction_data = self._load_json('../data/Command', instruction_name)

        for instruction in instruction_data:
            instruction['tags']['Occupied'] = 0

            # Set 'Disabled' to 0 if 'Disabled' is not in tags
            if 'Disabled' not in instruction['tags']:
                instruction['tags']['Disabled'] = 0

            # Set 'Charging' to 0 if 'Charging' is not in tags
            if 'Charging' not in instruction['tags']:
                instruction['tags']['Charging'] = 0

        return [Instruction(instruction_entry) for instruction_entry in instruction_data]

    def load_vision_path(self):
        experiment_paths = self.path_loader.load_path()
        for experiment_path in experiment_paths:
            park_num = os.path.basename(os.path.dirname(experiment_path))  # Park_1
            park_id = park_num.split('_')[-1]  # '1'

            experiment_id = os.path.basename(experiment_path)  # 20240423

            # subfolder
            first_subdir = next((os.path.join(experiment_path, d) for d in os.listdir(experiment_path)
                                 if os.path.isdir(os.path.join(experiment_path, d))), None)

            if first_subdir:
                path_num = len(os.listdir(first_subdir))
            else:
                raise ValueError(f"Invalid folder '{experiment_path}'. Please check your input.")

            return park_id, experiment_id, path_num


