import json

def instru_len(instruction_path):
    with open(instruction_path, 'r') as f:
        instruction_data = json.load(f)
    return len(instruction_data)

def meets_criteria(slot, tags):
    for key, value in tags.items():
        if slot.get(key) != value:
            return False
    return True

def get_result_tags(result_id, result_park, result_path):
    with open(f'../data/Vision/Park_{result_park}/{result_path}/parking_slots.json', 'r') as f:
        parking_slots = json.load(f)
    return next((slot for slot in parking_slots if slot['ParkingID'] == result_id), None)

def get_target_features(env):
    instruction_info = env.getTargetInstruction()
    results = []
    matching_slots = []

    if hasattr(instruction_info, 'tags'):
        with open(f'../data/Vision/Park_{instruction_info.park_id}/{instruction_info.scan}/parking_slots.json', 'r') as f:
            parking_slots = json.load(f)
        matching_slots = [slot['ParkingID'] for slot in parking_slots if meets_criteria(slot, instruction_info.tags)]

    for parking_id in matching_slots:
        slot = next(slot for slot in parking_slots if slot['ParkingID'] == parking_id)
        result = {
            "scan": instruction_info.scan,
            "path_id": slot['PathID'],
            "instruction": instruction_info.instruction,
            "tags": instruction_info.tags,
            "ParkingID": parking_id,
            "loc_id": slot['LocID'],
            "distance": slot['PathID'],
            "park_id": slot["park_id"],
            "path_num": slot["path_num"]
        }
        results.append(result)

    return matching_slots, results

