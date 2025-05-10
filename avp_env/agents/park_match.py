

def meets_criteria(slot, tags):
    for key, value in tags.items():
        if getattr(slot, key, None) != value:
            return False
    return True

def load_prefect_park(parking_instruction, parking_slots):

    matching_slots = [slot.ParkingID for slot in parking_slots if meets_criteria(slot, parking_instruction.tags)]
    return matching_slots
