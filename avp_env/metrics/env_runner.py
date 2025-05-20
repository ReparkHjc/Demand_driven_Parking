from avp_env.agents.image_process import get_view_image
from avp_env.agents.prompt_process import build_parking_prompt

def get_result_id(env, agent, view, instructions_index=None):
    state = env.reset(instructions_index)
    done = False
    instruction = env.getTargetInstruction().instruction

    while not done:
        position = env.getPosition()
        img_np = env.render()[0]  # HWC image as numpy

        img = get_view_image(img_np, view)
        prompt = build_parking_prompt(instruction, position, view)

        action = agent.get_action(img, prompt)
        state, reward, done, info = env.step(action)

    last_slots = env.getCurrentParkingSlot()
    path_id = env.getPosition()
    loc_id = action

    result_features = {
        "path_id": path_id,
        "loc_id": loc_id,
        "distance": path_id,
    }

    result_id = last_slots[0].ParkingID if last_slots else []
    return result_id, result_features

def get_rl_result_id(env, agent, view, instructions_index=None):
    state = env.reset(instructions_index)
    done = False

    while not done:
        action = agent.compute_single_action(state, explore=False)

        state, reward, done, info = env.step(action)

    last_slots = env.getCurrentParkingSlot()
    path_id = env.getPosition()
    loc_id = action

    result_features = {
        "path_id": path_id,
        "loc_id": loc_id,
        "distance": path_id,
    }

    result_id = last_slots[0].ParkingID if last_slots else []
    return result_id, result_features
