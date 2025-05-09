from avp_env.envs.avp_env import MetricsVLLMEnv
from avp_env.agents.LLM_agent import combineMultimodalLLMAgent
from avp_env.agents.image_process import combine_views, split_multi_view_image


def get_result_id(env, agent, instructions_index=None):
    state = env.reset(instructions_index)
    done = False
    instruction = env.getTargetInstruction().instruction

    while not done:
        position = env.getPosition()
        img_np = env.render()[0]  # HWC image as numpy
        front_img, left_img, right_img, back_img = split_multi_view_image(img_np)
        img = right_img

        action = agent.get_action(img, instruction, position)
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
