from avp_env.envs.avp_env import MetricsVLLMEnv
from avp_env.agents.LLM_agent import combineMultimodalLLMAgent
from avp_env.agents.image_process import combine_views, split_multi_view_image


def get_result_id(env, agent, instructions_index=None):
    obs = env.resetVLLM(instructions_index)
    done = False
    steps = 0

    while not done:
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        steps += 1

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
