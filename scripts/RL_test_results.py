import numpy as np
from avp_env.envs.avp_env import MetricsEnv, RllibEnv
from avp_env.agents.rule import RulebasedAgent
import json
import zipfile
import os
from ray.rllib.algorithms.ppo import PPOConfig

def getResultID(env, agent, instructions_index=None):

    observation = env.reset(instructions_index)

    done = False
    steps = 0

    while not done:
        action = agent.compute_single_action(observation, explore=False)
        observation, reward, done, info = env.step(action)  # 执行动作
        steps += 1

    last_slots = env.getCurrentParkingSlot()


    if last_slots:
        last_slot = last_slots[0]
        result_id = last_slot.ParkingID
    else:
        result_id = []

    return result_id

def get_experiment(env, agent, instru_num):
    experiments = []
    for instructions_index in range(instru_num):
        result_id = getResultID(env=env, agent=agent, instructions_index=instructions_index)
        # target_id = getTargetID(env)

        experiment = {
            "TestScenarioID": env.getScan(),
            "TestInstructionID": instructions_index,
            "VLPDecisionPositionID": result_id
        }

        experiments.append(experiment)

    return experiments

def instru_len(instruction_path):
    with open(instruction_path, 'r') as f:
        instruction_data = json.load(f)
    return len(instruction_data)

if __name__ == "__main__":
    # 创建 AutonomousParkingEnv 环境实例
    env = RllibEnv()
    # Algorithm Configuration List
    algo_config = PPOConfig()

    # Convolutional Filter Configuration
    conv_filters_1 = [
        (32, 8, 4),
        (64, 4, 2),
        (64, 3, 1)
    ]
    view = 'multi'
    checkpoint_path = f"../RL/checkpoints/PPO/{view}/30000/checkpoint_000030"

    os.makedirs(checkpoint_path, exist_ok=True)
    algo_config = algo_config.resources(num_gpus=1)
    algo_config = algo_config.rollouts(num_rollout_workers=1)

    # algo_config = algo_config.environment(env=AutonomousParkingEnv)
    algo_config = algo_config.environment(
        env=RllibEnv,
        env_config={
            "view": view,
        }
    )
    algo = algo_config.build()
    algo.restore(checkpoint_path)

    instruction_path = '../data/Command/test_command.json'
    instru_num = instru_len(instruction_path)

    experiments = get_experiment(env, algo, instru_num)
    # save experiments as JSON
    json_filename = '../result/test_results.json'
    json_folder = os.path.dirname(json_filename)
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    with open(json_filename, 'w') as json_file:
        json.dump(experiments, json_file, indent=4)

    # save json as zip
    zip_filename = '../result/test_results.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_filename)

    print(f"save {json_filename} and zip as {zip_filename}")