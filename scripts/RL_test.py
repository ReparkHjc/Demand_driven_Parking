import numpy as np
from avp_env.envs.avp_env import MetricsEnv, RllibEnv
from avp_env.agents.rule import RulebasedAgent
import json
import zipfile
import os
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from avp_env.metrics.experiment import run_rl_experiments
from avp_env.metrics.metrics import get_parking_metrics

def instru_len(instruction_path):
    with open(instruction_path, 'r') as f:
        instruction_data = json.load(f)
    return len(instruction_data)

if __name__ == "__main__":
    # 创建 AutonomousParkingEnv 环境实例
    env = RllibEnv()
    # Algorithm Configuration List
    algorithm_configs = {
        "PPO": PPOConfig(),
        "DQN": DQNConfig(),
        # "A2C": A2CConfig(),
    }
    for algo_name, algo_config in algorithm_configs.items():
        # Convolutional Filter Configuration
        conv_filters_1 = [
            (32, 8, 4),
            (64, 4, 2),
            (64, 3, 1)
        ]
        view = 'multi'
        checkpoint_path = f"../RL/checkpoints/{algo_name}/{view}/30000/checkpoint_000300"

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
        output_file = f'../results/RL/{algo_name}_result.json'
        log_file_path = f"../results/RL/{algo_name}_result.txt"

        experiments = run_rl_experiments(env, algo, instru_num, output_file, view)

        metrics = get_parking_metrics(experiments)

        log_text = (
                "=" * 40 +
                f"\nMetrics for agent '{algo_name}', view '{view}':\n" +
                f"{metrics}\n" +
                "=" * 40
        )
        print(log_text)


        with open(log_file_path, "a") as f:
            f.write(log_text + "\n")  # 写入文件