import ray
import os
# from ray import tune

from ray.rllib.algorithms.dqn import DQNConfig

from gymnasium.envs.registration import register
# Import custom environment
from avp_env.envs.avp_env import RllibEnv
import logging

# # Register custom environment
# register(
#     id='AutonomousParking',
#     entry_point='AVP_ENV:AutonomousParkingEnv',
# )

# Initialise Ray
ray.init(num_gpus=1, logging_level=logging.ERROR)

# Algorithm Configuration List
algorithm_configs = {
    "DQN": DQNConfig()
}

# Convolutional Filter Configuration
conv_filters_1 = [
    (32, 8, 4),
    (64, 4, 2),
    (64, 3, 1)
]
num_workers = 1
# Total time steps trained
total_timesteps = 100000


def run_algorithm(algo_config, algo_name, total_timesteps):
    checkpoint_dir = f"../RL/checkpoints/{algo_name}"
    log_dir = f"../RL/logs/{algo_name}"

    os.makedirs(checkpoint_dir, exist_ok=True)
    algo_config = algo_config.training(gamma=0.9, lr=1e-4)
    algo_config = algo_config.resources(num_gpus=1)
    algo_config = algo_config.rollouts(num_rollout_workers=num_workers)

    # algo_config = algo_config.environment(env=AutonomousParkingEnv)
    algo_config = algo_config.environment(
        env=RllibEnv,
        env_config={
            "env_type": "raw",
            "view": "multi",
            "args": []
        }
    )
    algo_config.replay_buffer_config.update({
        "capacity": 5000,
        "storage_unit": "timesteps",  # 避免按 episode 存储
        "compress_observations": True
    })

    # algo_config = algo_config.environment(env='AutonomousParking-v6')
    algo_config = algo_config.framework('torch')
    # algo_config = algo_config.model(conv_filters=conv_filters)
    algo_config.model["conv_filters"] = conv_filters_1

    # Add logger config for TensorBoard
    algo_config = algo_config.debugging(log_level="INFO", logger_config={
        "type": "ray.tune.logger.UnifiedLogger",
        "logdir": log_dir,
    })

    algo = algo_config.build()

    timesteps = 0
    while timesteps < total_timesteps:
        result = algo.train()
        timesteps = result["timesteps_total"]
        rwd_mean = result['episode_reward_mean']
        len_mean = result['episode_len_mean']
        print("=*=" * 10)
        # print(f"{algo_name} training at timestep {timesteps}/{total_timesteps}: {result}")
        print(f"{algo_name} training at timestep {timesteps}/{total_timesteps}")
        print(f"|| Episode Reward Mean: {rwd_mean}, Episode Length Mean: {len_mean} ||")
        print("=*=" * 10)

        # Save checkpoints
        if timesteps % 10000 == 0:
            checkpoint = algo.save(checkpoint_dir)
            print(f"Checkpoint saved at: {checkpoint}")


# Configure and run Benchmark for each online algorithm
for algo_name, algo_config in algorithm_configs.items():
    run_algorithm(algo_config, algo_name, total_timesteps)

ray.shutdown()
