import ray
import os
# from ray import tune

from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.algorithms.dqn import DQNConfig
# from rllib_a2c.a2c import A2C, A2CConfig

from gymnasium.envs.registration import register
# Import custom environment
from avp_env.envs.avp_env import RllibEnv
import logging

view = 'side'
resume = True
# Initialise Ray
ray.init(num_gpus=1, logging_level=logging.ERROR)

# Algorithm Configuration List
algorithm_configs = {
    "PPO": PPOConfig(),
    "DQN": DQNConfig(),
    # "A2C": A2CConfig(),
}

# Convolutional Filter Configuration
conv_filters_1 = [
    (32, 8, 4),
    (64, 4, 2),
    (64, 3, 1)
]
num_workers = 1
# Total time steps trained
total_timesteps = 30000


def run_algorithm(algo_config, algo_name, total_timesteps, view, resume=False):
    checkpoint_dir = f"../RL/checkpoints/{algo_name}/{view}/{total_timesteps}"

    os.makedirs(checkpoint_dir, exist_ok=True)
    algo_config = algo_config.training(gamma=0.9, lr=1e-4)
    algo_config = algo_config.resources(num_gpus=1)
    algo_config = algo_config.rollouts(num_rollout_workers=num_workers)

    # algo_config = algo_config.environment(env=AutonomousParkingEnv)
    algo_config = algo_config.environment(
        env=RllibEnv,
        env_config={
            "view": view,
        }
    )

    if algo_name == "DQN":
        algo_config.replay_buffer_config.update({
            "capacity": 2000,
            "storage_unit": "timesteps",  # 避免按 episode 存储
            "compress_observations": True
        })
    elif algo_name == "PPO":
        algo_config = algo_config.training(
        train_batch_size=50,
        sgd_minibatch_size=32,
        lr=1e-4,
        gamma=0.9
        )
    algo_config = algo_config.framework('torch')

    # algo_config = algo_config.model(conv_filters=conv_filters)
    algo_config.model["conv_filters"] = conv_filters_1

    algo = algo_config.build()

    timesteps = 0

    if resume:
        # 获取最近一次保存的 checkpoint 路径
        checkpoints = [os.path.join(checkpoint_dir, name) for name in os.listdir(checkpoint_dir)]
        checkpoints = [ckpt for ckpt in checkpoints if os.path.isdir(ckpt)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Restoring from checkpoint: {latest_checkpoint}")
            algo.restore(latest_checkpoint)
        else:
            print("No checkpoint found to resume from.")


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
        if timesteps % 3000 == 0:
            checkpoint = algo.save(checkpoint_dir)
            print(f"Checkpoint saved at: {checkpoint}")


# Configure and run Benchmark for each online algorithm
for algo_name, algo_config in algorithm_configs.items():
    run_algorithm(algo_config, algo_name, total_timesteps, view, resume)

ray.shutdown()
