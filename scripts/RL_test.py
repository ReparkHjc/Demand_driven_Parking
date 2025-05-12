import ray
import os
import datetime
from ray import tune
from ray.tune import TuneConfig, RunConfig
from ray.rllib.algorithms.dqn import DQNConfig
from avp_env.envs.avp_env import RllibEnv
import logging
from ray.air.config import CheckpointConfig

# 配置参数
view = 'side'
ray.init(num_gpus=1, logging_level=logging.ERROR)

# 算法列表
algorithm_configs = {
    "DQN": DQNConfig()
}

# 模型卷积层设置
conv_filters_1 = [
    (32, 8, 4),
    (64, 4, 2),
    (64, 3, 1)
]

# 总训练步数
total_timesteps = 50000
num_workers = 1


def run_algorithm(algo_config, algo_name, total_timesteps, view):
    # 日志 & checkpoint 路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{algo_name}_{view}_{timestamp}"
    results_dir = "./RL/results"
    log_path = os.path.abspath(results_dir)

    # 应用配置
    algo_config = (
        algo_config
        .training(gamma=0.9, lr=1e-4)
        .resources(num_gpus=1)
        .env_runners(
            num_env_runners=num_workers,
        )
        .environment(env=RllibEnv, env_config={"view": view})
        .framework("tf")  # 推荐 TF 以启用 TensorBoard
    )

    # 模型结构配置
    algo_config.model["conv_filters"] = conv_filters_1

    # Replay Buffer 配置
    algo_config.replay_buffer_config.update({
        "capacity": 2000,
        "storage_unit": "timesteps",
        "compress_observations": True
    })

    # 启动 Tuner（推荐方式）
    tuner = tune.Tuner(
        trainable=algo_name,
        param_space=algo_config.to_dict(),
        tune_config=TuneConfig(),
        run_config=RunConfig(
            name=experiment_name,
            storage_path=f"file://{log_path}",
            stop={"timesteps_total": total_timesteps},
            verbose=1,
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=2000,
            ),
        )
    )

    results = tuner.fit()
    print(f"Training complete for {algo_name}. Results at: {results_dir}/{experiment_name}")


# === 执行所有算法 ===
for algo_name, algo_config in algorithm_configs.items():
    run_algorithm(algo_config, algo_name, total_timesteps, view)

ray.shutdown()
