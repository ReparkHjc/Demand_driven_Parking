from avp_env.metrics.metrics import get_parking_metrics
from avp_env.metrics.utils import instru_len
from avp_env.metrics.experiment import run_experiments, load_experiments
from avp_env.envs.avp_env import MetricsVLLMEnv
from avp_env.agents.LLM_agent import DSVL7BAgent, QwenVLAgent, JanusAgent
import argparse
import os
import gc
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AVP experiments and compute metrics.")
    parser.add_argument('--load', action='store_true', help='Whether to load old evaluation')
    # parser.add_argument('--instr_type', type=str, default='raw', help='name to instruction file')
    parser.add_argument('--instr_types', nargs='+', type=str, default=['raw','change','long','short','abstract'], help='List of instruction types')
    # parser.add_argument('--model', type=str, default='deepseek-vl-7b-chat', help='Name of model')
    parser.add_argument('--models', nargs='+', type=str, default=['deepseek-vl-7b-chat','Qwen2.5-VL-7B-Instruct','Janus-Pro-7B'], help='Name of model')
    # parser.add_argument('--view', type=str, default='right', help='View of camera from vehicle')
    parser.add_argument('--views', nargs='+', type=str, default=['front', 'left', 'right', 'combined', 'multi', 'side'], help='View of camera from vehicle')
    args = parser.parse_args()
    log_file_path = "../results/metrics_results.txt"

    for instr_type in args.instr_types:
        for model in args.models:
            for view in args.views:
                instruction_path = f'../data/Command/{instr_type}_command.json'
                output_file = f'../results/{model}/{view}/{instr_type}_result.json'

                output_dir = os.path.dirname(output_file)
                os.makedirs(output_dir, exist_ok=True)

                env = MetricsVLLMEnv(env_type=instr_type)

                if model == 'deepseek-vl-7b-chat':
                    agent = DSVL7BAgent()
                elif model == 'Qwen2.5-VL-7B-Instruct':
                    agent = QwenVLAgent()
                # elif model == 'deepseek-vl2-tiny':
                #     agent = DeepseekVL2Agent()
                elif model == 'Janus-Pro-7B':
                    agent = JanusAgent()
                else:
                    raise ValueError(f"Invalid model name '{model}'. Please check your input.")

                if args.load:
                    experiments = load_experiments(output_file)
                else:
                    instru_num = instru_len(instruction_path)
                    experiments = run_experiments(env, agent, instru_num, output_file, view)

                metrics = get_parking_metrics(experiments)
                log_text = (
                        "=" * 40 +
                        f"\nMetrics for instr_type '{instr_type}', model '{model}', view '{view}':\n" +
                        f"{metrics}\n" +
                        "=" * 40
                )
                print(log_text)

                if args.load:
                    with open(log_file_path, "a") as f:
                        f.write(log_text + "\n")  # 写入文件

                del agent
                del env
                gc.collect()
                torch.cuda.empty_cache()