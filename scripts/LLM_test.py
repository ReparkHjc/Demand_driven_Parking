from avp_env.metrics.metrics import get_parking_metrics
from avp_env.metrics.utils import instru_len
from avp_env.metrics.experiment import run_experiments, load_experiments
from avp_env.envs.avp_env import MetricsVLLMEnv
from avp_env.agents.LLM_agent import DSVL7BAgent
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AVP experiments and compute metrics.")
    parser.add_argument('--load', action='store_true', help='Whether to load old evaluation')
    parser.add_argument('--instr_type', type=str, default='raw', help='name to instruction file')
    parser.add_argument('--model', type=str, default='deepseek-vl-7b-chat', help='Name of model')
    parser.add_argument('--view', type=str, default='right', help='View of camera from vehicle')
    args = parser.parse_args()

    instruction_path = f'../data/Command/{args.instr_type}_command.json'
    output_file = f'../results/{args.model}/{args.view}/{args.instr_type}_command.json'

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    env = MetricsVLLMEnv()

    if args.model == 'deepseek-vl-7b-chat':
        agent = DSVL7BAgent()
    else:
        raise ValueError(f"Invalid model name '{args.model}'. Please check your input.")

    if args.load:
        experiments = load_experiments(output_file)
    else:
        instru_num = instru_len(instruction_path)
        experiments = run_experiments(env, agent, instru_num, output_file, args.view)

    metrics = get_parking_metrics(experiments)
    print("Metrics:", metrics)
