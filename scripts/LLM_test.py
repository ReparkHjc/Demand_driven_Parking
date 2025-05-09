from avp_env.metrics.metrics import get_parking_metrics
from avp_env.metrics.utils import instru_len
from avp_env.metrics.experiment import run_experiments, load_experiments
from avp_env.envs.avp_env import MetricsVLLMEnv
from avp_env.agents.LLM_agent import combineMultimodalLLMAgent


if __name__ == "__main__":
    new_eval = True
    instruction_path = '../data/Command/target_command.json'

    if new_eval:
        env = MetricsVLLMEnv()
        agent = combineMultimodalLLMAgent()
        instru_num = instru_len(instruction_path)
        experiments = run_experiments(env, agent, instru_num)
    else:
        experiments = load_experiments('VLLM_results.json')

    metrics = get_parking_metrics(experiments)
    print("Metrics:", metrics)
