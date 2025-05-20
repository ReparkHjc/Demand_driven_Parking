import json
from avp_env.metrics.env_runner import get_result_id, get_rl_result_id
from avp_env.metrics.utils import get_target_features

def run_experiments(env, agent, instru_num, output_file, view):
    experiments = []
    with open(output_file, 'w') as json_file:
        json_file.write("[\n")

        for idx in range(instru_num):
            result_id, result_features = get_result_id(env, agent, view, idx)
            target_id, target_features = get_target_features(env)

            experiment = {
                "result_id": result_id,
                "target_id": target_id,
                "target_features": target_features,
                "result_features": result_features,
            }

            if idx > 0:
                json_file.write(",\n")
            json.dump(experiment, json_file, indent=4)
            experiments.append(experiment)
            print(f'parking at {result_id}, {idx+1}/{instru_num}')

        json_file.write("\n]")
    return experiments

def run_rl_experiments(env, agent, instru_num, output_file, view):
    experiments = []
    with open(output_file, 'w') as json_file:
        json_file.write("[\n")

        for idx in range(instru_num):
            result_id, result_features = get_rl_result_id(env, agent, view, idx)
            target_id, target_features = get_target_features(env)

            experiment = {
                "result_id": result_id,
                "target_id": target_id,
                "target_features": target_features,
                "result_features": result_features,
            }

            if idx > 0:
                json_file.write(",\n")
            json.dump(experiment, json_file, indent=4)
            experiments.append(experiment)
            print(f'parking at {result_id}, {idx+1}/{instru_num}')

        json_file.write("\n]")
    return experiments

def load_experiments(input_file):
    with open(input_file, 'r') as f:
        return json.load(f)
