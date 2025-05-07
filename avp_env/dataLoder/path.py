class PathLoader:
    def __init__(self, env_type):
        self.env_type = env_type

    def load_path(self):
        if self.env_type == 'train':
            # experiment_paths = ['../data/Vision/20240518_01']
            experiment_paths = ['D:/1Epan/TJ_VLP_V2/data/Vision/Park_1/20250422']
        elif self.env_type == 'test':
            # experiment_paths = ['../data/Vision/20240521_01']
            experiment_paths = ['D:/1Epan/TJ_VLP_V2/data/Vision/Park_1/20250422']
        else:
            experiment_paths = []
        return experiment_paths
