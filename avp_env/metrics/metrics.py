from avp_env.metrics.utils import get_result_tags

def calculate_success_rate(experiments):
    """
    计算停车系统的成功率

    参数:
    experiments (list of dict): 每个字典包含 result_id 和 target_id，
                                例如: [{"result_id": 5, "target_id": 5}, {"result_id": 12, "target_id": 8}, ...]
    返回:
    float: 成功率（以百分比表示）
    """
    # 初始化成功次数
    success_count = 0
    error_count = 0
    # 总实验次数
    total_experiments = len(experiments)
    # 遍历每次实验的结果
    for experiment in experiments:
        result_id = experiment["result_id"]
        target_id = experiment["target_id"]
        # 检查系统决策的停车位是否与最佳停车位匹配
        if result_id and result_id in target_id:
            success_count += 1
        else:
            error_count += 1

    # 计算成功率
    success_rate = success_count / total_experiments

    # 返回成功率（百分比形式）
    return success_rate


def calculate_weighted_success_rate(experiments):
    """
       计算距离加权的停车系统成功率

       参数:
       experiments (list of dict): 每个字典包含 result_id, target_id 和 distance，
                                   例如: [{"result_id": 5, "target_id": 5, "distance": 10}, {"result_id": 12, "target_id": 8, "distance": 15}, ...]
       返回:
       float: 距离加权成功率（以百分比表示）
    """
    # 初始化成功次数
    weighted_success_sum = 0
    # 总实验次数
    total_experiments = len(experiments)
    # 遍历每次实验的结果
    for experiment in experiments:
        result_id = experiment["result_id"]
        target_id = experiment["target_id"]
        distance = experiment["result_features"]["distance"]

        # 检查系统决策的停车位是否与最佳停车位匹配
        if result_id and result_id in target_id:
            weighted_success_sum += 1 / distance

    # 计算成功率
    weighted_success_rate = weighted_success_sum / total_experiments

    # 返回成功率（百分比形式）
    return weighted_success_rate


def calculate_navigation_errors(experiments):
    """
    计算导航错误的次数

    参数:
    experiments (list of dict): 每个字典包含 result_id 和 target_id，
                                例如: [{"result_id": 5, "target_id": 5}, {"result_id": 12, "target_id": 8}, ...]


    返回:
    float: 导航错误的百分比
    """
    # 初始化错误次数
    error_count = 0
    # 总实验次数
    total_experiments = len(experiments)


    # 遍历每次实验的结果
    for experiment in experiments:
        if experiment["target_features"]:
            result_id = experiment["result_id"]
            result_path = experiment["target_features"][0]["scan"]
            result_park = experiment["target_features"][0]['park_id']

            if not result_id:
                continue
            result_tags = get_result_tags(result_id, result_park, result_path)
            if result_tags["Occupied"] != 0:
                error_count += 1



    # 计算导航错误率
    error_rate = error_count / total_experiments

    # 返回导航错误率（百分比形式）
    return error_rate


def calculate_absolute_parking_slot_error(experiments):
    """
    计算绝对车位误差指标（APSE）。

    参数:
    experiments (list of dict): 每个字典包含 result_id, target_ids 和 distance，
                                例如: [{"result_id": (path_id1, loc_id1), "target_ids": [(path_id2, loc_id2), ...]}, ...]

    返回:
    float: 绝对车位误差（APSE）
    """
    # 初始化误差和实验次数
    total_error = 0

    total_experiments = len(experiments)
    # 遍历每次实验的结果
    for experiment in experiments:
        target_positions = [target["distance"] for target in experiment["target_features"]]
        result_position = experiment["result_features"]["distance"]

        # 跳过没有决策车位或没有理想车位的实验
        if not result_position and target_positions:
            min_distance = min(abs((87 - target_position)/ 87) for target_position in target_positions)
        elif not target_positions and result_position:
            min_distance = min(abs((result_position - 87)/ 87), abs(result_position/ 87))
        elif not target_positions and not result_position:
            min_distance = 0
        elif target_positions and result_position:
            # 计算决策车位与每个理想车位的绝对距离，选择最近的距离
            min_distance = min(abs((result_position - target_position)/ 87) for target_position in target_positions)

        # 累加最小距离
        total_error += min_distance

        # # 跳过没有决策车位或没有理想车位的实验
        # if not result_position and target_positions:
        #     min_distance = 1
        # elif not target_positions and result_position:
        #     min_distance = 1
        # elif not target_positions and not result_position:
        #     min_distance = 0
        # elif target_positions and result_position:
        #     # 计算决策车位与每个理想车位的绝对距离，选择最近的距离
        #     min_distance = min(abs((result_position - target_position)/ 87) for target_position in target_positions)



    # 计算绝对车位误差
    apse = total_error / total_experiments

    return apse


def calculate_miss_rate(experiments):
    """
    计算错失率。

    参数:
    experiments (list of dict): 每个字典包含 result_id 和 target_ids，
                                例如: [{"result_id": (path_id1, loc_id1), "target_ids": [(path_id2, loc_id2), ...]}, ...]

    返回:
    float: 错失率
    """
    # 初始化错失总数和总指令数
    total_miss = 0
    total_experiments = len(experiments)

    # 遍历每次实验的结果
    for experiment in experiments:


        target_positions = [target["distance"] for target in experiment["target_features"]]
        result_position = experiment["result_features"]["distance"]

        # if not result_position and target_positions:
        #     total_miss += 1
        # elif not target_positions and result_position:
        #     total_miss += 0
        # elif not target_positions and not result_position:
        #     total_miss += 0
        # elif target_positions and result_position:
        #     targets_before_result = sum(1 for target_position in target_positions if target_position < result_position)
        #     equivalent_miss = targets_before_result / len(target_positions)
        #     # 累加等效错失车位
        #     total_miss += equivalent_miss
        #     total_experiments += 1

        if target_positions and result_position:
            targets_before_result = sum(1 for target_position in target_positions if target_position < result_position)
            # equivalent_miss = targets_before_result / len(target_positions)
            equivalent_miss = targets_before_result / len(target_positions)
            # 累加等效错失车位
            total_miss += equivalent_miss


    # 计算错失率
    miss_rate = total_miss / total_experiments if total_experiments > 0 else 0

    return miss_rate


def calculate_advance_rate(experiments):
    """
    计算提前率。

    参数:
    experiments (list of dict): 每个字典包含 result_id 和 target_ids，
                                例如: [{"result_id": (path_id1, loc_id1), "target_ids": [(path_id2, loc_id2), ...]}, ...]

    返回:
    float: 提前率
    """
    # 初始化提前总数和总指令数
    total_advance = 0
    total_experiments = len(experiments)

    # 遍历每次实验的结果
    for experiment in experiments:
        target_positions = [target["distance"] for target in experiment["target_features"]]
        result_position = experiment["result_features"]["distance"]
        if not result_position and target_positions:
            total_advance += 0
        elif not target_positions and result_position:
            total_advance += 1
        elif not target_positions and not result_position:
            total_advance += 0
            # 如果决策车位在所有的理想车位之前，等效提前车位为1
        elif all(result_position < target_position for target_position in target_positions):
            total_advance += 1

    # 计算提前率
    advance_rate = total_advance / total_experiments if total_experiments > 0 else 0

    return advance_rate


def calculate_matching_rate(experiments):
    """
    计算车位匹配度。

    参数:
    experiments (list of dict): 每个字典包含 result_id, target_ids, target_features，
                                例如: [{"result_id": (path_id1, loc_id1), "target_ids": [(path_id2, loc_id2), ...], "target_features": {"tag": {"tag1": value1, ...}}, "result_tags": {"tag1": value1, ...}}, ...]

    返回:
    float: 车位匹配度
    """
    total_matching_score = 0
    total_experiments = len(experiments)

    # 遍历每次实验的结果
    for experiment in experiments:
        if experiment["target_features"]:

            result_id = experiment["result_id"]
            result_path = experiment["target_features"][0]["scan"]
            result_park = experiment["target_features"][0]['park_id']

            if not result_id:
                continue
            target_tags = experiment["target_features"][0]['tags']
            result_tags = get_result_tags(result_id, result_park, result_path)
            # 跳过没有决策车位或没有用户指令标签的实验

            # 计算完全相同的tags数量
            matching_tags = 0
            for tag, value in target_tags.items():
                if result_tags.get(tag) == value:
                    matching_tags += 1

            # 用户指令包含的tags数量
            total_tags = len(target_tags)

            # 计算匹配度
            matching_score = matching_tags / total_tags if total_tags > 0 else 0



            # 累加匹配度
            total_matching_score += matching_score

    # 计算加权匹配度
    weighted_matching_rate = total_matching_score / total_experiments if total_experiments > 0 else 0

    return weighted_matching_rate

def calculate_weighted_matching_rate(experiments):
    """
    计算车位匹配度。

    参数:
    experiments (list of dict): 每个字典包含 result_id, target_ids, target_features，
                                例如: [{"result_id": (path_id1, loc_id1), "target_ids": [(path_id2, loc_id2), ...], "target_features": {"tag": {"tag1": value1, ...}}, "result_tags": {"tag1": value1, ...}}, ...]

    返回:
    float: 车位匹配度
    """
    total_matching_score = 0
    total_experiments = len(experiments)

    # 遍历每次实验的结果
    for experiment in experiments:
        if experiment["target_features"]:

            result_id = experiment["result_id"]
            result_path = experiment["target_features"][0]["scan"]
            result_park = experiment["target_features"][0]['park_id']
            if not result_id:
                continue
            target_tags = experiment["target_features"][0]['tags']
            result_tags = get_result_tags(result_id, result_park, result_path)
            # 跳过没有决策车位或没有用户指令标签的实验
            distance = experiment["result_features"]["distance"]

            # 计算完全相同的tags数量
            matching_tags = 0
            for tag, value in target_tags.items():
                if result_tags.get(tag) == value:
                    matching_tags += 1

            # 用户指令包含的tags数量
            total_tags = len(target_tags)

            # 计算匹配度
            matching_score = matching_tags / total_tags / distance if total_tags > 0 else 0

            # 累加匹配度
            total_matching_score += matching_score

    # 计算加权匹配度
    weighted_matching_rate = total_matching_score / total_experiments if total_experiments > 0 else 0

    return weighted_matching_rate

def calculate_distance(experiments):
    """

    """
    # 初始化错失总数和总指令数
    total_result_position = 0
    total_experiments = len(experiments)

    # 遍历每次实验的结果
    for experiment in experiments:

        result_position = experiment["result_features"]["distance"]


        # 累加等效错失车位
        total_result_position += result_position


    # 计算错失率
    result_rate = total_result_position / total_experiments if total_experiments > 0 else 0

    return result_rate


def get_parking_metrics(experiments):
    NE_Metrics = calculate_navigation_errors(experiments)
    SR_Metrics = calculate_success_rate(experiments)
    SRL_Metrics = calculate_weighted_success_rate(experiments)
    APSE_Metrics = calculate_absolute_parking_slot_error(experiments)
    MR_Metrics = calculate_miss_rate(experiments)
    CSR_Metrics = calculate_matching_rate(experiments)
    CSRL_Metrics = calculate_weighted_matching_rate(experiments)
    result_dis = calculate_distance(experiments)
    return NE_Metrics, SR_Metrics, SRL_Metrics, APSE_Metrics, MR_Metrics, CSR_Metrics, CSRL_Metrics, result_dis
    # CR   ,  SR  ,  TE, GDE  ,   MR ,   IGA,    _____,   DIS
