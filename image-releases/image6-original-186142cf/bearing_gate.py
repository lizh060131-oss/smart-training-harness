"""按轴承等权评估历史回放，为任务 B 的上线门控提供排序分数。"""
from collections import defaultdict
from statistics import fmean

import numpy as np


def bearing_replay_groups(replay_rows, historical_ids):
    """返回历史轴承在回放表中的索引，缺少设备身份时沿用基础门控。"""
    groups = defaultdict(list)
    for index, row in enumerate(replay_rows):
        if row["id"] not in historical_ids:
            continue
        meta = row.get("meta")
        bearing = meta.get("bearing") if isinstance(meta, dict) else None
        if not isinstance(bearing, str) or not bearing.strip():
            return {}
        groups[bearing].append(index)
    return dict(groups)


def bearing_group_rmse(predicted, expected, groups):
    """使用同一份回放预测，计算每根轴承的 RMSE。"""
    squared_error = np.square(np.asarray(predicted) - np.asarray(expected))
    return {bearing: float(np.sqrt(np.mean(squared_error[indices])))
            for bearing, indices in groups.items()}


def bearing_selection_score(candidate, adopted, baseline_groups):
    """对已通过原门控的候选，权衡当批增益与历史设备的增益。"""
    current = adopted["primary_val"]
    current_gain = (current - candidate["primary_val"]) / max(current, 1.0)
    history_gain = fmean(
        (adopted["bearing_rmse"][bearing] - candidate["bearing_rmse"][bearing]) / max(reference, 1.0)
        for bearing, reference in baseline_groups.items()
    )
    return 0.55 * current_gain + 0.45 * history_gain
