import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def estimate_curvature(xy):
    """计算平均角度变化来衡量“曲率”，越小越直"""
    deltas = np.diff(xy, axis=0)
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    directions = deltas / (norms + 1e-8)

    angle_changes = np.arccos(np.clip(np.sum(directions[1:] * directions[:-1], axis=1), -1.0, 1.0))
    mean_angle_change = np.mean(angle_changes)
    return mean_angle_change

def analyze_trajectory_straightness(csv_path, angle_thresh_deg=2.0):
    df = pd.read_csv(csv_path)
    grouped = df.groupby("Vehicle_ID")

    straight_count = 0
    curve_count = 0
    straight_ids = []
    curved_ids = []

    for vid, group in grouped:
        group = group.sort_values("Global_Time_New")
        xy = group[["Global_X", "Global_Y"]].values
        if len(xy) < 25:
            continue
        curvature = estimate_curvature(xy)
        if curvature < np.deg2rad(angle_thresh_deg):  # threshold in radians
            straight_count += 1
            straight_ids.append(vid)
        else:
            curve_count += 1
            curved_ids.append(vid)

    print(f"✅ 近似直线轨迹: {straight_count}")
    print(f"✅ 曲线轨迹: {curve_count}")
    print(f"➡️ 直线比例: {straight_count / (straight_count + curve_count):.2%}")

    return straight_ids, curved_ids

# 示例调用
if __name__ == '__main__':
    csv_path = "Generated_Synthetic_Trajectories.csv"
    straight_ids, curved_ids = analyze_trajectory_straightness(csv_path)
