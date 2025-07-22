import pandas as pd
import numpy as np


def create_dataset(df, look_back=20, pred_len=5):
    dataX, dataY = [], []

    grouped = df.groupby("Vehicle_ID")
    for _, group in grouped:
        group = group.sort_values(by="Global_Time_New")  # 确保时间顺序
        local_xy = group[["Global_X", "Global_Y"]].values
        if len(local_xy) < look_back + pred_len:
            continue

        # 归一化
        x = local_xy[:, 0]
        y = local_xy[:, 1]
        scalar_x = np.max(x) - np.min(x)
        if scalar_x == 0:
            x = np.zeros_like(x) + 0.001  # 避免除0
        else:
            x = ((x - np.min(x)) / scalar_x) + 0.001
        scalar_y = np.max(y) - np.min(y)
        if scalar_y == 0:
            y = np.zeros_like(y)
        else:
            y = (y - np.min(y)) / scalar_y
        seq = np.arange(len(local_xy)) / (len(local_xy) - 1)

        norm_traj = np.stack([seq, x, y], axis=1)

        # ✅ 改为预测绝对位置（不再是 delta）
        for i in range(len(norm_traj) - look_back - pred_len):
            a = norm_traj[i: (i + look_back), :]
            future = norm_traj[(i + look_back):(i + look_back + pred_len), 1:3]  # 直接用绝对坐标
            dataX.append(a)
            dataY.append(future)

    return np.array(dataX), np.array(dataY)


def load_data(data_size=None, look_back=20, pred_len=5):
    df = pd.read_csv("../Generated_Synthetic_Trajectories.csv")
    if data_size:
        df = df.iloc[:data_size]

    data_X, data_Y = create_dataset(df, look_back, pred_len)

    # 划分数据集
    train_size = int(len(data_X) * 0.7)
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]

    return (
        train_X.reshape(-1, look_back, 3),
        train_Y.reshape(-1, pred_len * 2),  # 每步2个坐标
        test_X.reshape(-1, look_back, 3),
        test_Y.reshape(-1, pred_len * 2)
    )
