import pandas as pd
import numpy as np


# def create_dateset(trajectory, look_back=20, pred_len=5):
#     dataX, dataY = [], []
#     for i in range(len(trajectory) - look_back - pred_len):
#         a = trajectory[i: (i+look_back), :]
#         dataX.append(a)
#
#         input_end = trajectory[i + look_back - 1, 1:3]
#         future = trajectory[(i + look_back):(i + look_back + pred_len), 1:3]
#         delta = future - input_end
#         dataY.append(delta)
#
#     return np.array(dataX), np.array(dataY)

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
            y = np.zeros_like(y)  # 或者 y = y * 0.0
        else:
            y = (y - np.min(y)) / scalar_y
        seq = np.arange(len(local_xy)) / (len(local_xy) - 1)

        norm_traj = np.stack([seq, x, y], axis=1)

        for i in range(len(norm_traj) - look_back - pred_len):
            a = norm_traj[i: (i + look_back), :]
            input_end = norm_traj[i + look_back - 1, 1:3]
            future = norm_traj[(i + look_back):(i + look_back + pred_len), 1:3]
            delta = future - input_end
            dataX.append(a)
            dataY.append(delta)

    return np.array(dataX), np.array(dataY)



def load_data(data_size=None, look_back=20, pred_len=5):
    # 这个是多种轨迹的
    # df = pd.read_csv("../Generated_Synthetic_Trajectories.csv")

    # 这个是只有s 曲线
    df = pd.read_csv("../50_S_Curve.csv")

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
        train_Y.reshape(-1, pred_len * 2),
        test_X.reshape(-1, look_back, 3),
        test_Y.reshape(-1, pred_len * 2)
    )

# def load_data(data_size, look_back=20, pred_len=5):
#     data_csv = pd.read_csv("../Generated_Synthetic_Trajectories.csv")
#     trajectory1 = np.array(data_csv, dtype=np.float64)
#     if data_size is None or data_size > len(trajectory1):
#         data_size = len(trajectory1)
#
#     local_xy = trajectory1[0:data_size, 3:5]
#     x = local_xy[:, 0]
#     y = local_xy[:, 1]
#     scalar_x = np.max(x) - np.min(x)
#     x = ((x - np.min(x)) / scalar_x) + 0.001
#     scalar_y = np.max(y) - np.min(y)
#     y = (y - np.min(y)) / scalar_y
#
#     seq = np.arange(data_size)
#     seq = (seq - np.min(seq)) / (np.max(seq) - np.min(seq))
#
#     trajectory = np.hstack([seq.reshape(-1, 1), x.reshape(-1, 1), y.reshape(-1, 1)])
#
#     # ✅ 把参数传进去
#     data_X, data_Y = create_dateset(trajectory, look_back, pred_len)
#
#     # 划分数据集
#     train_size = int(len(data_X) * 0.7)
#     train_X = data_X[:train_size]
#     train_Y = data_Y[:train_size]
#     test_X = data_X[train_size:]
#     test_Y = data_Y[train_size:]
#
#     # ✅ 正确 reshape
#     train_X = train_X.reshape(-1, look_back, 3)
#     train_Y = train_Y.reshape(-1, pred_len * 2)
#     test_X = test_X.reshape(-1, look_back, 3)
#     test_Y = test_Y.reshape(-1, pred_len * 2)
#
#     return train_X, train_Y, test_X, test_Y


# def load_data(data_size):
#
#     # data_csv = pd.read_csv("../NGSIM/NGSIM_trajectories_data/trajectories1.csv")
#     data_csv = pd.read_csv("../Generated_Synthetic_Trajectories.csv")
#
#     trajectory1 = np.array(data_csv, dtype=np.float64)  # trajectory1[:, 5:7]
#
#     if data_size is None or data_size > len(trajectory1):
#         data_size = len(trajectory1)
#
#     local_xy = trajectory1[0:data_size, 3:5]
#     # local_xy = trajectory1[0:data_size, 5:7]
#
#     # 取前400个数据作为预测对象
#
#     x = local_xy[:, 0].tolist()
#     y = local_xy[:, 1].tolist()
#
#     # x归一化
#     # print(np.max(x))
#     # print(np.min(x))
#     scalar_x = np.max(x) - np.min(x)
#     x = ((x - np.min(x)) / scalar_x) + 0.001
#     # print(x)
#     # y归一化
#     scalar_y = np.max(y) - np.min(y)
#     y = (y - np.min(y)) / scalar_y
#
#     # 构建数据集，根据前10个轨迹估计后是个轨迹
#     x = x.reshape(data_size, 1)
#     y = np.array(y).reshape(data_size, 1)
#     seq = np.arange(data_size)
#     scalar_seq = np.max(seq) - np.min(seq)
#     seq = (seq - np.min(seq)) / scalar_seq
#     seq = seq.reshape(data_size, 1)
#     trajectory = np.hstack([seq, x, y])  # 三维数据， 1维是序号
#     # print(trajectory)  # 生成二维轨迹
#
#     data_X, data_Y = create_dateset(trajectory)
#     # 划分训练集和测试集，7/3
#     train_size = int(len(data_X) * 0.7)
#     test_size = len(data_X) - train_size
#     train_X = data_X[:train_size]
#     train_Y = data_Y[:train_size]
#     test_X = data_X[train_size:]
#     test_Y = data_Y[train_size:]
#
#     train_X = train_X.reshape(-1, 10, 3)
#
#     # TODO
#     # train_Y = train_Y.reshape(-1, 30)
#     train_Y = train_Y.reshape(-1, 20)
#
#     test_X = test_X.reshape(-1, 10, 3)
#     # test_Y = test_X.reshape(-1, 30)
#     # 得到280个训练集， 120个测试集
#
#     # TODO
#     return train_X, train_Y, test_X, test_Y
#     # return train_X, train_Y, test_X

