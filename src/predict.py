import pandas as pd
import numpy as np
from keras.models import load_model
import data as dt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    look_back = 20
    pred_len = 5
    train_x, train_y, test_x, test_y_true = dt.load_data(data_size=1500, look_back=look_back, pred_len=pred_len)
    model = load_model("model_new/eps_100_bs_16_dp_0.2(256).h5")

    # 模型预测
    # 自动推断 pred_len
    test_y = model.predict(test_x)
    assert test_y.shape[1] % 2 == 0, "预测维度应为2的倍数"
    pred_len = test_y.shape[1] // 2

    test_y = test_y.reshape(-1, pred_len, 2)  # Δx, Δy
    test_y_true = test_y_true.reshape(-1, pred_len, 2)
    last_point = test_x[:, -1, 1:3]
    pred_absolute = last_point[:, np.newaxis, :] + np.cumsum(test_y, axis=1)
    gt_absolute = last_point[:, np.newaxis, :] + np.cumsum(test_y_true, axis=1)

    test_x_seq = test_x  # (N, 20, 3)
    test_x_flat = test_x_seq.reshape(-1, 3)
    pred_flat = pred_absolute.reshape(-1, 2)
    gt_flat = gt_absolute.reshape(-1, 2)

    print(f"test_y shape: {test_y.shape}")
    print(f"test_y_true shape: {test_y_true.shape}")
    print(f"pred_absolute shape: {pred_absolute.shape}")
    print(f"gt_absolute shape: {gt_absolute.shape}")
    print(f"test_x_seq shape: {test_x_seq.shape}")

    # TODO
    # test_y = model.predict(test_x)
    # test_y = test_y.reshape(-1, 10, 2)  # Δx, Δy
    # last_point = test_x[:, -1, 1:3]  # 每条输入的最后一个位置 x, y
    # pred_absolute = last_point[:, np.newaxis, :] + np.cumsum(test_y, axis=1)
    #
    # # 真实轨迹 ground truth
    # test_y_true = test_y_true.reshape(-1, 5, 2)  # ✅ reshape 保证维度一致
    # gt_absolute = last_point[:, np.newaxis, :] + np.cumsum(test_y_true, axis=1)  # ✅ 替换为 test_y_true
    #
    # # 对应的输入轨迹 reshape
    # test_x_seq = test_x.reshape(-1, 5, 3)
    #
    # # 保存为 CSV 可视化
    # test_x_flat = test_x_seq.reshape(-1, 3)
    # pred_flat = pred_absolute.reshape(-1, 2)
    # gt_flat = gt_absolute.reshape(-1, 2)
    #
    # print(f"test_y shape: {test_y.shape}")
    # print(f"test_y_true shape: {test_y_true.shape}")
    # print(f"pred_absolute shape: {pred_absolute.shape}")
    # print(f"gt_absolute shape: {gt_absolute.shape}")
    # print(f"test_x_seq shape: {test_x.reshape(-1, 10, 3).shape}")

    pd.DataFrame(test_x_flat, columns=['seq', 'Local_X', 'Local_Y']).to_csv('test_x.csv')
    pd.DataFrame(pred_flat, columns=['Local_X', 'Local_Y']).to_csv('pred_y.csv')
    pd.DataFrame(gt_flat, columns=['Local_X', 'Local_Y']).to_csv('gt_y.csv')

    # 可视化第 i 条轨迹的预测 vs 实际
    for i in range(30):  # 画前5条轨迹
        plt.figure(figsize=(6, 4))
        plt.plot(test_x_seq[i, :, 1], test_x_seq[i, :, 2], 'go-', label='Input Trajectory')
        plt.plot(gt_absolute[i, :, 0], gt_absolute[i, :, 1], 'bo--', label='Ground Truth')
        plt.plot(pred_absolute[i, :, 0], pred_absolute[i, :, 1], 'ro--', label='Predicted Trajectory')
        plt.title(f"Trajectory #{i}")
        plt.xlabel("Local_X")
        plt.ylabel("Local_Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    for i in range(5):
        print(f"\nSample {i}:")
        print(f"  Input trajectory points: {np.sum(~np.isnan(test_x_seq[i, :, 1:3]).all(axis=1))}")
        print(f"  Predicted trajectory points: {np.sum(~np.isnan(pred_absolute[i]).all(axis=1))}")
        print(f"  GT trajectory points: {np.sum(~np.isnan(gt_absolute[i]).all(axis=1))}")
        print(
            f"  Pred range: X({pred_absolute[i, :, 0].min():.3f}, {pred_absolute[i, :, 0].max():.3f}), Y({pred_absolute[i, :, 1].min():.3f}, {pred_absolute[i, :, 1].max():.3f})")