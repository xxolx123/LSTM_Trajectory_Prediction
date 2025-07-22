import pandas as pd
import matplotlib.pyplot as plt

# === 加载数据 ===
df = pd.read_csv("D:\Pycharm_project\LSTM_Trajectory_Prediction-main\Generated_Synthetic_Trajectories.csv")

# === 可选：按时间排序 ===
df = df.sort_values(by=["Vehicle_ID", "Global_Time_New"])

# === 可视化轨迹 ===
plt.figure(figsize=(10, 8))

# 遍历每个 Vehicle_ID，画出轨迹
for vid, group in df.groupby("Vehicle_ID"):
    if len(group) < 2:
        continue  # 忽略只有一个点的轨迹
    plt.plot(group["Global_X"], group["Global_Y"], marker='o', label=f"Vehicle {vid}")

plt.xlabel("Global X")
plt.ylabel("Global Y")
plt.title("Vehicle Trajectories")
plt.legend(fontsize="small", loc="upper right")
plt.axis('equal')  # 保持比例一致，防止变形
plt.grid(True)
plt.show()
