import pandas as pd
import numpy as np

def generate_parabolic_data(vid, frame_id, t_start, num_points, dt):
    vx = np.random.uniform(5, 10)
    vy = np.random.uniform(10, 15)
    g = 9.81
    x0, y0 = np.random.uniform(6451100, 6451200), np.random.uniform(1873250, 1873300)

    times = np.arange(0, num_points * dt, dt)
    xs = x0 + vx * times
    ys = y0 + vy * times - 0.5 * g * times ** 2
    global_time = np.arange(t_start, t_start + num_points * dt * 1000, dt * 1000).astype(int)

    df = pd.DataFrame({
        "Vehicle_ID": [vid] * num_points,
        "Frame_ID": [frame_id] * num_points,
        "Global_X": xs,
        "Global_Y": ys,
        "Global_Time_New": global_time
    })

    return df


# === 构造多条轨迹 ===
trajectories = []
for i in range(5):  # 5 条轨迹示例
    df = generate_parabolic_data(
        vid=np.random.randint(10000, 999999),
        frame_id=739,
        t_start=70000 + i * 100,
        num_points=20,
        dt=0.1
    )
    trajectories.append(df)

# 合并并添加 Index 列
df_all = pd.concat(trajectories, ignore_index=True)
df_all.insert(0, "Index", range(len(df_all)))  # 在最前面插入 Index 列

# 保存为 CSV
df_all.to_csv("parabolic_trajectories_with_index.csv", index=False)
