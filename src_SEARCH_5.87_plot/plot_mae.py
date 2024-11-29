# 作者：刘成广
# 时间：2024/11/13 下午1:56
import matplotlib.pyplot as plt
from IPython.display import clear_output

# 从文本文件读取列表
with open('test_mae_history_standards.txt', 'r') as f:
    test_mae_history1 = [float(line.strip()) for line in f.readlines()]

# 从文本文件读取列表
with open('test_mae_history_average.txt', 'r') as f:
    test_mae_history2 = [float(line.strip()) for line in f.readlines()]

# 初始化用于存储训练和测试 MAE 的列表
test_mae_plot1 = []
test_mae_plot2 = []

# 设置绘图窗口
plt.ion()  # 开启交互模式
fig, ax = plt.subplots(figsize=(10, 6))

for epoch in range(len(test_mae_history1)):
    # 清除之前的图像
    clear_output(wait=True)
    # 更新图像
    ax.clear()
    try:
        test_mae1 = test_mae_history1[epoch]
        # 将当前 epoch 的 MAE 添加到列表中
        test_mae_plot1.append(test_mae1)

    except:
        pass
    try:
        test_mae2 = test_mae_history2[epoch]
        test_mae_plot2.append(test_mae2)
    except:
        pass

    ax.plot(test_mae_plot2, marker='s', linestyle='-', label="Equal Interval", color=(0.545, 0.333, 0.608),
            markersize=6, markeredgewidth=1.5, markerfacecolor='none', linewidth=0.5)
    ax.plot(test_mae_plot1, marker='o', linestyle='-', label="Standard Interval", color=(0.337, 0.667, 0.369),
            markersize=6, markeredgewidth=1.5, markerfacecolor='none', linewidth=0.5)
    ax.set_xlabel("Number of epochs", fontsize=18)
    ax.set_ylabel("Mean absolute error (MAE)", fontsize=18)
    ax.legend(fontsize=18)
    ax.grid(True)  # 启用网格

    # 暂停 0.1 秒以更新图形并继续训练
    plt.pause(0.0001)

plt.show()  # 最后显示完整图表

# 保存图形到文件
plt.savefig('mae_plot_MISA.png', dpi=300)  # 将图表保存为PNG文件，可以更改文件名和格式