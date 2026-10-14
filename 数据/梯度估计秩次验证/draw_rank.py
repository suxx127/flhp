import matplotlib.pyplot as plt
import numpy as np

# 您提供的数据
means = [0.531733333, 0.658733333, 0.690466667, 0.626966667, 0.698433333, 0.6032,
         0.706366667, 0.531733333, 0.753933333, 0.515866667, 0.5635, 0.7143,
         0.746033333, 0.7619, 0.880933333, 0.7857]

std_devs = [0.157152799, 0.168385695, 0.070099279, 0.12647836, 0.062467129, 0.198585196,
            0.059367631, 0.1847767, 0.095905729, 0.099720454, 0.176373637, 0.051413876,
            0.163044622, 0.140209867, 0.051460168, 0.033658283]

# 创建x轴数据点（假设是16个连续的数据点）
x_values = list(range(1, len(means) + 1))

# 计算置信区间（这里使用均值±标准差）
ci_lower = [means[i] - std_devs[i] for i in range(len(means))]
ci_upper = [means[i] + std_devs[i] for i in range(len(means))]

# 创建图形
plt.figure(figsize=(8, 4.5))

# 绘制置信区间（阴影区域）
plt.fill_between(x_values, ci_lower, ci_upper,
                 color='pink', alpha=0.5, label='±1 Std Dev')

# 绘制均值折线
plt.plot(x_values, means,
         color='red', marker='o', linewidth=2, markersize=6,
         markerfacecolor='red', markeredgecolor='darkred',
         label='Mean Value')

# 设置坐标轴
plt.xlim(0.5, len(means) + 0.5)
plt.ylim(0, 1.0)  # 因为您的数据都在0-1之间

# 设置刻度
plt.xticks(x_values)
plt.yticks([i * 0.1 for i in range(11)])
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tick_params(axis='both', which='minor', labelsize=18)

# 添加标签和标题
plt.xlabel('Layer index', fontsize=18)
plt.ylabel('SRC', fontsize=18)
# plt.title('Spearman rank correlation coefficient between estimated gradient and real gradient',
#           fontsize=24, fontweight='bold')

# 添加图例
plt.legend(loc='lower right',
           fontsize=18, frameon=True)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.3)

# 调整布局
plt.tight_layout()

# 显示图形

plt.savefig('plot_exp_rank.pdf', bbox_inches='tight')
plt.savefig('plot_exp_rank.svg', bbox_inches='tight')
plt.show()