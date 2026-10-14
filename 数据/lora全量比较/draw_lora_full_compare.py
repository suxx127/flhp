import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams['pdf.fonttype'] = 42


def get_file(fp):
    data = []
    with open(fp, 'r') as f:
        for line in f:
            value = float(line.strip())
            data.append(value)
    return data


fs = 28
colors = ['black', 'red', 'green', 'darkviolet', 'darkorange', 'cornflowerblue', 'magenta']
marker = ['*', 'X', '^', 'v', 'o', 's', 'p']
xy_label_size = 28
s_size = 170
l_w = 4

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

legend_elements = [
    Line2D([0], [-2], marker=marker[1], markersize=12, label='LLaMA3.2-1B+LoRA', lw=3, color=colors[1]),
    Line2D([0], [-2], marker=marker[2], markersize=12, label='LLaMA3.2-1B+full-tuning', lw=3, color=colors[2]),
    Line2D([0], [-2], marker=marker[5], markersize=12, label='LLaMA3.2-3B+LoRA', lw=3, color=colors[5]),
    Line2D([0], [-2], marker=marker[6], markersize=12, label='LLaMA3.2-1B+full-tuning', lw=3, color=colors[6]),
]

data_1B_h_1 = get_file('./acc_o_5_llama3.2-1B_1.0.txt')
data_1B_h_2 = get_file('./full/acc_o_5_llama3.2-1B_1.0.txt')
data_1B_h_3 = get_file('./acc_o_5_llama3.2-3B_1.0.txt')
data_1B_h_4 = get_file('./full/acc_o_5_llama3.2-3B_1.0.txt')

x0 = np.linspace(0, 5, 6)
# axs[0].plot(x0, data_distill_0[:600:step], color=colors[0], linewidth=l_w)
axs[0].plot(x0, data_1B_h_1, color=colors[1], linewidth=l_w)
axs[0].plot(x0, data_1B_h_2, color=colors[2], linewidth=l_w)
axs[0].plot(x0, data_1B_h_3, color=colors[5], linewidth=l_w)
axs[0].plot(x0, data_1B_h_4, color=colors[6], linewidth=l_w)

axs[0].scatter(x0, data_1B_h_1, color=colors[1], s=s_size, marker=marker[1])
axs[0].scatter(x0, data_1B_h_2, color=colors[2], s=s_size, marker=marker[2])
axs[0].scatter(x0, data_1B_h_3, color=colors[5], s=s_size, marker=marker[5])
axs[0].scatter(x0, data_1B_h_4, color=colors[6], s=s_size, marker=marker[6])

axs[0].set_ylabel('Accuracy', fontsize=xy_label_size)
axs[0].set_xlabel('Comm Round', fontsize=xy_label_size)
axs[0].tick_params(axis='x', labelsize=15)
axs[0].tick_params(axis='y', labelsize=15)
axs[0].text(0.5, -0.22, "a) Accuracy comparison", fontsize=fs, ha='center', va='center',
            transform=axs[0].transAxes)


memory = [7.8, 19.5, 18.9, 48]
objects = ['1B+LoRA', '1B+full', '3B+LoRA', '3B+full']
colors_list = [colors[1], colors[2], colors[5], colors[6]]
bars = axs[1].bar(objects, memory,
           color=colors_list,
           alpha=0.85,
           edgecolor='black',
           linewidth=1.5)
for bar in bars:
    height = bar.get_height()  # 获取柱子的高度（数据值）
    axs[1].text(
        bar.get_x() + bar.get_width() / 2,  # X坐标：柱子中心位置
        height + 0.5,                       # Y坐标：柱子顶部上方偏移
        f'{height:.1f}',                    # 显示带1位小数的数值
        ha='center',                        # 水平居中
        va='bottom',                        # 垂直底部对齐
        fontsize=19,                         # 字体大小
        fontweight='bold'                   # 可选：加粗字体
    )
axs[1].set_yticks(np.arange(0, 61, step=10))
axs[1].set_ylabel('Memory/GB', fontsize=xy_label_size)
axs[1].text(0.5, -0.22, "b) Memory comparison", fontsize=fs, ha='center', va='center',
            transform=axs[1].transAxes)
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
fig.legend(handles=legend_elements, ncol=2, loc='upper center', fontsize=20,
           bbox_to_anchor=(0.5, 1.05), bbox_transform=plt.gcf().transFigure)
# axs[1].set_xlabel('Memory', fontsize=xy_label_size)
plt.savefig('plot_exp4.pdf', bbox_inches='tight', dpi=500)
plt.savefig('plot_exp4.svg', bbox_inches='tight')
