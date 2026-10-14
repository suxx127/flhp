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

fig, axs = plt.subplots(1, 1, figsize=(15, 4))

legend_elements = [
    Line2D([0], [-2], marker=marker[1], markersize=12, label='dynamic', lw=3, color=colors[1]),
    Line2D([0], [-2], marker=marker[2], markersize=12, label='ours', lw=3, color=colors[2]),
    Line2D([0], [-2], marker=marker[3], markersize=12, label='average', lw=3, color=colors[3]),
]
d_1 = get_file('./dynamic_layer.txt')
d_2 = get_file('./static_layer.txt')
d_3 = get_file('./local.txt')


x0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
plt.ylim(0.27, 0.75)
axs.plot(x0, d_1, color=colors[1], linewidth=l_w)
axs.plot(x0, d_2, color=colors[2], linewidth=l_w)
axs.plot(x0, d_3, color=colors[3], linewidth=l_w)

axs.scatter(x0, d_1, color=colors[1], s=s_size, marker=marker[1])
axs.scatter(x0, d_2, color=colors[2], s=s_size, marker=marker[2])
axs.scatter(x0, d_3, color=colors[3], s=s_size, marker=marker[3])

axs.set_ylabel('Accuracy', fontsize=xy_label_size)
axs.set_xlabel('Comm Round', fontsize=xy_label_size)

axs.set_yticks(np.arange(0.30, 0.72, step=0.10))
axs.set_xticks(np.arange(0, 16, step=3))
axs.tick_params(axis='both', which='major', labelsize=25)
axs.tick_params(axis='both', which='minor', labelsize=25)

leg = axs.legend(handles=legend_elements, loc='lower right',
                 fontsize=28, )
# 设置图例背景半透明，这样不会完全遮挡数据
leg.get_frame().set_alpha(0.5)

plt.savefig('plot_exp8.pdf', bbox_inches='tight')
plt.savefig('plot_exp8.svg', bbox_inches='tight')
