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

fig, axs = plt.subplots(1, 1, figsize=(16, 5))

legend_elements = [
    Line2D([0], [-2], marker=marker[1], markersize=12, label='0.0', lw=3, color=colors[1]),
    Line2D([0], [-2], marker=marker[2], markersize=12, label='0.2', lw=3, color=colors[2]),
    Line2D([0], [-2], marker=marker[3], markersize=12, label='0.4', lw=3, color=colors[3]),
    Line2D([0], [-2], marker=marker[4], markersize=12, label='0.6', lw=3, color=colors[4]),
    Line2D([0], [-2], marker=marker[5], markersize=12, label='0.8', lw=3, color=colors[5]),
    Line2D([0], [-2], marker=marker[6], markersize=12, label='1.0', lw=3, color=colors[6]),
]
data_1B_h_1 = get_file('0.0.txt')
data_1B_h_2 = get_file('0.2.txt')
data_1B_h_3 = get_file('0.4.txt')
data_1B_h_4 = get_file('0.6.txt')
data_1B_h_5 = get_file('0.8.txt')
data_1B_h_6 = get_file('1.0.txt')

x0 = np.linspace(0, 10, 11)
# axs[0].plot(x0, data_distill_0[:600:step], color=colors[0], linewidth=l_w)
axs.plot(x0, data_1B_h_1, color=colors[1], linewidth=l_w)
axs.plot(x0, data_1B_h_2, color=colors[2], linewidth=l_w)
axs.plot(x0, data_1B_h_3, color=colors[3], linewidth=l_w)
axs.plot(x0, data_1B_h_4, color=colors[4], linewidth=l_w)
axs.plot(x0, data_1B_h_5, color=colors[5], linewidth=l_w)
axs.plot(x0, data_1B_h_6, color=colors[6], linewidth=l_w)
plt.ylim(0.37, 0.68)
axs.set_yticks(np.arange(0.40, 0.72, step=0.10))
axs.set_xticks(np.arange(0, 11, step=2), )
axs.scatter(x0, data_1B_h_1, color=colors[1], s=s_size, marker=marker[1])
axs.scatter(x0, data_1B_h_2, color=colors[2], s=s_size, marker=marker[2])
axs.scatter(x0, data_1B_h_3, color=colors[3], s=s_size, marker=marker[3])
axs.scatter(x0, data_1B_h_4, color=colors[4], s=s_size, marker=marker[4])
axs.scatter(x0, data_1B_h_5, color=colors[5], s=s_size, marker=marker[5])
axs.scatter(x0, data_1B_h_6, color=colors[6], s=s_size, marker=marker[6])
axs.tick_params(axis='x', labelsize=25)
# 设置y轴刻度字体大小
axs.tick_params(axis='y', labelsize=25)
axs.set_ylabel('Accuracy', fontsize=xy_label_size)
axs.set_xlabel('Comm Round', fontsize=xy_label_size)
# axs.text(0.5, -0.22, "a) Heterogeneous constraint ", fontsize=fs, ha='center', va='center',
#             transform=axs.transAxes)
fig.legend(handles=legend_elements, ncol=6, loc='upper center', fontsize=28,
           bbox_to_anchor=(0.5, 1.09), bbox_transform=plt.gcf().transFigure)
plt.savefig('plot_exp7.pdf', bbox_inches='tight')
plt.savefig('plot_exp7.svg', bbox_inches='tight')
