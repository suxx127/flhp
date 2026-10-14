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

legend_elements1 = [
    Line2D([0], [-2], marker=marker[1], markersize=12, label='Wikitext2', lw=3, color=colors[1]),
    Line2D([0], [-2], marker=marker[2], markersize=12, label='C4', lw=3, color=colors[2]),
    Line2D([0], [-2], marker=marker[5], markersize=12, label='random data', lw=3, color=colors[5]),
    Line2D([0], [-2], marker=marker[6], markersize=12, label='20newsgroup', lw=3, color=colors[6]),
]

data_1B_h_1 = get_file('./数据集/wikitext.txt')
data_1B_h_2 = get_file('./数据集/c4.txt')
data_1B_h_3 = get_file('./数据集/random.txt')
data_1B_h_4 = get_file('./数据集/20newsgroup.txt')

x0 = np.linspace(0, 10, 11)
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
axs[0].text(0.5, -0.22, "a) calibdation-dataset", fontsize=fs, ha='center',
            va='center',
            transform=axs[0].transAxes)  # of different calibration datasets
axs[0].legend(handles=legend_elements1, fontsize=25)

data_1B_h_1 = get_file('./剪枝方法/our.txt')
data_1B_h_2 = get_file('./剪枝方法/Wanda-sp.txt')
data_1B_h_3 = get_file('./剪枝方法/random.txt')
data_1B_h_4 = get_file('./剪枝方法/FLAP.txt')

x0 = np.linspace(0, 10, 11)
# axs[0].plot(x0, data_distill_0[:600:step], color=colors[0], linewidth=l_w)
axs[1].plot(x0, data_1B_h_1, color=colors[1], linewidth=l_w)
axs[1].plot(x0, data_1B_h_2, color=colors[2], linewidth=l_w)
axs[1].plot(x0, data_1B_h_3, color=colors[5], linewidth=l_w)
axs[1].plot(x0, data_1B_h_4, color=colors[6], linewidth=l_w)

axs[1].scatter(x0, data_1B_h_1, color=colors[1], s=s_size, marker=marker[1])
axs[1].scatter(x0, data_1B_h_2, color=colors[2], s=s_size, marker=marker[2])
axs[1].scatter(x0, data_1B_h_3, color=colors[5], s=s_size, marker=marker[5])
axs[1].scatter(x0, data_1B_h_4, color=colors[6], s=s_size, marker=marker[6])

axs[1].set_ylabel('Accuracy', fontsize=xy_label_size)
axs[1].set_xlabel('Comm Round', fontsize=xy_label_size)
axs[1].tick_params(axis='x', labelsize=15)
axs[1].tick_params(axis='y', labelsize=15)
axs[1].text(0.5, -0.22, "b) method", fontsize=fs, ha='center', va='center',
            transform=axs[1].transAxes)  # of different pruning methods
legend_elements2 = [
    Line2D([0], [-2], marker=marker[1], markersize=12, label='ours', lw=3, color=colors[1]),
    Line2D([0], [-2], marker=marker[2], markersize=12, label='Wanda-sp', lw=3, color=colors[2]),
    Line2D([0], [-2], marker=marker[5], markersize=12, label='random', lw=3, color=colors[5]),
    Line2D([0], [-2], marker=marker[6], markersize=12, label='FLAP', lw=3, color=colors[6]),
]
axs[1].legend(handles=legend_elements2, fontsize=25)
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)

# axs[1].set_xlabel('Memory', fontsize=xy_label_size)
plt.savefig('plot_exp5.pdf', bbox_inches='tight')
plt.savefig('plot_exp5.svg', bbox_inches='tight')
