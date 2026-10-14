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
colors = ['black', 'red', 'green', 'purple', 'orange', 'blue', 'magenta']
marker = ['o', 'd', '^', 'X', '*', 's', 'p']
xy_label_size = 28
s_size = 150
l_w = 4

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

legend_elements = [
    # Line2D([0], [-2], marker=marker[0], markersize=12, label='FedDSE', lw=3, color=colors[0]),
    Line2D([0], [-2], marker=marker[1], markersize=12, label='FedRolex', lw=3, color=colors[1]),
    Line2D([0], [-2], marker=marker[2], markersize=12, label='FedMP', lw=3, color=colors[2]),
    Line2D([0], [-2], marker=marker[3], markersize=12, label='HeteroFL', lw=3, color=colors[3]),
    Line2D([0], [-2], marker=marker[4], markersize=12, label='ScaleFL', lw=3, color=colors[4]),
    Line2D([0], [-2], marker=marker[5], markersize=12, label='FedDSE', lw=3, color=colors[5]),
    Line2D([0], [-2], marker=marker[6], markersize=12, label='FL-HP(ours)', lw=3, color=colors[6]),
]

data_1B_h_1 = get_file('./50/acc_FedRolex_10_llama3.2-1B_0.0.txt')
data_1B_h_2 = get_file('./50/acc_FedMP_10_llama3.2-1B_0.0.txt')
data_1B_h_3 = get_file('./50/acc_HeteroFL_10_llama3.2-1B_0.0.txt')
data_1B_h_4 = get_file('./50/acc_ScaleFL_10_llama3.2-1B_0.0.txt')
data_1B_h_5 = get_file('./50/acc_FedDSE_10_llama3.2-1B_0.0.txt')
data_1B_h_6 = get_file('./50/acc_o_10_llama3.2-1B_0.0.txt')

step = 10
x0 = np.linspace(0, 10, 11)
# axs[0].plot(x0, data_distill_0[:600:step], color=colors[0], linewidth=l_w)
axs[0].plot(x0, data_1B_h_1, color=colors[1], linewidth=l_w)
axs[0].plot(x0, data_1B_h_2, color=colors[2], linewidth=l_w)
axs[0].plot(x0, data_1B_h_3, color=colors[3], linewidth=l_w)
axs[0].plot(x0, data_1B_h_4, color=colors[4], linewidth=l_w)
axs[0].plot(x0, data_1B_h_5, color=colors[5], linewidth=l_w)
axs[0].plot(x0, data_1B_h_6, color=colors[6], linewidth=l_w)
axs[0].set_yticks(np.arange(0.20, 0.72, step=0.10))
axs[0].set_xticks(np.arange(0, 11, step=2))
axs[0].scatter(x0, data_1B_h_1, color=colors[1], s=s_size, marker=marker[1])
axs[0].scatter(x0, data_1B_h_2, color=colors[2], s=s_size, marker=marker[2])
axs[0].scatter(x0, data_1B_h_3, color=colors[3], s=s_size, marker=marker[3])
axs[0].scatter(x0, data_1B_h_4, color=colors[4], s=s_size, marker=marker[4])
axs[0].scatter(x0, data_1B_h_5, color=colors[5], s=s_size, marker=marker[5])
axs[0].scatter(x0, data_1B_h_6, color=colors[6], s=s_size, marker=marker[6])

axs[0].set_ylabel('Accuracy', fontsize=xy_label_size)
axs[0].set_xlabel('Comm Round', fontsize=xy_label_size)
axs[0].text(0.5, -0.22, "a) 50 Clients", fontsize=fs, ha='center', va='center',
            transform=axs[0].transAxes)

data_1B_05_1 = get_file('./100/acc_FedRolex_10_llama3.2-1B_0.0.txt')
data_1B_05_2 = get_file('./100/acc_FedMP_10_llama3.2-1B_0.0.txt')
data_1B_05_3 = get_file('./100/acc_HeteroFL_10_llama3.2-1B_0.0.txt')
data_1B_05_4 = get_file('./100/acc_ScaleFL_10_llama3.2-1B_0.0.txt')
data_1B_05_5 = get_file('./100/acc_FedDSE_10_llama3.2-1B_0.0.txt')
data_1B_05_6 = get_file('./100/acc_o_10_llama3.2-1B_0.0.txt')

x2 = np.linspace(0, 10, 11)
axs[1].plot(x2, data_1B_05_1, color=colors[1], linewidth=l_w)
axs[1].plot(x2, data_1B_05_2, color=colors[2], linewidth=l_w)
axs[1].plot(x2, data_1B_05_3, color=colors[3], linewidth=l_w)
axs[1].plot(x2, data_1B_05_4, color=colors[4], linewidth=l_w)
axs[1].plot(x2, data_1B_05_5, color=colors[5], linewidth=l_w)
axs[1].plot(x2, data_1B_05_6, color=colors[6], linewidth=l_w)
axs[1].set_xlabel('Comm Round', fontsize=xy_label_size)

axs[1].scatter(x2, data_1B_05_1, color=colors[1], s=s_size, marker=marker[1])
axs[1].scatter(x2, data_1B_05_2, color=colors[2], s=s_size, marker=marker[2])
axs[1].scatter(x2, data_1B_05_3, color=colors[3], s=s_size, marker=marker[3])
axs[1].scatter(x2, data_1B_05_4, color=colors[4], s=s_size, marker=marker[4])
axs[1].scatter(x2, data_1B_05_5, color=colors[5], s=s_size, marker=marker[5])
axs[1].scatter(x2, data_1B_05_6, color=colors[6], s=s_size, marker=marker[6])

axs[1].text(0.5, -0.22, "b) 100 Clients", fontsize=fs, ha='center', va='center',
            transform=axs[1].transAxes)

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)
fig.legend(handles=legend_elements, ncol=3, loc='upper center', fontsize=28,
           bbox_to_anchor=(0.5, 1.13), bbox_transform=plt.gcf().transFigure)
plt.savefig('plot_exp9.pdf', bbox_inches='tight')
plt.savefig('plot_exp9.jpg', bbox_inches='tight')
