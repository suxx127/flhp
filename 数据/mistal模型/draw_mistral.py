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
    # Line2D([0], [-2], marker=marker[0], markersize=12, label='FedDSE', lw=3, color=colors[0]),
    Line2D([0], [-2], marker=marker[1], markersize=12, label='FedRolex', lw=3, color=colors[1]),
    Line2D([0], [-2], marker=marker[2], markersize=12, label='FedMP', lw=3, color=colors[2]),
    Line2D([0], [-2], marker=marker[3], markersize=12, label='HeteroFL', lw=3, color=colors[3]),
    Line2D([0], [-2], marker=marker[4], markersize=12, label='ScaleFL', lw=3, color=colors[4]),
    Line2D([0], [-2], marker=marker[5], markersize=12, label='FedDSE', lw=3, color=colors[5]),
    Line2D([0], [-2], marker=marker[6], markersize=12, label='FL-HP(ours)', lw=3, color=colors[6]),
]

data_1B_h_1 = get_file('./acc_FedRolex_5_mistral-7B_1.0.txt')
data_1B_h_2 = get_file('./acc_FedMP_5_mistral-7B_1.0.txt')
data_1B_h_3 = get_file('./acc_HeteroFL_5_mistral-7B_1.0.txt')
data_1B_h_4 = get_file('./acc_ScaleFL_5_mistral-7B_1.0.txt')
data_1B_h_5 = get_file('./acc_FedDSE_5_mistral-7B_1.0.txt')
data_1B_h_6 = get_file('./acc_o_5_mistral-7B_1.0.txt')

step = 10
x0 = np.linspace(0, 5, 6)
# axs.plot(x0, data_distill_0[:600:step], color=colors[0], linewidth=l_w)
axs.plot(x0, data_1B_h_1, color=colors[1], linewidth=l_w)
axs.plot(x0, data_1B_h_2, color=colors[2], linewidth=l_w)
axs.plot(x0, data_1B_h_3, color=colors[3], linewidth=l_w)
axs.plot(x0, data_1B_h_4, color=colors[4], linewidth=l_w)
axs.plot(x0, data_1B_h_5, color=colors[5], linewidth=l_w)
axs.plot(x0, data_1B_h_6, color=colors[6], linewidth=l_w)
axs.set_yticks(np.arange(0.10, 0.72, step=0.10))
axs.set_xticks(np.arange(0, 6, step=1))
axs.scatter(x0, data_1B_h_1, color=colors[1], s=s_size, marker=marker[1])
axs.scatter(x0, data_1B_h_2, color=colors[2], s=s_size, marker=marker[2])
axs.scatter(x0, data_1B_h_3, color=colors[3], s=s_size, marker=marker[3])
axs.scatter(x0, data_1B_h_4, color=colors[4], s=s_size, marker=marker[4])
axs.scatter(x0, data_1B_h_5, color=colors[5], s=s_size, marker=marker[5])
axs.scatter(x0, data_1B_h_6, color=colors[6], s=s_size, marker=marker[6])

axs.set_ylabel('Accuracy', fontsize=xy_label_size)
axs.set_xlabel('Comm Round', fontsize=xy_label_size)


axs.tick_params(axis='both', which='major', labelsize=25)
axs.tick_params(axis='both', which='minor', labelsize=25)
fig.legend(handles=legend_elements, ncol=3, loc='upper center', fontsize=28,
           bbox_to_anchor=(0.5, 1.26), bbox_transform=plt.gcf().transFigure)
plt.savefig('plot_exp_mistral.pdf', bbox_inches='tight')

