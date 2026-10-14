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
#
# line_raw = Line2D([-1], [0], label='Uncompress', lw=3, color=colors[0])
# line_cvlc = Line2D([-1], [0], label='Fed-CVLC', lw=3, color=colors[1])
# line_ComPEFT = Line2D([-1], [0], label='ComPEFT', lw=3, color=colors[2])
# line_FLASC = Line2D([-1], [0], label='FLASC', lw=3, color=colors[3])
# line_smartidx = Line2D([-1], [0], label='SmartIdx', lw=3, color=colors[4])
# line_fedcom = Line2D([-1], [0], label='FedComp', lw=3, color=colors[5])
# line_block = Line2D([-1], [0], label='FedBAO(ours)', lw=3, color=colors[6])

# handles = [line_raw, line_cvlc, line_ComPEFT, line_FLASC, line_smartidx, line_fedcom, line_block]

# plt.figure(figsize=(60, 6))
fig, axs = plt.subplots(1, 6, figsize=(42, 7))
# plt.axis('off')

legend_elements = [
    # Line2D([0], [-2], marker=marker[0], markersize=12, label='FedDSE', lw=3, color=colors[0]),
    Line2D([0], [-2], marker=marker[1], markersize=12, label='FedRolex', lw=3, color=colors[1]),
    Line2D([0], [-2], marker=marker[2], markersize=12, label='FedMP', lw=3, color=colors[2]),
    Line2D([0], [-2], marker=marker[3], markersize=12, label='HeteroFL', lw=3, color=colors[3]),
    Line2D([0], [-2], marker=marker[4], markersize=12, label='ScaleFL', lw=3, color=colors[4]),
    Line2D([0], [-2], marker=marker[5], markersize=12, label='FedDSE', lw=3, color=colors[5]),
    Line2D([0], [-2], marker=marker[6], markersize=12, label='FL-HP(ours)', lw=3, color=colors[6]),
]

# data_1B_00_0 = get_file('/full/1b-0.0-0.8/acc_FedDSE_5_llama3.2-1B_0.0.txt')
data_1B_00_1 = get_file('./1b-iid/acc_FedRolex_5_llama3.2-1B_0.0.txt')
data_1B_00_2 = get_file('./1b-iid/acc_FedMP_5_llama3.2-1B_0.0.txt')
data_1B_00_3 = get_file('./1b-iid/acc_HeteroFL_5_llama3.2-1B_0.0.txt')
data_1B_00_4 = get_file('./1b-iid/acc_ScaleFL_5_llama3.2-1B_0.0.txt')
data_1B_00_5 = get_file('./1b-iid/acc_FedDSE_5_llama3.2-1B_0.0.txt')
data_1B_00_6 = get_file('./1b-iid/acc_o_5_llama3.2-1B_0.0.txt')

step = 10
x0 = np.linspace(0, 5, 6)
# axs[0].plot(x0, data_distill_0[:600:step], color=colors[0], linewidth=l_w)
axs[0].plot(x0, data_1B_00_1, color=colors[1], linewidth=l_w)
axs[0].plot(x0, data_1B_00_2, color=colors[2], linewidth=l_w)
axs[0].plot(x0, data_1B_00_3, color=colors[3], linewidth=l_w)
axs[0].plot(x0, data_1B_00_4, color=colors[4], linewidth=l_w)
axs[0].plot(x0, data_1B_00_5, color=colors[5], linewidth=l_w)
axs[0].plot(x0, data_1B_00_6, color=colors[6], linewidth=l_w)
axs[0].set_yticks(np.arange(0.20, 0.72, step=0.10))

# axs[0].scatter(x0, data_distill_0[:600:step], color=colors[0], s=s_size, marker=marker[0])
axs[0].scatter(x0, data_1B_00_1, color=colors[1], s=s_size, marker=marker[1])
axs[0].scatter(x0, data_1B_00_2, color=colors[2], s=s_size, marker=marker[2])
axs[0].scatter(x0, data_1B_00_3, color=colors[3], s=s_size, marker=marker[3])
axs[0].scatter(x0, data_1B_00_4, color=colors[4], s=s_size, marker=marker[4])
axs[0].scatter(x0, data_1B_00_5, color=colors[5], s=s_size, marker=marker[5])
axs[0].scatter(x0, data_1B_00_6, color=colors[6], s=s_size, marker=marker[6])

axs[0].set_ylabel('Accuracy', fontsize=xy_label_size)
axs[0].set_xlabel('Comm Round', fontsize=xy_label_size)

axs[0].text(0.5, -0.22, "a) LLaMa3.2-1B IID", fontsize=fs, ha='center', va='center',
            transform=axs[0].transAxes)

# data_distill_iid_0 = get_file(
#     './1b-0.0-0.8/acc_FedRolex_5_llama3.2-1B_0.0.txt')
data_1B_10_1 = get_file(
    './1b-1.0/acc_FedRolex_5_llama3.2-1B_1.0.txt')
data_1B_10_2 = get_file(
    './1b-1.0/acc_FedMP_5_llama3.2-1B_1.0.txt')
data_1B_10_3 = get_file(
    './1b-1.0/acc_HeteroFL_5_llama3.2-1B_1.0.txt')
data_1B_10_4 = get_file(
    './1b-1.0/acc_ScaleFL_5_llama3.2-1B_1.0.txt')
data_1B_10_5 = get_file(
    './1b-1.0/acc_FedDSE_5_llama3.2-1B_1.0.txt')
data_1B_10_6 = get_file(
    './1b-1.0/acc_o_5_llama3.2-1B_1.0.txt')

x1 = np.linspace(0, 5, 6)
axs[1].plot(x1, data_1B_10_1, color=colors[1], linewidth=l_w)
axs[1].plot(x1, data_1B_10_2, color=colors[2], linewidth=l_w)
axs[1].plot(x1, data_1B_10_3, color=colors[3], linewidth=l_w)
axs[1].plot(x1, data_1B_10_4, color=colors[4], linewidth=l_w)
axs[1].plot(x1, data_1B_10_5, color=colors[5], linewidth=l_w)
axs[1].plot(x1, data_1B_10_6, color=colors[6], linewidth=l_w)


axs[1].scatter(x1, data_1B_10_1, color=colors[1], s=s_size, marker=marker[1])
axs[1].scatter(x1, data_1B_10_2, color=colors[2], s=s_size, marker=marker[2])
axs[1].scatter(x1, data_1B_10_3, color=colors[3], s=s_size, marker=marker[3])
axs[1].scatter(x1, data_1B_10_4, color=colors[4], s=s_size, marker=marker[4])
axs[1].scatter(x1, data_1B_10_5, color=colors[5], s=s_size, marker=marker[5])
axs[1].scatter(x1, data_1B_10_6, color=colors[6], s=s_size, marker=marker[6])

axs[1].set_xlabel('Comm Round', fontsize=xy_label_size)
# axs[1].set_xticks(np.arange(0, 420, step=100))
axs[1].text(0.5, -0.22, "b) LLaMa3.2-1B non-IID(1.0)", fontsize=fs, ha='center', va='center',
            transform=axs[1].transAxes)

data_1B_01_1 = get_file('./1b-0.1/acc_FedRolex_10_llama3.2-1B_0.1.txt')
data_1B_01_2 = get_file('./1b-0.1/acc_FedMP_10_llama3.2-1B_0.1.txt')
data_1B_01_3 = get_file('./1b-0.1/acc_HeteroFL_10_llama3.2-1B_0.1.txt')
data_1B_01_4 = get_file('./1b-0.1/acc_ScaleFL_10_llama3.2-1B_0.1.txt')
data_1B_01_5 = get_file('./1b-0.1/acc_FedDSE_10_llama3.2-1B_0.1.txt')
data_1B_01_6 = get_file('./1b-0.1/acc_o_10_llama3.2-1B_0.1.txt')

x2 = np.linspace(0, 10, 11)
axs[2].plot(x2, data_1B_01_1, color=colors[1], linewidth=l_w)
axs[2].plot(x2, data_1B_01_2, color=colors[2], linewidth=l_w)
axs[2].plot(x2, data_1B_01_3, color=colors[3], linewidth=l_w)
axs[2].plot(x2, data_1B_01_4, color=colors[4], linewidth=l_w)
axs[2].plot(x2, data_1B_01_5, color=colors[5], linewidth=l_w)
axs[2].plot(x2, data_1B_01_6, color=colors[6], linewidth=l_w)
axs[2].set_xlabel('Comm Round', fontsize=xy_label_size)

# axs[2].scatter(x2, data_roberta_0[:100:step], color=colors[0], s=s_size, marker=marker[0])
axs[2].scatter(x2, data_1B_01_1, color=colors[1], s=s_size, marker=marker[1])
axs[2].scatter(x2, data_1B_01_2, color=colors[2], s=s_size, marker=marker[2])
axs[2].scatter(x2, data_1B_01_3, color=colors[3], s=s_size, marker=marker[3])
axs[2].scatter(x2, data_1B_01_4, color=colors[4], s=s_size, marker=marker[4])
axs[2].scatter(x2, data_1B_01_5, color=colors[5], s=s_size, marker=marker[5])
axs[2].scatter(x2, data_1B_01_6, color=colors[6], s=s_size, marker=marker[6])
# axs[2].set_xticks(np.arange(0, 120, step=20))
axs[2].text(0.5, -0.22, "c) LLaMa3.2-1B non-IID(0.1)", fontsize=fs, ha='center', va='center',
            transform=axs[2].transAxes)

plt.subplot(1, 6, 4)
# data_roberta_iid_0 = get_file('./1b-0.0-0.8/acc_FedRolex_5_llama3.2-1B_0.0.txt')
data_3B_00_1 = get_file('./3b-iid/acc_FedRolex_5_llama3.2-3B_0.0.txt')
data_3B_00_2 = get_file('./3b-iid/acc_FedMP_5_llama3.2-3B_0.0.txt')
data_3B_00_3 = get_file('./3b-iid/acc_HeteroFL_5_llama3.2-3B_0.0.txt')
data_3B_00_4 = get_file('./3b-iid/acc_ScaleFL_5_llama3.2-3B_0.0.txt')
data_3B_00_5 = get_file('./3b-iid/acc_FedDSE_5_llama3.2-3B_0.0.txt')
data_3B_00_6 = get_file('./3b-iid/acc_o_5_llama3.2-3B_0.0.txt')

x3 = np.linspace(0, 5, 6)
# axs[3].plot(x3, data_3B_00_1, color=colors[0], linewidth=l_w)
axs[3].plot(x3, data_3B_00_1, color=colors[1], linewidth=l_w)
axs[3].plot(x3, data_3B_00_2, color=colors[2], linewidth=l_w)
axs[3].plot(x3, data_3B_00_3, color=colors[3], linewidth=l_w)
axs[3].plot(x3, data_3B_00_4, color=colors[4], linewidth=l_w)
axs[3].plot(x3, data_3B_00_5, color=colors[5], linewidth=l_w)
axs[3].plot(x3, data_3B_00_6, color=colors[6], linewidth=l_w)

# axs[3].scatter(x3, data_3B_00_1, color=colors[0], s=s_size, marker=marker[0])
axs[3].scatter(x3, data_3B_00_1, color=colors[1], s=s_size, marker=marker[1])
axs[3].scatter(x3, data_3B_00_2, color=colors[2], s=s_size, marker=marker[2])
axs[3].scatter(x3, data_3B_00_3, color=colors[3], s=s_size, marker=marker[3])
axs[3].scatter(x3, data_3B_00_4, color=colors[4], s=s_size, marker=marker[4])
axs[3].scatter(x3, data_3B_00_5, color=colors[5], s=s_size, marker=marker[5])
axs[3].scatter(x3, data_3B_00_6, color=colors[6], s=s_size, marker=marker[6])
# axs[3].set_xticks(np.arange(0, 55, step=10))
axs[3].set_yticks(np.arange(0.20, 0.72, step=0.10))
axs[3].set_xlabel('Comm Round', fontsize=xy_label_size)
axs[3].text(0.5, -0.22, "d) LLaMa3.2-3B IID", fontsize=fs, ha='center', va='center',
            transform=axs[3].transAxes)

# data_llama_0 = get_file('./llama/accuracy_raw_1_1.0_50_18_20_newsgroups_mrpc_llama-2-7B.txt')
data_3B_10_1 = get_file('./3b-1.0/acc_FedRolex_5_llama3.2-3B_1.0.txt')
data_3B_10_2 = get_file('./3b-1.0/acc_FedMP_5_llama3.2-3B_1.0.txt')
data_3B_10_3 = get_file('./3b-1.0/acc_HeteroFL_5_llama3.2-3B_1.0.txt')
data_3B_10_4 = get_file('./3b-1.0/acc_ScaleFL_5_llama3.2-3B_1.0.txt')
data_3B_10_5 = get_file('./3b-1.0/acc_FedDSE_5_llama3.2-3B_1.0.txt')
data_3B_10_6 = get_file('./3b-1.0/acc_o_5_llama3.2-3B_1.0.txt')

x4 = np.linspace(0, 5, 6)
# axs[4].plot(x4, data_3B_10_1, color=colors[0], linewidth=l_w)
axs[4].plot(x4, data_3B_10_1, color=colors[1], linewidth=l_w)
axs[4].plot(x4, data_3B_10_2, color=colors[2], linewidth=l_w)
axs[4].plot(x4, data_3B_10_3, color=colors[3], linewidth=l_w)
axs[4].plot(x4, data_3B_10_4, color=colors[4], linewidth=l_w)
axs[4].plot(x4, data_3B_10_5, color=colors[5], linewidth=l_w)
axs[4].plot(x4, data_3B_10_6, color=colors[6], linewidth=l_w)
axs[4].set_xlabel('Comm Round', fontsize=xy_label_size)


# axs[4].scatter(x4, data_3B_10_1, color=colors[0], s=s_size, marker=marker[0])
axs[4].scatter(x4, data_3B_10_1, color=colors[1], s=s_size, marker=marker[1])
axs[4].scatter(x4, data_3B_10_2, color=colors[2], s=s_size, marker=marker[2])
axs[4].scatter(x4, data_3B_10_3, color=colors[3], s=s_size, marker=marker[3])
axs[4].scatter(x4, data_3B_10_4, color=colors[4], s=s_size, marker=marker[4])
axs[4].scatter(x4, data_3B_10_5, color=colors[5], s=s_size, marker=marker[5])
axs[4].scatter(x4, data_3B_10_6, color=colors[6], s=s_size, marker=marker[6])
# axs[4].set_xticks(np.arange(0, 33, step=5))
axs[4].text(0.5, -0.22, "e) LLaMa3.2-3B non-IID(1.0)", fontsize=fs, ha='center', va='center',
            transform=axs[4].transAxes)
axs[4].set_yticks(np.arange(0.1, 0.71, step=0.1))

# data_3B_01_1 = get_file('./llama/iid/accuracy_raw_1024_1.0_20_18_20_newsgroups_mrpc_llama-2-7B.txt')
data_3B_01_1 = get_file('./3b-0.1/acc_FedRolex_10_llama3.2-3B_0.1.txt')
data_3B_01_2 = get_file('./3b-0.1/acc_FedMP_10_llama3.2-3B_0.1.txt')
data_3B_01_3 = get_file('./3b-0.1/acc_HeteroFL_10_llama3.2-3B_0.1.txt')
data_3B_01_4 = get_file('./3b-0.1/acc_ScaleFL_10_llama3.2-3B_0.1.txt')
data_3B_01_5 = get_file('./3b-0.1/acc_FedDSE_10_llama3.2-3B_0.1.txt')
data_3B_01_6 = get_file('./3b-0.1/acc_o_10_llama3.2-3B_0.1.txt')

x5 = np.linspace(0, 10, 11)

# axs[5].plot(x5, data_llama_iid_0, color=colors[0], linewidth=l_w)
axs[5].plot(x5, data_3B_01_1, color=colors[1], linewidth=l_w)
axs[5].plot(x5, data_3B_01_2, color=colors[2], linewidth=l_w)
axs[5].plot(x5, data_3B_01_3, color=colors[3], linewidth=l_w)
axs[5].plot(x5, data_3B_01_4, color=colors[4], linewidth=l_w)
axs[5].plot(x5, data_3B_01_5, color=colors[5], linewidth=l_w)
axs[5].plot(x5, data_3B_01_6, color=colors[6], linewidth=l_w)


# axs[5].scatter(x5, data_3B_01_1, color=colors[0], s=s_size, marker=marker[0])
axs[5].scatter(x5, data_3B_01_1, color=colors[1], s=s_size, marker=marker[1])
axs[5].scatter(x5, data_3B_01_2, color=colors[2], s=s_size, marker=marker[2])
axs[5].scatter(x5, data_3B_01_3, color=colors[3], s=s_size, marker=marker[3])
axs[5].scatter(x5, data_3B_01_4, color=colors[4], s=s_size, marker=marker[4])
axs[5].scatter(x5, data_3B_01_5, color=colors[5], s=s_size, marker=marker[5])
axs[5].scatter(x5, data_3B_01_6, color=colors[6], s=s_size, marker=marker[6])

# axs[5].set_xticks(np.arange(0, 21, step=4))
axs[5].set_xlabel('Comm Round', fontsize=xy_label_size)

axs[5].text(0.5, -0.22, "f) LLaMa3.2-3B non-IID(0.1)", fontsize=fs, ha='center', va='center',
            transform=axs[5].transAxes)
# plt.tight_layout()
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)
fig.legend(handles=legend_elements, ncol=7, loc='upper center', fontsize=28,
           bbox_to_anchor=(0.5, 1.03), bbox_transform=plt.gcf().transFigure)
plt.savefig('plot_exp1.pdf', bbox_inches='tight')
plt.savefig('plot_exp1.jpg', bbox_inches='tight')
# plt.show()
