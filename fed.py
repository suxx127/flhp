import argparse
import random

import numpy as np
import torch
import transformers
from server import Server


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)
    print('--------- random seed {} ---------'.format(seed))


def main(args):
    server = Server(args=args)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, default='llama3.2-1B'
        # llama3.2-1B
        # qw3-1.7B
        # mistral-7B
    )
    parser.add_argument(
        '--seed', type=int, default=42
    )
    parser.add_argument(
        "--dataset", type=str,
        default='20newsgroup'
        # 20newsgroup, `patient2trial`, 'trial_design'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128, help='Number of calibration samples.'
    )
    parser.add_argument(
        "--prune_method", type=str,
        default="o",  # choices=['FedMP', 'FedDSE', 'FedRolex', 'heteroFL', 'FedSPU']
    )
    parser.add_argument(
        '--sample_fraction', type=float, default=0.2,
        help='how many clients are sampled in each round')
    parser.add_argument(
        '--client_num', type=int, default=10,
        help='number of clients in a fedrate cluster'
    )
    parser.add_argument(
        '--lora', action='store_true'
    )
    parser.add_argument(
        '--lora_alpha', type=int, default=32,
        help='scaling factors for LoRA'
    )
    parser.add_argument(
        '--lora_rank', type=int, default=16,
        help='rank of matrices in LoRA'
    )
    parser.add_argument(
        '--comm_round', type=int, default=5,
        help='number of maximum communication round'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.0,
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
    )
    parser.add_argument(
        '--bs', type=int, default=1,
    )
    arguments = parser.parse_args()
    print(arguments)
    set_random_seed(arguments.seed)
    if 'design' in arguments.dataset:
        arguments.gen = True
    else:
        arguments.gen = False
    main(arguments)
