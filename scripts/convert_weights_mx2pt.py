import argparse
from pathlib import Path
from collections import OrderedDict
import mxnet as mx
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='The path to the MXNet checkpoint.')

    return parser.parse_args()


def main():
    args = parse_args()

    mx_weights = mx.nd.load(args.checkpoint)

    pt_weights = OrderedDict()
    for weight_name, data in mx_weights.items():
        pt_weight = torch.tensor(data.asnumpy(), device='cuda:0')
        pt_name = convert_mx2pt(weight_name, mx_weights)
        pt_weights[pt_name] = pt_weight

    mx_weights_path = Path(args.checkpoint)
    pt_weights_path = mx_weights_path.parent / (mx_weights_path.stem + '.pth')
    with open(pt_weights_path, 'wb') as f:
        torch.save(pt_weights, f)
    print(f'Converted weights saved to {pt_weights_path}')


def convert_mx2pt(x, weights_names):
    if x.endswith('.beta') and x[:-4] + 'running_var' in weights_names:
        x = x.replace('.beta', '.bias')
    if x.endswith('.gamma') and x[:-5] + 'running_var' in weights_names:
        x = x.replace('.gamma', '.weight')

    return x


if __name__ == '__main__':
    main()
