import argparse


def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='runs')
    parser.add_argument('--src', type=str, default='runs')
    parser.add_argument('--tgt', type=str, default='runs')
    parser.add_argument('--srctest', type=str, default='runs')
    parser.add_argument('--tgttest', type=str, default='runs')
    parser.add_argument('--lr', type=float, default='0.001')
    parser.add_argument('--alpha', type=float, default='1')
    parser.add_argument('--beta', type=float, default='-1')
    parser.add_argument('--gamma', type=float, default='1')
    parser.add_argument('--key', type=str, default='taskname')
    parser.add_argument('--seed', type=int, default='2019')
    parser.add_argument('--k', type=int, default='5')
    parser.add_argument('--dataset', type=str, default='officehome')
    parser.add_argument('--method', type=str, default='source_only')
    parser.add_argument('--prefix', type=str, default='/data/office-home/images')
    parser.add_argument('--start_iter', type=int, default='0')
    parser.add_argument('--resume', type=int, default='0')
    parser.add_argument('--saved_model', type=str, default='')
    return parser.parse_args()
