import torch
from tensorboardX import SummaryWriter

from data.dataloader import load_images
from model.distance import Distance
from utils.evaluator import evaluate_distance
from utils.randomness import set_randomness, worker_init_fn
from .datasets import *
from .opts import read_config

args = read_config()
GLOBAL_SEED = args.seed


def main():
    set_randomness(GLOBAL_SEED, deterministic=True, benchmark=False)
    # Prepare data
    datasets = data[args.dataset]
    K = num_class[args.dataset]
    print(args)
    train_target_file_path = datasets[args.tgt]
    train_source_file_path = datasets[args.src]
    batch_size = 16
    train_source_loader = load_images(train_source_file_path, batch_size=batch_size, resize_size=256, is_train=True, crop_size=224, worker_init=worker_init_fn, prefix=args.prefix)
    train_target_loader = load_images(train_target_file_path, batch_size=batch_size, resize_size=256, is_train=True, crop_size=224, worker_init=worker_init_fn, prefix=args.prefix)
    writer = SummaryWriter('%s/%s' % (args.logdir, args.key))
    # Init model
    target_model = torch.load(args.saved_model)
    model_instance = Distance(g_net=target_model.g_net, class_num=K, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    # Evaluate model
    result = evaluate_distance(model_instance, train_source_loader, train_target_loader)
    file_path = 'distance_test/%s.txt' % args.key
    with open(file_path, 'w') as result_file:
        result_file.writelines(result)
    print('write result into %s.' % file_path)


if __name__ == '__main__':
    main()
