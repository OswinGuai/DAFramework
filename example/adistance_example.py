from tensorboardX import SummaryWriter
from torch import optim

from data.dataloader import load_images
from model.adistance import ADis
from utils.lr_scheduler import INVScheduler
from utils.randomness import set_randomness, worker_init_fn
from utils.trainer import train_da_evaluator
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
    model_instance = ADis(class_num=K, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    parameter_list = model_instance.get_parameter_list()
    optimizer_list = []
    for parameter in parameter_list:
        optimizer = optim.SGD(parameter, lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
        optimizer_list.append(optimizer)
    scheduler = INVScheduler(gamma=0.001, power=0.75, decay_rate=0.0002, init_lr=args.lr)
    train_da_evaluator(model_instance, train_source_loader, train_target_loader, num_iterations=100000, optimizer_list=optimizer_list, lr_scheduler=scheduler, writer=writer)


if __name__ == '__main__':
    main()
