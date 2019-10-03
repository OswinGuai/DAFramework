import torch
from tensorboardX import SummaryWriter
from torch import optim

from data.dataloader import load_images
from model.cdan import CDAN
from utils.evaluator import evaluate_classification
from utils.lr_scheduler import INVScheduler
from utils.randomness import set_randomness, worker_init_fn
from utils.trainer import train_da
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
    test_file_path = datasets[args.tgttest]
    model_path = 'results/final_model_%s.pkl' % args.key
    batch_size = 32
    train_source_loader = load_images(train_source_file_path, batch_size=batch_size, resize_size=256, is_train=True, crop_size=224, worker_init=worker_init_fn, prefix=args.prefix)
    train_target_loader = load_images(train_target_file_path, batch_size=batch_size, resize_size=256, is_train=True, crop_size=224, worker_init=worker_init_fn, prefix=args.prefix)
    test_target_loader = load_images(test_file_path, batch_size=batch_size, resize_size=256, is_train=False, crop_size=224, worker_init=worker_init_fn, prefix=args.prefix)
    
    writer = SummaryWriter('%s/%s' % (args.logdir, args.key))
    # Init model
    if args.resume == 1:
        model_instance = torch.load(args.saved_model)
    else:
        model_instance = CDAN(class_num=K, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    start_iter = args.start_iter
    parameter_list = model_instance.get_parameter_list()
    optimizer_list = []
    for parameter in parameter_list:
        optimizer = optim.SGD(parameter, lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
        optimizer_list.append(optimizer)
    scheduler = INVScheduler(gamma=0.001, power=0.75, decay_rate=0.0005, init_lr=args.lr)
    # Train model
    train_da(model_instance, train_source_loader, train_target_loader, test_target_loader, num_iterations=100000, optimizer_list=optimizer_list, lr_scheduler=scheduler, writer=writer, key=args.key, do_eval=True, model_dir='results', start_iter=start_iter)
    # Evaluate model
    print("All training is finished.")
    eval_result = evaluate_classification(model_instance, test_target_loader)
    print(eval_result)
    # Save model
    torch.save(model_instance, model_path)


if __name__ == '__main__':
    main()
