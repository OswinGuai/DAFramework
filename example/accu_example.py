import torch
from tensorboardX import SummaryWriter

from data.dataloader import load_images
from utils.randomness import set_randomness, gen_init_fn
from utils.evaluator import evaluate_classification_perclass
from dataset_info import *
from opts import read_config

args = read_config()
GLOBAL_SEED = args.seed


def main():
    set_randomness(GLOBAL_SEED, deterministic=True, benchmark=False)
    # Prepare data
    datasets = data[args.dataset]
    K = num_class[args.dataset]
    print(args)
    target_dataset = datasets[args.tgt]
    train_target_file_path = target_dataset
    batch_size = 16
    train_target_loader = load_images(train_target_file_path, batch_size=batch_size, resize_size=256, is_train=True, crop_size=224, worker_init=gen_init_fn(GLOBAL_SEED), prefix=args.prefix)
    writer = SummaryWriter('%s/%s' % (args.logdir, args.key))
    # Init model
    target_model = torch.load(args.saved_model)
    # Evaluate model
    result = evaluate_classification_perclass(target_model, K, train_target_loader)
    file_path = 'distance_test/classaccu_%s.txt' % args.key
    with open(file_path, 'w') as result_file:
        result_file.writelines(result)
    print('write result into %s.' % file_path)


if __name__ == '__main__':
    main()
