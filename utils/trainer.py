import os
import sys

import torch

from model.da import DomainAdaptation
from model.da_evaluator import DomainAdaptationEvaluator
from utils.evaluator import evaluate_classification, evaluate_classification_perclass


def train_da_evaluator(model_instance, train_source_loader, train_target_loader, num_iterations, optimizer_list, lr_scheduler, writer):
    assert isinstance(model_instance, DomainAdaptationEvaluator)
    num_batch_train_source = len(train_source_loader) - 1
    num_batch_train_target = len(train_target_loader) - 1
    model_instance.set_train(True)
    ## train one iter
    print("start train...")
    iter_source = iter(train_source_loader)
    iter_target = iter(train_target_loader)
    for iter_num in range(num_iterations):
        optimizer_list = [lr_scheduler.next_optimizer(optimizer_list[o_i], iter_num) for o_i in range(len(optimizer_list))]
        if iter_num % num_batch_train_source == 0:
            iter_source = iter(train_source_loader)
        if iter_num % num_batch_train_target == 0:
            iter_target = iter(train_target_loader)
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source = inputs_source.to(model_instance.device)
        inputs_target = inputs_target.to(model_instance.device)
        train_train_da_evaluator_batch(model_instance, inputs_source, labels_source, inputs_target, labels_target, optimizer_list, iter_num, writer)

        if iter_num % 100 == 0:
            sys.stdout.write('iteration proceed %.2f%%     \r' % (float(100) * iter_num/num_iterations))
            sys.stdout.flush()
    print("\nfinish train.")


def train_train_da_evaluator_batch(model_instance, inputs_source, labels_source, inputs_target, labels_target, optimizer_list, iter_num, writer):
    total_loss = model_instance.get_loss(inputs_source, labels_source, inputs_target, labels_target, iter_num, writer)
    if not (isinstance(total_loss, list) or isinstance(total_loss, tuple)): 
        total_loss = [total_loss]

    for l_i, item_loss in enumerate(total_loss):
        optimizer_list[l_i].zero_grad()
        if not isinstance(item_loss, int) and not isinstance(item_loss, float):
            if l_i == len(total_loss)-1:
                item_loss.backward()
            else:
                item_loss.backward(retain_graph=True)
        optimizer_list[l_i].step()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_da(model_instance, train_source_loader, train_target_loader, test_target_loader, num_iterations, optimizer_list, lr_scheduler, writer, key=None, do_eval=True, do_save=True, model_dir='results', start_iter=0, eval_perclass=False, class_num=None):
    assert isinstance(model_instance, DomainAdaptation)
    num_batch_train_source = len(train_source_loader) - 1
    num_batch_train_target = len(train_target_loader) - 1
    model_instance.set_train(True)
    ## train one iter
    print("start train...")
    iter_source = iter(train_source_loader)
    iter_target = iter(train_target_loader)
    for iter_num in range(start_iter, num_iterations):
        optimizer_list = [lr_scheduler.next_optimizer(optimizer_list[o_i], iter_num) for o_i in range(len(optimizer_list))]
        if iter_num % num_batch_train_source == 0:
            iter_source = iter(train_source_loader)
        if iter_num % num_batch_train_target == 0:
            iter_target = iter(train_target_loader)
        inputs_source, labels_source = iter_source.next()
        inputs_target, _ = iter_target.next()
        inputs_source = inputs_source.to(model_instance.device)
        inputs_target = inputs_target.to(model_instance.device)
        labels_source = labels_source.to(model_instance.device)
        train_da_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer_list, iter_num, writer)

        if iter_num > 0 and do_eval and ((iter_num <= 1000 and iter_num % 100 == 0) or (iter_num <= 5000 and iter_num % 500 == 0) or iter_num % 1000 == 0):
            eval_result = evaluate_classification(model_instance, test_target_loader)
            print('')
            print('iteration number %s' % iter_num)
            print(eval_result)
            writer.add_scalar('eval/tgt-accu', eval_result['accuracy'], iter_num)
            if eval_perclass:
                per_class_eval_result = evaluate_classification_perclass(model_instance, test_target_loader, class_num)
                writer.add_scalars('perclass', per_class_eval_result, iter_num)
                print("-"*4 + "Accuracy per class" + "-"*4 )
                print(per_class_eval_result)
                print("-"*8)
        else:
            sys.stdout.write('iteration proceed %.2f%%     \r' % (float(100) * iter_num/num_iterations))
            sys.stdout.flush()
        if do_save and iter_num % 1000 == 0:
            torch.save(model_instance, os.path.join(model_dir, '%s_%d.pkl' % (key, iter_num)))
    print("\nfinish train.")


def train_da_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer_list, iter_num, writer):
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    total_loss = model_instance.get_loss(inputs, labels_source, iter_num, writer)
    if not (isinstance(total_loss, list) or isinstance(total_loss, tuple)): 
        total_loss = [total_loss]
    for l_i, item_loss in enumerate(total_loss):
        optimizer_list[l_i].zero_grad()
        if not isinstance(item_loss, int) and not isinstance(item_loss, float):
            if l_i == len(total_loss)-1:
                item_loss.backward()
            else:
                item_loss.backward(retain_graph=True)
        optimizer_list[l_i].step()
        curr_lr = get_lr(optimizer_list[l_i])
        writer.add_scalar('train/lr_%d' % l_i, curr_lr, iter_num)
