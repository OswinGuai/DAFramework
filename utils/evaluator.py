import math

import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import confusion_matrix

from model.base import Classification


def evaluate_classification(model_instance, input_loader, key=None):
    assert isinstance(model_instance, Classification)
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    # TODO notice the number of iteration
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        inputs = inputs.to(model_instance.device)
        labels = labels.to(model_instance.device)
        probabilities = model_instance.predict(inputs)
        probabilities = probabilities.data.float()
        labels = labels.data.long()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)
    model_instance.set_train(ori_train_state)
    _, predict = torch.max(all_probs, 1)
    if key is not None:
        confusion_matrix(all_labels, torch.squeeze(predict).long())
        np.save('confusion_matrix_%s.npy' % (key), confusion_matrix)
    accuracy = float(torch.sum(torch.squeeze(predict).long() == all_labels)) / float(all_labels.size()[0])
    return {'accuracy':accuracy}


def evaluate_classification_numpy(model_instance, tgt_f, tgt_l):
    assert isinstance(model_instance, Classification)
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    # TODO notice the number of iteration
    first_test = True
    batch_size = 1
    start = 0
    batch_num = math.ceil(len(tgt_f)/batch_size)
    print('batch_num:')
    print(batch_num)
    for i in range(batch_num):
        inputs = torch.from_numpy(tgt_f[i]).view(1,2048)
        inputs = inputs.to(model_instance.device)
        probabilities = model_instance.predict(inputs)
        probabilities = probabilities.data.float()
        if first_test:
            all_probs = probabilities
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
        start = start + batch_size
    model_instance.set_train(ori_train_state)
    _, predict = torch.max(all_probs, 1)
    accuracy = float(np.sum(torch.squeeze(predict).long().cpu().detach().numpy() == tgt_l)) / len(tgt_l)
    return {'accuracy':accuracy}


def evaluate_classification_perclass(model_instance, input_loader, class_num):
    assert isinstance(model_instance, Classification)
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    # TODO notice the number of iteration
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    per_class_correct = torch.zeros([class_num]).float().to(model_instance.device)
    per_class_counter = torch.zeros([class_num]).float().to(model_instance.device)
    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        inputs = inputs.to(model_instance.device)
        labels = labels.to(model_instance.device)
        labels = labels.data.long()
        probabilities = model_instance.predict(inputs)
        probabilities = probabilities.data.float()
        _, predict = torch.max(probabilities, 1)
        for j, p in enumerate(torch.squeeze(predict)):
            per_class_correct[labels[j]] += (torch.squeeze(p).long() == labels[j]).float()
            per_class_counter[labels[j]] += 1
    model_instance.set_train(ori_train_state)
    accuracy = per_class_correct / per_class_counter
    return dict(zip(['class_%d' % i for i in range(class_num)], accuracy.tolist()))


def vis_labeled_features(model_instance, input_loader, writer):
    assert isinstance(model_instance, Classification)
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    # TODO notice the number of iteration
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        inputs = inputs.to(model_instance.device)
        labels = labels.to(model_instance.device)
        features, probabilities = model_instance.output(inputs)
        features = features.data.float()
        labels = labels.data.long()
        if first_test:
            all_labels = labels
            all_features = features
            first_test = False
        else:
            all_labels = torch.cat((all_labels, labels), 0)
            all_features = torch.cat((all_features, features), 0)
    model_instance.set_train(ori_train_state)
    writer.add_embedding(all_features.data, metadata=all_labels.data, global_step=0)


def diff_classifiers(model1_instance, model2_instance, input_loader, key=None, iter_num=None, model1_name='model1', model2_name='model2'):
    assert isinstance(model1_instance, Classification)
    assert isinstance(model2_instance, Classification)
    ori_train_state1 = model1_instance.is_train
    ori_train_state2 = model2_instance.is_train
    model1_instance.set_train(False)
    model2_instance.set_train(False)
    # TODO notice the number of iteration
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model1_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        probabilities_1 = model1_instance.predict(inputs)
        probabilities_2 = model2_instance.predict(inputs)
        probabilities_1 = probabilities_1.data.float()
        probabilities_2 = probabilities_2.data.float()
        labels = labels.data.long()
        if first_test:
            all_probs_1 = probabilities_1
            all_probs_2 = probabilities_2
            all_labels = labels
            first_test = False
        else:
            all_probs_1 = torch.cat((all_probs_1, probabilities_1), 0)
            all_probs_2 = torch.cat((all_probs_2, probabilities_2), 0)
            all_labels = torch.cat((all_labels, labels), 0)
    model1_instance.set_train(ori_train_state1)
    model2_instance.set_train(ori_train_state2)
    _, predict_1 = torch.max(all_probs_1, 1)
    _, predict_2 = torch.max(all_probs_2, 1)
    if key is not None and iter_num is not None:
        cm_1 = confusion_matrix(all_labels, torch.squeeze(predict_1).long())
        np.save('%s_confusion_matrix_%s_%s.npy' % (model1_name, key, iter_num), cm_1)
        cm_2 = confusion_matrix(all_labels, torch.squeeze(predict_2).long())
        np.save('%s_confusion_matrix_%s_%s.npy' % (model2_name, key, iter_num), cm_2)
    right_1 = torch.squeeze(predict_1).long() == all_labels
    right_2 = torch.squeeze(predict_2).long() == all_labels
    accuracy_1 = float(torch.sum(right_1)) / float(all_labels.size()[0])
    accuracy_2 = float(torch.sum(right_2)) / float(all_labels.size()[0])
    diff_times = int(torch.sum(torch.squeeze(predict_1).long() != torch.squeeze(predict_2).long()))
    diff_ratio = float(torch.sum(torch.squeeze(predict_1).long() != torch.squeeze(predict_2).long() )) / float(all_labels.size()[0])
    both_right_times = int(torch.sum(right_1 + right_2 > 1))
    any_right_times = int(torch.sum(right_1 + right_2 > 0))
    both_right_ratio = float(both_right_times) / float(all_labels.size()[0])
    any_right_ratio = float(any_right_times) / float(all_labels.size()[0])
    return {'%s_accuracy' % model1_name : '%.2f' % accuracy_1,
            '%s_accuracy' % model2_name : '%.2f' % accuracy_2,
            'miss_times' : '%.2f' % diff_times,
            'diff_ratio' : '%.2f' % diff_ratio,
            'both_right_ratio' : '%.2f' % both_right_ratio,
            'any_right_ratio' : '%.2f' % any_right_ratio,
            }


def evaluate_classification_by_net(net, input_loader, device):
    # TODO notice the number of iteration
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        inputs = inputs.to(device)
        labels = labels.to(device)
        probabilities = net(inputs)
        probabilities = probabilities.data.float()
        labels = labels.data.long()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)
    _, predict = torch.max(all_probs, 1)
    accuracy = float(torch.sum(torch.squeeze(predict).long() == all_labels)) / float(all_labels.size()[0])
    return {'accuracy':accuracy}


def get_label_features(model_instance, input_loader):
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        inputs = inputs.to(model_instance.device)
        labels = labels.to(model_instance.device)
        features, probabilities = model_instance.output(inputs)
        features = features.data.float()
        labels = labels.data.long()
        if first_test:
            all_labels = labels
            all_features = features
            first_test = False
        else:
            all_labels = torch.cat((all_labels, labels), 0)
            all_features = torch.cat((all_features, features), 0)
    return all_features, all_labels


def vis_tsne(model_instance, input_loader_list, writer):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    # TODO notice the number of iteration
    first_test = True
    for input_loader in input_loader_list:
        if first_test:
            all_features, all_labels = get_label_features(model_instance, input_loader)
            first_test = False
        else:
            features, labels = get_label_features(model_instance, input_loader)
            all_labels = torch.cat((all_labels, labels), 0)
            all_features = torch.cat((all_features, features), 0)
    model_instance.set_train(ori_train_state)
    writer.add_embedding(
            all_features.data,
            metadata=all_labels.data,
            global_step=0
            )


def evaluate_distance(model_instance, src_input_loader, tgt_input_loader):
    src_num_iter = len(src_input_loader)
    tgt_num_iter = len(tgt_input_loader)
    src_iter = iter(src_input_loader)
    tgt_iter = iter(tgt_input_loader)
    for i in range(src_num_iter):
        data = src_iter.next()
        inputs = data[0].detach()
        labels = data[1].detach()
        inputs = inputs.to(model_instance.device)
        labels = labels.to(model_instance.device)
        model_instance.accumulate_src(inputs, labels)
    for i in range(tgt_num_iter):
        data = tgt_iter.next()
        inputs = data[0].detach()
        labels = data[1].detach()
        inputs = inputs.to(model_instance.device)
        labels = labels.to(model_instance.device)
        model_instance.accumulate_tgt(inputs, labels)
    model_instance.cal_centors()
    src_iter = iter(src_input_loader)
    tgt_iter = iter(tgt_input_loader)
    for i in range(src_num_iter):
        data = src_iter.next()
        inputs = data[0].detach()
        labels = data[1].detach()
        inputs = inputs.to(model_instance.device)
        labels = labels.to(model_instance.device)
        model_instance.cal_src_dis(inputs, labels)
    for i in range(tgt_num_iter):
        data = tgt_iter.next()
        inputs = data[0].detach()
        labels = data[1].detach()
        inputs = inputs.to(model_instance.device)
        labels = labels.to(model_instance.device)
        model_instance.cal_tgt_dis(inputs, labels)
    result = []
    for i in range(model_instance.class_num):
        src_dis = model_instance.src_class_dis_sum[i] / model_instance.src_class_num[i]
        tgt_dis = model_instance.tgt_class_dis_sum[i] / model_instance.tgt_class_num[i]
        common_dis = model_instance.common_class_dis_sum[i] / model_instance.tgt_class_num[i]
        line = '%2f %2f %2f\n' % (src_dis.item(),tgt_dis.item(),common_dis.item())
        result.append(line)
    return result
