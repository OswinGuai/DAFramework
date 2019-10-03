import torch

def onehot(y, class_num):
    batch_size = y.size(0)
    y_onehot = torch.FloatTensor(batch_size, class_num).to(y.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(batch_size, 1), 1)
    return y_onehot
