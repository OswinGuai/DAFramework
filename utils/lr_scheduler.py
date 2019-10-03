import math


class INVScheduler(object):
    def __init__(self, gamma, power, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr
        self.power = power

    def next_optimizer(self, optimizer, iter_num):
        lr = self.init_lr * (1 + self.gamma * iter_num) ** (-self.power)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = self.decay_rate * param_group['decay_mult']
            i+=1
        return optimizer

    def simple_ajustment(self, epoch, group_ratios, optimizer):
        optim_factor = 0
        if(epoch > 160):
            optim_factor = 3
        elif(epoch > 120):
            optim_factor = 2
        elif(epoch > 60):
            optim_factor = 1
        lr = self.init_lr * math.pow(0.2, optim_factor)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i+=1
        return optimizer
