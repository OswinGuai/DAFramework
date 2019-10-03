
import os
import sys
import random

list_file = open(sys.argv[1])
train_rate = float(sys.argv[2])
lines = list_file.readlines()
num_lines = len(lines)
indices = range(num_lines)
train_num = int(num_lines * train_rate)
train_indices = random.sample(indices, train_num)
test_indices = list(set(indices) - set(train_indices))
train_lines = [lines[i] for i in train_indices]
test_lines = [lines[i] for i in test_indices]

with open(sys.argv[3], 'w') as train_list_file:
    train_list_file.writelines(train_lines)
with open(sys.argv[4], 'w') as test_list_file:
    test_list_file.writelines(test_lines)

