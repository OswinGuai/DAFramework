import sys
import numpy as np
import random

input_file = sys.argv[1]
tag = sys.argv[2]
train_num = int(sys.argv[3])
split_num = int(sys.argv[4])
if input_file.endswith('.txt'):
    output_tag = input_file[:-4]
else:
    output_tag = output_tag

tips = '''
Generate samples for target domain in few-shot DA.
The number of target samples for per category: %d.
The number of splits: %d.
The tag of output file: %s.
The output files name like %s_train_1.txt and %s_test_1.txt.
''' % (train_num, split_num, output_tag, output_tag, output_tag)
print(tips)

list_file = open(input_file)
list_images = list_file.readlines()
list_file.close()

labeled_images = {}
for line in list_images:
    item = line.strip().split(' ')
    addr = item[0]
    l = item[1]
    if l not in labeled_images:
        labeled_images[l] = []
    labeled_images[l].append(addr)

for s in range(split_num):
    train_list = []
    test_list = []
    for key in labeled_images:
        condidates = labeled_images[key]
        train_images = random.sample(condidates, train_num)
        test_images = set(condidates).difference(train_images)
        train_list.extend(['%s %s\n' % (image, key) for image in train_images])
        test_list.extend(['%s %s\n' % (image, key) for image in test_images])
    output_file = '%s_sampled_%d_%s-%d.txt' % (output_tag, train_num, tag, s+1)
    with open(output_file, 'w') as output:
        output.writelines(train_list)
    output_file = '%s_left_%d_%s-%d.txt' % (output_tag, train_num, tag, s+1)
    with open(output_file, 'w') as output:
        output.writelines(test_list)

print('Sampling is finished.')
