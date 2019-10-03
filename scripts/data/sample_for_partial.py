import sys
import numpy as np
import random

input_file = sys.argv[1]
tag = sys.argv[2]
if input_file.endswith('.txt'):
    output_tag = input_file[:-4]
else:
    output_tag = output_tag

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

first = 2
second = 20
third = 60
selected_classes_1st = random.sample(labeled_images.keys(), first)
selected_classes_2nd = random.sample(labeled_images.keys(), second)
selected_classes_3rd = random.sample(labeled_images.keys(), third)
selected_sample_num_1st = sum([len(labeled_images[key]) for key in selected_classes_1st])
selected_sample_num_2nd = sum([len(labeled_images[key]) for key in selected_classes_2nd])
selected_sample_num_3rd = len(list_images)
rate_1st = 1
rate_2nd = float(selected_sample_num_1st) / selected_sample_num_2nd
rate_3rd = float(selected_sample_num_1st) / selected_sample_num_3rd
output_1st = []
output_2nd = []
output_3rd = []
for (k,v) in labeled_images.items():
    if k in selected_classes_1st:
        output_1st.extend(['%s %s\n' % (image, k) for image in labeled_images[k]])
    if k in selected_classes_2nd:
        sampled_num = int(len(labeled_images[k]) * rate_2nd)
        sampled = random.sample(labeled_images[k], sampled_num)
        output_2nd.extend(['%s %s\n' % (image, k) for image in sampled])
    if k in selected_classes_3rd:
        sampled_num = int(len(labeled_images[k]) * rate_3rd)
        sampled = random.sample(labeled_images[k], sampled_num)
        output_3rd.extend(['%s %s\n' % (image, k) for image in sampled])

output_file_1st = '%s_partial_%d_%s.txt' % (output_tag, first, tag)
with open(output_file_1st, 'w') as output:
    output.writelines(output_1st)

output_file_2nd = '%s_partial_%d_%s.txt' % (output_tag, second, tag)
with open(output_file_2nd, 'w') as output:
    output.writelines(output_2nd)

output_file_3rd = '%s_partial_%d_%s.txt' % (output_tag, third, tag)
with open(output_file_3rd, 'w') as output:
    output.writelines(output_3rd)

print('Sampling is finished.')

