import sys
import os

key = sys.argv[1]
scale = sys.argv[2]
num_class = int(sys.argv[3])
start,end = map(int, scale.split(','))
src_lines = ['' for i in range(num_class)]
for i in range(start, end+1):
    file_path = 'distance_test/%s-%03d.txt' % (key, i)
    with open(file_path) as ff:
        ll = ff.readlines()
        for j,l in enumerate(ll):
            a = l.strip()
            src_lines[j] += '%s\t' % a

for i in range(len(src_lines)):
    src_lines[i] = src_lines[i][:-1] + '\n'

with open('%s_src_dis.txt' % key, 'w') as ff:
    ff.writelines(src_lines)

