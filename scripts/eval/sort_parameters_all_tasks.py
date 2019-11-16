
import sys
import os
import re


model = sys.argv[1]
tasks = sys.argv[2]
task_list = tasks.split(',')
root = "outputs"
start_str_list = ['nohup_%s_%s_' % (model, task) for task in task_list]
end_str = '.output'

def cal_acc(f_name, sample):
    with open(f_name) as f:
        lines = f.readlines()
        lines.reverse()
        # {'accuracy': 0.5725949591764288}
        head = "{'accuracy': "
        num = 0
        total = 0
        for l in lines:
            if l.startswith(head):
                num += 1
                total += float(l[len(head):min(len(head)+6, len(l)-2)])
            if num >= sample:
                break
        if num != 0:
            acc = total / num
        else:
            acc = 0
    return acc


record = {}
def read_record_general(root, start_str, end_str, record):
    for rt, dirs, files in os.walk(root):
        #print(files)
        for f in files:
            if f.startswith(start_str):
                acc = cal_acc('%s/%s' % (root,f), 3)
                key = f[len(start_str):-len(end_str)]
                if key in record:
                    record[key] += acc
                else:
                    record[key] = acc

for start_str in start_str_list:
    read_record_general(root, start_str, end_str, record)

sorted_record = sorted(record.items(), key=lambda kv: kv[1])
sorted_record.reverse()
for c in sorted_record:
    print(c)


