import os
import sys
import signal


def main():
    tag = sys.argv[1]
    if len(sys.argv) > 2:
        scale = sys.argv[2]
    else:
        scale = tag.split('-')[-1]
        tag = tag[:-(len(scale)+1)]
    rm = 'no'
    if len(sys.argv) > 3:
        rm = sys.argv[3]
    numbers = scale.split(',')
    for number in numbers:
        number = number.strip()
        if '-' in number:
            ss = number.split('-')
            start = int(ss[0])
            end = int(ss[1])
            for i in range(start, end+1):
                kill_by_task_id(tag, i, rm)
        else:
            kill_by_task_id(tag, number, rm)

def kill_by_task_id(tag, id, rm):
    task_file = 'records/%s-%s_task.record' % (tag, id)
    #vis_file = 'records/%s-%d_vis.record' % (tag, id)
    try:
        pid_list = get_pid_from_record(task_file)
        if rm == 'f':
            os.remove(task_file)
            #os.remove(vis_file)
    except FileNotFoundError:
        print('%s not Found. Ignore.' % task_file)
        return
    print('following pid is being killed...')
    print(pid_list)
    for pid in pid_list:
        kill_by_pid(int(pid))

def kill_by_pid(pid):
    try:
        a = os.kill(pid, signal.SIGKILL)
        print('killed: %d!' % pid)
    except OSError:
        print('没有如此进程: %d!' % pid)


def get_pid_from_record(name):
    with open(name) as rr:
        lines = rr.readlines()
    pids = []
    for l in lines:
        if 'tensorboard_pid=' in l:
            pid = int(l[len('tensorboard_pid='):].strip())
            pids.append(pid)
        if 'train_pid=' in l:
            pid = int(l[len('train_pid='):].strip())
            pids.append(pid)
    return pids

if __name__=='__main__':
    main()
