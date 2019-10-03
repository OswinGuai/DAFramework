
model=lambdatest
version=earlystop_iter_10000
gpu=7

prefix=/
dataset=office
noiselevel=0
src=webcam-train
tgt=amazon-train
srctest=webcam-test
tgttest=amazon-test

method='none'
lr=0.001
alpha=1
beta=1
gamma=1
seed=2019

saved_model='results/dann_webcam2amazon-earlystop_f_net_10000.pkl'
resume=0
start_iter=0
pretrain='none'

tag=${model}_${src}2${tgt}
vis_log=vis_log
#vis_log=vis_log/${tag}

start=0
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}
start=$((start + 1))

