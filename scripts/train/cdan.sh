
model=cdan
#version=mme_ada_alter20_002
#version=mme_ada_delay1000_less_evat_001
version=ori_103
gpu=6

prefix=/
dataset=office
src=webcam
tgt=amazon
tgttest=${tgt}

method='none'
lr=0.001
alpha=1
beta=0
gamma=0
seed=2020

noiselevel=0
resume=0
start_iter=0
saved_model='/'
pretrain='none'

tag=${model}_${src}2${tgt}
vis_log=vis_log

start=0
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter}
start=$((start + 1))

