
model=srconly
version=srconly_002

prefix=/
dataset=office
noiselevel=0
src=amazon
tgt=webcam
tgttest=${tgt}

method='none'
gpu=7
pretrain=0
lr=0.0001
alpha=1
beta=1
gamma=1
seed=2019

resume=False
start_iter=0
saved_model='/'

tag=${src}2${tgt}
vis_log=vis_log_${tag}

start=0
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${pretrain} ${noiselevel} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter}
start=$((start + 1))

