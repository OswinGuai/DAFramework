
model=$7
version=$6
gpu=$1

prefix=$2
dataset=$3
src=$4
tgt=$5
tgttest=${tgt}

lr=0.0005
alpha=1
beta=10
gamma=0.1

seed=2000
noiselevel=0
noise=0
method='none'

resume=0
start_iter=0
saved_model=results/dann_webcam2amazon-normal_5001_${start_iter}.pkl
pretrain='none'

tag=${model}_${src}2${tgt}_${alpha}_${beta}_${gamma}
vis_log=vis_log

start=0
sh scripts/train/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter}
start=$((start + 1))

