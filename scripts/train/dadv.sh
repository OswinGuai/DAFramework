
model=dadv
version=ilevel_dnet_cls_mix_2000
#version=ilevel_dnet_mix_2011
#version=ori_2001
gpu=7

#prefix=/
#dataset=office
prefix=/data/office-home/images
dataset=officehome
noiselevel=0
#src=amazon
#tgt=webcam
src=Art
tgt=Product
tgttest=${tgt}

method='none'
lr=0.001
alpha=1
beta=1
gamma=1
seed=2019

resume=0
start_iter=0
saved_model=results/dann_webcam2amazon-normal_5001_${start_iter}.pkl
pretrain='none'

tag=${model}_${src}2${tgt}
vis_log=vis_log

start=0
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter}
start=$((start + 1))

