
model=mdd
<<<<<<< HEAD
#version=ori_1003
#version=ilevel_srcadv_tgtadv_mix_1004
version=ilevel_srcadv_tgtadv_mix_2000
gpu=0
=======
version=ilevel_srcadv_tgtadv_mix_2000
#version=ori_1004
gpu=1
>>>>>>> bf100f4f8bdaaade78b77ccbdabb71be92d19777

#prefix=/
#dataset=office
#src=amazon
#tgt=webcam
#tgttest=${tgt}

prefix=/data/office-home/images
dataset=officehome
src=Art
tgt=Clipart
tgttest=${tgt}

#prefix=/
#dataset=domainnet
#src=infograph
#tgt=real
#tgttest=${tgt}-test

lr=0.001
alpha=1
beta=1
gamma=0

seed=2019
noiselevel=0
noise=0
method='none'

resume=0
start_iter=0
saved_model=results/dann_webcam2amazon-normal_5001_${start_iter}.pkl
pretrain='none'

tag=${model}_${src}2${tgt}
vis_log=vis_log

start=0
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter}
start=$((start + 1))

