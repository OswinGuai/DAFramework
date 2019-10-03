
model=tsne

prefix=/
dataset=office
noiselevel=0
src=webcam
tgt=amazon
srctest=$src
tgttest=$tgt

method='none'
lr=0.001
alpha=1
beta=1
gamma=1
seed=2019

resume=0
start_iter=0
pretrain='none'

vis_log=vis_log
#vis_log=vis_log/${tag}
#target_model=cdadv_amazon2webcam-cdane_ori_1000
target_model=cdadv_amazon2webcam-cdane_ilevel_d_net_mix_2006
#target_model=cdadv_webcam2amazon-cdane_ilevel_d_net_mix_2008
#target_model=cdadv_webcam2amazon-cdane_ori_1001

iter=20000

gpu=6
#version=102
version=4000
saved_model=results/${target_model}_${iter}.pkl
tag=${model}_${src}2${tgt}
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}

