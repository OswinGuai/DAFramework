
model=distance

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
#target_model=dann_amazon2webcam-normal_5001
target_model=dann_webcam2amazon-normal_5001

gpu=0
version=000
iter=0
saved_model=results/${target_model}_${iter}.pkl
tag=${model}_${src}2${tgt}
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}

version=001
iter=2000
saved_model=results/${target_model}_${iter}.pkl
tag=${model}_${src}2${tgt}
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}

version=002
iter=4000
saved_model=results/${target_model}_${iter}.pkl
tag=${model}_${src}2${tgt}
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}

gpu=1
version=003
iter=6000
saved_model=results/${target_model}_${iter}.pkl
tag=${model}_${src}2${tgt}
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}

version=004
iter=8000
saved_model=results/${target_model}_${iter}.pkl
tag=${model}_${src}2${tgt}
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}

version=005
iter=10000
saved_model=results/${target_model}_${iter}.pkl
tag=${model}_${src}2${tgt}
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}

gpu=2
version=006
iter=12000
saved_model=results/${target_model}_${iter}.pkl
tag=${model}_${src}2${tgt}
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}

version=007
iter=14000
saved_model=results/${target_model}_${iter}.pkl
tag=${model}_${src}2${tgt}
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}

version=008
iter=16000
saved_model=results/${target_model}_${iter}.pkl
tag=${model}_${src}2${tgt}
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}

gpu=3
version=009
iter=18000
saved_model=results/${target_model}_${iter}.pkl
tag=${model}_${src}2${tgt}
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}

version=010
iter=20000
saved_model=results/${target_model}_${iter}.pkl
tag=${model}_${src}2${tgt}
sh scripts/run_once.sh ${version} ${gpu} ${src} ${tgt} ${tgttest} ${lr} ${alpha} ${beta} ${gamma} ${seed} ${tag} ${model} ${dataset} ${prefix} ${method} ${vis_log} ${resume} ${saved_model} ${start_iter} ${tgttest}

