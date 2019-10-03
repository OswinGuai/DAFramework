
model=dadv
version=deep_l2_100
prefix=/
dataset=office
start=4

src=amazon
tgt=webcam
gpu=${start}
sh scripts/${model}_batch.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version}
start=$((start + 1))

src=webcam
tgt=amazon
gpu=${start}
sh scripts/${model}_batch.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version}
start=$((start + 1))

src=amazon
tgt=dslr
gpu=${start}
sh scripts/${model}_batch.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version}
start=$((start + 1))

src=dslr
tgt=amazon
gpu=${start}
sh scripts/${model}_batch.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version}
start=$((start + 1))
