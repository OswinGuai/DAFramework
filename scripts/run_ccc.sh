
model=ccc
version=hard_ent_102
prefix=/
dataset=office
start=4

src=amazon
tgt=webcam
gpu=${start}
sh scripts/train/run_with_params.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version} ${model}
start=$((start + 1))

src=webcam
tgt=amazon
gpu=${start}
sh scripts/train/run_with_params.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version} ${model}
start=$((start + 1))

src=amazon
tgt=dslr
gpu=${start}
sh scripts/train/run_with_params.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version} ${model}
start=$((start + 1))

src=dslr
tgt=amazon
gpu=${start}
sh scripts/train/run_with_params.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version} ${model}
start=$((start + 1))

