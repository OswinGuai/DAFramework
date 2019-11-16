
model=mixup_cdan
version=test
#prefix=/
#dataset=office
dataset=officehome
prefix=/data/office-home/images
start=0

src=Art
tgt=Product
gpu=${start}
sh scripts/train/run_with_params.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version} ${model}
start=$((start + 1))

src=Art
tgt=Clipart
gpu=${start}
sh scripts/train/run_with_params.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version} ${model}
start=$((start + 1))

