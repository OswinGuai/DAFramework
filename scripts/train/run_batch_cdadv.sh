
model=cdadv
version=zero_1000
#prefix=/
#dataset=office
dataset=officehome
prefix=/data/office-home/images
start=0

src=Art
tgt=Product
gpu=${start}
sh scripts/${model}_batch.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version}
start=$((start + 1))

src=Art
tgt=Clipart
gpu=${start}
sh scripts/${model}_batch.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version}
start=$((start + 1))

src=Art
tgt=RealWorld
gpu=${start}
sh scripts/${model}_batch.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version}
start=$((start + 1))

src=Product
tgt=RealWorld
gpu=${start}
sh scripts/${model}_batch.sh ${gpu} ${prefix} ${dataset} ${src} ${tgt} ${version}
start=$((start + 1))

