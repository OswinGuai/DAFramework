mkdir records
mkdir outputs
mkdir model/bak
mkdir results

model_name=${12}
experiment_id=${11}-$1
gpu=$2
src=$3
tgt=$4
tgttest=$5
lr=$6
alpha=$7
beta=$8
gamma=$9
seed=${10}
dataset=${13}
prefix=${14}
method=${15}
log=${16}
resume=${17}
saved_model=${18}
start_iter=${19}

export CUDA_VISIBLE_DEVICES=${gpu}
tag=${experiment_id}

echo -----back up model/${model_name}.py-----
now=$(date +"%Y%m%d_%T")
cp model/${model_name}.py model/bak/${model_name}_${experiment_id}_${now}.py

cmd="python example/${model_name}_example.py --key ${tag} --lr ${lr} --seed ${seed} --alpha ${alpha} --beta ${beta} --gamma ${gamma} --src ${src} --tgt ${tgt} --tgttest ${tgttest} --logdir ${log} --dataset ${dataset} --prefix ${prefix} --method ${method} --saved_model ${saved_model} --start_iter ${start_iter} --resume ${resume}"
nohup $cmd >> outputs/nohup_${tag}.output &
train_pid=$!
echo -----cmd------
echo ${cmd}
echo ----output----
echo "tail -f outputs/nohup_${tag}.output"

#port=$(shuf -i 10000-14900 -n 1)
#while lsof -i:${port}
#do
#    port=$(shuf -i 10000-14900 -n 1)
#done
#local_ip=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1')

#nohup tensorboard --logdir ${log} --port ${port}&
#tensorboard_pid=$!

echo -----tag------
echo Start task of ${tag}...
echo $cmd >> records/${tag}_task.record
echo gpu=${gpu} >> records/${tag}_task.record
echo train_pid=${train_pid} >> records/${tag}_task.record
#echo tensorboard_pid=${tensorboard_pid} >> ${tag}_task.record
cat records/${tag}_task.record
echo -------------- >> records/${tag}_task.record
#echo http://${local_ip}:${port} >> ${tag}_vis.record
#cat ${tag}_vis.record


