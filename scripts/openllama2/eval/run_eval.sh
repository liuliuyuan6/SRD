BASE_PATH=${1-"/home/srd"}
port=2040


for data in dolly self_inst vicuna sinst uinst   
do
    # Evaluate     
    for seed in 10 20 30 40 50
    do
        ckpt="gpt2_pocl"
        
        bash ${base_path}/scripts/gpt2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 16
    done

    
done