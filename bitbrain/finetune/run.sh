##########################
# llama factory start script
##########################
#CUDA_VISIBLE_DEVICES=5,6 python ../LLaMA-Factory/src/train.py \
export LLaMA_PATH=/home/chenyuhang/LLaMA-Factory
OUTPUT_DIR="输出路径"
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5,6,7 
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 可设置混合策略：     --mix_strategy interleave_over\
#    --interleave_probs 0.1,0.35,0.2,0.2,0.1,0.05 \

# 可设置自定义的评估数据集  --eval_dataset ceval,cmmlu \


FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=4,5,6,7  torchrun --nproc_per_node 4 $LLaMA_PATH/src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path "预训练模型路径" \
    --cutoff_len 2048 \
    --dataset_dir "数据集路径" \
    --dataset shared_gpt_format\
    --overwrite_cache \
    --max_samples 5000000 \
    --packing True \
    --use_swanlab true \
    --report_to swanlab \
    --run_name sft_bit-brain \
    --preprocessing_num_workers 30 \
    --template qwen \
    --finetuning_type full \
    --output_dir ${OUTPUT_DIR}/sft \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --do_eval \
    --val_size 100 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --flash_attn sdpa\
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --bf16 \
    --resume_from_checkpoint "恢复训练的checkpoint路径"

