# llama factory start script
##########################
export LLaMA_PATH="llama_factory本地路径"
OUTPUT_DIR="输出路径"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3   torchrun --nproc_per_node 4 $LLaMA_PATH/src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path "模型路径" \
    --cutoff_len 2048 \
    --dataset_dir "数据集路径" \
    --dataset chinese_instruct,deepctrl_200k,ultrachat_200k,code_feedback_custom \
    --overwrite_cache \
    --enable_liger_kernel True\
    --packing False \
    --use_swanlab true \
    --report_to swanlab \
    --swanlab_project "swanlab项目名称" \
    --run_name "运行名称" \
    --preprocessing_num_workers 30 \
    --template qwen \
    --finetuning_type full \
    --output_dir ${OUTPUT_DIR}/"本次运行保存子路径" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_eval \
    --val_size 100 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --flash_attn fa2\
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0125 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 5000 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16 
    
    #--resume_from_checkpoint ${OUTPUT_DIR}/"本次运行保存子路径/checkpoint-xxxx"


