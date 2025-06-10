#!  在v6的基础上，进一步添加训练过程中的测试，并添加权重衰减
#*  默认使用混合精度，并启动 torch.compile加速
import os
import sys
import argparse
import time  
import math  
import torch.compiler 
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.distributed as dist  
from torch.nn.parallel import DistributedDataParallel as DDP  
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  
from torch.amp import GradScaler, autocast  
from modelscope import AutoConfig, AutoTokenizer
from bitbrain.dataset.pretrain_dataset_arrow import PretrainDataset
from transformers import get_cosine_schedule_with_warmup
from loguru import logger
#! 从新的 mfu.py 文件导入 MFU 相关函数
from bitbrain.train.tools.mfu import get_gpu_peak_flops, estimate_model_flops, calculate_mfu_distributed
from bitbrain.train.tools.utils import test_model_on_prompts
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import swanlab

#! 禁用CUDA Graph Trees
torch._inductor.config.triton.cudagraph_trees = False

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="本地Qwen3路径", help="模型文件路径")
#! 训练数据集相关参数
parser.add_argument("--data_path", type=str, default="分词后的.arrow文件路径", help="训练数据集路径")
parser.add_argument("--num_epochs", type=int, default=1, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--seq_len", type=int, default=2048, help="训练时使用的序列长度")
#! 添加分布式训练相关参数
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
parser.add_argument("--world_size", type=int, default=1, help="Number of processes for distributed training")
parser.add_argument("--master_addr", type=str, default="localhost", help="Master address for distributed training")
parser.add_argument("--master_port", type=str, default="12355", help="Master port for distributed training")
#! 添加模型保存相关参数
parser.add_argument("--save_dir", type=str, default="./out", help="用于保存在epoch中途的检查点的目录。默认: 'checkpoints_in_epoch'")
parser.add_argument("--save_interval", type=int, default=5000, help="每N个原始批次（dataloader的批次）保存一次检查点。默认: 1000。如果为0，则禁用epoch中途保存。")
args = parser.parse_args()

#! (1)初始化分布式训练环境
def setup_distributed():
    """初始化分布式训练环境"""
    # 检查是否在分布式环境中运行
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 从环境变量获取分布式参数（推荐方式）
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 从命令行参数获取（备用方式）
        rank = args.local_rank
        world_size = args.world_size
        local_rank = args.local_rank
        
        # 设置环境变量
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
    
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

if torch.distributed.is_available() and torch.cuda.device_count() > 1:
    rank, world_size, local_rank = setup_distributed()
    is_distributed = True
    device = f"cuda:{local_rank}"
    
    # 只在主进程输出初始化信息
    if rank == 0:
        logger.info(f"分布式训练已启用: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        logger.info(f"使用 {world_size} 个GPU进行训练")
else:
    rank = 0
    world_size = 1
    local_rank = 0
    is_distributed = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("使用单卡训练模式")

# 只在主进程输出日志的装饰器
def main_process_only(func):
    """装饰器：只在主进程执行函数"""
    def wrapper(*args, **kwargs):
        if rank == 0:
            return func(*args, **kwargs)
        return None
    return wrapper

# 包装logger的info方法
original_logger_info = logger.info
logger.info = main_process_only(original_logger_info)

#! (1) 加载与模型配置匹配的分词器
logger.info(f"Loading tokenizer for {args.model_id}...")
tokenizer = AutoTokenizer.from_pretrained(args.model_id,trust_remote_code=True)
logger.info(f"Tokenizer for {args.model_id} loaded successfully.")

logger.info(f"Loading configuration for {args.model_id} from ModelScope...")
qwen_config = AutoConfig.from_pretrained(args.model_id,trust_remote_code=True)
logger.info(f"Configuration for {args.model_id} loaded successfully.")

#! 修改一些config信息, 便于重新训练
qwen_config.rope_theta = 10000           #! 原来为1000000.0
qwen_config.max_position_embeddings = 4096  #! 使用 args.seq_len 更新模型配置
logger.info(f"Model's max_position_embeddings set to: {qwen_config.max_position_embeddings} from args.seq_len")

#! (2) 从配置初始化新模型（权重随机初始化）
logger.info(f"Initializing a new model from configuration: {args.model_id} (training from scratch)...")
model = AutoLigerKernelForCausalLM.from_config(config=qwen_config,trust_remote_code=True)
logger.info(f"New model initialized successfully with random weights based on {args.model_id} configuration.")

#! (3) 加载数据类
train_dataset = PretrainDataset(data_path=args.data_path)                              
#! 使用分布式采样器
if is_distributed:
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

#* 训练配置参数
gradient_accumulation_steps = 8  # 梯度累计步数，实际batch_size = batch_size * gradient_accumulation_steps

mixed_precision_dtype = torch.bfloat16  # 或者 torch.float16，根据需要配置
compile_mode = "default"  # 编译模式：'default', 'reduce-overhead', 'max-autotune'
torch.set_float32_matmul_precision('high')

# todo 开启梯度检查点 会和 torch.compile 冲突
is_gradient_checkpointing_enabled = False
logger.info("使用torch.compile时关闭梯度检查点，兼容性考虑")

#* 用于保存检查点
config_to_save = qwen_config

model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Total parameters: {total_params / 1e6:.2f} M")

#! 尝试 torch.compile 支持 (在DDP包装之前)
is_model_compiled = False
logger.info(f"尝试使用 torch.compile (模式: {compile_mode}) 编译模型以优化性能...")
try:
    # 检查 PyTorch 版本是否支持 compile
    if hasattr(torch, 'compile'):
        compile_start_time = time.time()
        # 先编译模型，再包装DDP
        model = torch.compile(model, mode=compile_mode, fullgraph=True)
        compile_end_time = time.time()
        compile_duration = compile_end_time - compile_start_time
        logger.info(f"模型编译完成！编译耗时: {compile_duration:.2f}s")
        logger.info("注意：首次前向传播可能会有额外的编译开销")
        is_model_compiled = True
    else:
        logger.warning("当前 PyTorch 版本不支持 torch.compile，跳过编译优化")
except Exception as e:
    logger.error(f"模型编译失败，将继续使用未编译版本: {e}")


#! 包装模型为DDP（在编译之后）
if is_distributed:
    logger.info("将模型包装为DistributedDataParallel...")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    logger.info("DDP包装完成")

#! 添加性能监控初始化所需参数
seq_len = args.seq_len 
vocab_size = qwen_config.vocab_size

# 估算模型FLOPS
#! 调用时传入 logger 对象
model_flops = estimate_model_flops(model, logger)

# 获取单个GPU理论峰值FLOPS
#! 调用时传入 logger 对象
single_gpu_peak_flops = get_gpu_peak_flops(logger)

# 性能统计变量
total_tokens_processed = 0
total_training_time = 0

# 在优化器配置部分添加学习率调度相关参数
lr_scheduler_config = {
    "scheduler_type": "cosine_with_warmup",  # 调度器类型
    "max_lr": 1e-4,                          # 最大学习率
    "min_lr": 1e-5,                          # 最小学习率（最大学习率的10%）
    "warmup_steps": 2500,                     # 热身步数
    "warmup_ratio": 0.25,                    # 热身比例（如果warmup_steps为None则使用此值）
}
#! 添加权重衰减参数
weight_decay = 0.1 # 你要求的权重衰减值

#! (4) 设置优化器和改进的学习率调度器
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=lr_scheduler_config["max_lr"],
    weight_decay=weight_decay, #! 应用权重衰减
    betas=(0.9, 0.95)
)

# 动态计算总训练步数
steps_per_epoch = len(train_loader) // gradient_accumulation_steps
total_training_steps = steps_per_epoch 

# 计算热身步数
if lr_scheduler_config.get("warmup_steps") is not None:
    warmup_steps = lr_scheduler_config["warmup_steps"]
else:
    warmup_steps = int(total_training_steps * lr_scheduler_config["warmup_ratio"])

logger.info(f"学习率调度配置:")
logger.info(f"  - 调度器类型: {lr_scheduler_config['scheduler_type']}")
logger.info(f"  - 最大学习率: {lr_scheduler_config['max_lr']}")
logger.info(f"  - 最小学习率: {lr_scheduler_config['min_lr']}")
logger.info(f"  - 总训练步数: {total_training_steps}")
logger.info(f"  - 热身步数: {warmup_steps}")
logger.info(f"  - 每轮步数: {steps_per_epoch}")

# 创建自定义的带热身的余弦退火调度器
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    """
    创建带热身的余弦退火学习率调度器
    
    Args:
        optimizer: 优化器
        num_warmup_steps: 热身步数
        num_training_steps: 总训练步数
        min_lr_ratio: 最小学习率与最大学习率（优化器初始学习率）的比例。
                      例如，如果 min_lr = 1e-5, max_lr = 1e-4, 则 min_lr_ratio = 0.1。
        last_epoch: 上次训练的epoch
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            
            warmup_progress = float(current_step) / float(max(1, num_warmup_steps))
            
            # 计算学习率因子。这个因子会乘以优化器中的初始学习率 (max_lr)。
            # 初始因子为 min_lr_ratio (使得初始学习率为 min_lr)。
            # 最终因子为 1 (使得最终学习率为 max_lr)。
            return min_lr_ratio + (1.0 - min_lr_ratio) * warmup_progress
        
        # 余弦退火阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

# 使用改进的调度器
min_lr_ratio = lr_scheduler_config["min_lr"] / lr_scheduler_config["max_lr"]
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
    min_lr_ratio=min_lr_ratio
)

#! 在设置完分布式训练环境后，初始化SwanLab（只在主进程）
if rank == 0:
    # 初始化SwanLab实验跟踪
    logger.info("初始化SwanLab实验跟踪...")
    # 整合所有配置到 swanlab_config
    swanlab_config = {
        # 模型相关参数
        "model_id": args.model_id,
        "total_params_M": total_params / 1e6,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        
        # 训练相关参数
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * gradient_accumulation_steps,
        "global_effective_batch_size": args.batch_size * gradient_accumulation_steps * world_size,
        "num_epochs": args.num_epochs, # 使用提前定义的 num_epochs
        
        # 学习率调度相关参数
        "learning_rate_max": lr_scheduler_config["max_lr"],
        "learning_rate_min": lr_scheduler_config["min_lr"],
        "scheduler_type": lr_scheduler_config["scheduler_type"],
        "warmup_steps": warmup_steps,
        "total_training_steps": total_training_steps,
        "steps_per_epoch": steps_per_epoch,
        "weight_decay": weight_decay,
        
        # 分布式训练参数
        "world_size": world_size,
        "is_distributed": is_distributed,
        
        # 优化相关参数
        "mixed_precision_dtype": str(mixed_precision_dtype),
        "gradient_checkpointing": is_gradient_checkpointing_enabled,
        "torch_compile": is_model_compiled,
        "compile_mode": compile_mode,
        
        # 硬件相关
        "device": device,
        "gpu_count": world_size,
    }
    swanlab_run = swanlab.init(
        # 设置项目名称
        project="bitbrain-pretrain_v1",
        # 设置实验名称（可选）
        experiment_name=f"bitbrain-pretrain-{time.strftime('%Y%m%d_%H%M%S')}",
        # 记录超参数和实验配置
        config=swanlab_config,
        # 添加实验描述
        description="使用Qwen3-0.6B架构进行预训练"
    )
    logger.info("SwanLab实验跟踪初始化完成")
else:
    swanlab_run = None

#! 初始化 GradScaler
# 根据 mixed_precision_dtype 初始化 scaler
# 如果使用 bfloat16，scaler 通常为 None
# 如果使用 float16，则需要 GradScaler 实例
if mixed_precision_dtype == torch.float16:
    scaler = GradScaler()
    logger.info("使用 torch.float16 混合精度，已初始化 GradScaler。")
elif mixed_precision_dtype == torch.bfloat16:
    scaler = None # bfloat16 不需要 GradScaler
    logger.info("使用 torch.bfloat16 混合精度，GradScaler 设置为 None。")
else:
    # 对于其他情况或未指定的情况，默认为 None，并发出警告
    scaler = None
    logger.warning(f"未知的 mixed_precision_dtype: {mixed_precision_dtype}。GradScaler 设置为 None。")

logger.info(f"训练配置:")
logger.info(f"  - 分布式训练: {is_distributed}")
if is_distributed:
    logger.info(f"  - World size: {world_size}")
    logger.info(f"  - Rank: {rank}")
logger.info(f"  - 梯度累积步数: {gradient_accumulation_steps}")
logger.info(f"  - 混合精度训练: 已启用")
logger.info(f"  - 混合精度数据类型: {mixed_precision_dtype}")
logger.info(f"  - 梯度检查点状态: {'已启用' if is_gradient_checkpointing_enabled else '尝试启用失败或模型不支持'}")
logger.info(f"  - Torch compile 状态: {'已启用 (模式: ' + compile_mode + ')' if is_model_compiled else ('尝试编译失败或PyTorch/模型不支持 (尝试模式: ' + compile_mode + ')' if hasattr(torch, 'compile') else 'PyTorch版本不支持 torch.compile')}")
logger.info(f"  - 单GPU有效批处理大小: {train_loader.batch_size * gradient_accumulation_steps}")
logger.info(f"  - 全局有效批处理大小: {train_loader.batch_size * gradient_accumulation_steps * world_size}")

#! 辅助函数：保存检查点
def save_checkpoint_helper(model, optimizer, scheduler, scaler, config_to_save, epoch, current_optimizer_step_in_epoch, global_optimizer_step, args_namespace, is_final_checkpoint=False):
    """
    辅助函数，用于保存模型检查点。
    """
    if rank != 0: # 只在主进程保存
        return

    # 确定保存目录和文件名
    if is_final_checkpoint:
        save_dir = "checkpoints" # 最终检查点保存目录
        filename_prefix = f"bitbrain_pretrain_epoch_{epoch}" # epoch是1-based
        checkpoint_filename = f"{filename_prefix}.pt"
    else:
        save_dir = args_namespace.save_dir # 中途检查点保存目录
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # 使用优化器步数命名，更清晰
        filename_prefix = f'pretrain_epoch{epoch}_optstep{current_optimizer_step_in_epoch}'
        checkpoint_filename = f'{filename_prefix}_{timestamp}.pth'
    
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, checkpoint_filename)

    # 获取模型状态字典
    model_state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

    checkpoint_data = {
        "model_config": config_to_save,  # 模型配置 (例如 qwen_config)，用于重新实例化模型结构
        "model_state_dict": model_state_dict, # 模型权重 (半精度)
        "optimizer_state_dict": optimizer.state_dict(), # 优化器状态，用于恢复优化器
        "scheduler_state_dict": scheduler.state_dict(), # 学习率调度器状态
        "scaler_state_dict": scaler.state_dict() if scaler else None, # 混合精度 GradScaler 状态
        "epoch": epoch,  
        "global_optimizer_step": global_optimizer_step, # 全局优化器步数
        "args": vars(args_namespace), # 训练启动时的命令行参数，确保恢复环境一致性
        # 以下是一些有用的元数据，用于校验和确保一致性
        # 保存时实际使用的混合精度类型 (例如 'torch.bfloat16' 或 'torch.float16')
        "mixed_precision_dtype_at_save": str(mixed_precision_dtype),
        "weight_decay_at_save": weight_decay, # 保存时使用的权重衰减值
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    original_logger_info(f"检查点已保存至: {checkpoint_path} (Epoch {epoch}, Optimizer Step in Epoch {current_optimizer_step_in_epoch})")


#! (5) 修改训练循环支持分布式训练和SwanLab记录
def train(model, optimizer, scheduler, train_loader, device,
           epoch, scaler, gradient_accumulation_steps=4, 
           tokens_per_optimizer_step_local=0): 
    model.train()

    accumulated_loss = 0 # 用于累积一个梯度更新步的loss
    
    if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)
    
    epoch_start_time = time.time()
    step_start_time = None
    first_batch_done = False
    
    # 计算当前epoch的 optimizer step 总数
    num_optimizer_steps_per_epoch = len(train_loader) // gradient_accumulation_steps

    for batch_idx, (x, y, loss_mask) in enumerate(train_loader):
        if (batch_idx) % gradient_accumulation_steps == 0:
            step_start_time = time.time()
            if not first_batch_done and is_model_compiled: # 使用 is_model_compiled
                logger.info("开始首次前向传播 (模型已编译)，可能包含编译开销...")
        
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
        
        #! 前向传播 (混合精度默认启用)
        with autocast(device_type='cuda', dtype=mixed_precision_dtype):
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps

        #! 反向传播 (混合精度默认启用)
        if scaler is not None: # 意味着是 float16
            scaler.scale(loss).backward()
        else: # bfloat16 (或 float16 但 scaler 配置不当)
            loss.backward()
        
        accumulated_loss += loss.item()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None: # float16
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isfinite(total_norm):
                    scaler.step(optimizer)
                else:
                    logger.warning(f"Skipping optimizer step at epoch {epoch}, batch_idx {batch_idx} due to non-finite gradients (norm: {total_norm}).")
                scaler.update() # 无论是否跳过step，都需要update scaler
            else: # bfloat16 (或未使用scaler)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            
            # 更新全局处理的 token 数量
            global total_tokens_processed
            total_tokens_processed += tokens_per_optimizer_step_local * world_size

            if not first_batch_done and is_model_compiled: # 使用 is_model_compiled
                first_batch_done = True
                logger.info(f"首次步骤完成 (模型已编译，包含编译开销): {step_time:.3f}s")
            
            # current_optimizer_step_in_epoch 是从1开始的当前epoch内的优化器步数
            current_optimizer_step_in_epoch = (batch_idx // gradient_accumulation_steps) + 1
            
            if current_optimizer_step_in_epoch % 10 == 0: # 每10个优化器步骤记录一次
                # 计算全局每秒处理的token数量
                tokens_per_sec_global = (tokens_per_optimizer_step_local * world_size) / step_time if step_time > 0 else 0
                
                mfu = calculate_mfu_distributed(
                    tokens_per_optimizer_step_local, step_time, model_flops, 
                    single_gpu_peak_flops, world_size
                ) if step_time > 0 and model_flops > 0 else 0
                

                actual_fwd_flops_per_sec_global = (tokens_per_optimizer_step_local * model_flops * world_size) / step_time if step_time > 0 and model_flops > 0 else 0
                current_lr = scheduler.get_last_lr()[0]
                
                if rank == 0:
                    log_info = [
                        f'Epoch:[{epoch+1}/{args.num_epochs}]', # epoch is 0-based from loop, +1 for display
                        f'Step:[{current_optimizer_step_in_epoch}/{num_optimizer_steps_per_epoch}]',
                        f'Loss:{accumulated_loss:.4f}', # This is loss for one optimization step
                        f'LR:{current_lr:.8f}',
                        f'Tokens/s (Global):{tokens_per_sec_global:.0f}',
                        f'StepTime:{step_time:.3f}s',
                        f'MFU:{mfu*100:.2f}%',
                        f'Eff FWD TFLOPS (Global):{actual_fwd_flops_per_sec_global/1e12:.2f}', # Effective FWD TFLOPS
                        f'GPUs:{world_size}'
                    ]
                    log_message = ' '.join([info for info in log_info if info])
                    original_logger_info(log_message)
                    
                    # 使用SwanLab记录训练指标
                    if swanlab_run:
                        # 计算全局步数
                        global_step = epoch * (len(train_loader) // gradient_accumulation_steps) + current_optimizer_step_in_epoch
                        
                        swanlab.log({
                            # 训练指标
                            "train/loss": accumulated_loss,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch + 1,
                            "train/step": current_optimizer_step_in_epoch,
                            "train/global_step": global_step,
                            
                            # 性能指标
                            "performance/tokens_per_second_global": tokens_per_sec_global,
                            "performance/mfu_percent": mfu * 100,
                            "performance/step_time_seconds": step_time,
                            "performance/eff_fwd_tflops_global": actual_fwd_flops_per_sec_global / 1e12,
                            
                            # 硬件指标
                            "hardware/gpu_count": world_size,
                            "hardware/batch_size": args.batch_size,
                            "hardware/effective_batch_size": args.batch_size * gradient_accumulation_steps * world_size,
                        })
            
            accumulated_loss = 0 # Reset for the next optimizer step
            
            # 计算全局优化器步数
            global_optimizer_step = (epoch * num_optimizer_steps_per_epoch) + current_optimizer_step_in_epoch

            if args.save_interval > 0 and global_optimizer_step % args.save_interval == 0 : # 使用全局优化器步数判断
                if rank == 0: # 测试和保存只在主进程进行
                    #! 保存前进行测试
                    test_model_on_prompts(model, tokenizer, device, max_seq_len=80)
                    
                    save_checkpoint_helper(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler, 
                        config_to_save=config_to_save, # qwen_config
                        epoch=epoch + 1, # Pass 1-based epoch
                        current_optimizer_step_in_epoch=current_optimizer_step_in_epoch,
                        global_optimizer_step=global_optimizer_step,
                        args_namespace=args,
                        is_final_checkpoint=False
                    )
                    # 确保模型返回训练模式
                    model.train()
                
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    # global total_training_time # total_training_time is already global
    total_training_time += epoch_duration
    
    logger.info(
        f"Epoch {epoch + 1} 完成 - " # epoch is 0-based, +1 for display
        f"用时: {epoch_duration:.2f}s"
    )

#! 训练主循环

logger.info(f"Starting pretraining for {args.num_epochs} epochs...")

# 记录总训练开始时间
total_start_time = time.time()

# 分布式训练中每个优化器步骤单GPU处理的token数量
tokens_per_optimizer_step_local = seq_len * args.batch_size * gradient_accumulation_steps

for epoch in range(args.num_epochs): # epoch will be 0, 1, ...
    logger.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")
    
    # 计算当前epoch的 optimizer step 总数
    # steps_per_epoch variable already holds len(train_loader) // gradient_accumulation_steps
    # num_optimizer_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    num_optimizer_steps_per_epoch = steps_per_epoch # Use existing variable

    train(model, optimizer, scheduler, train_loader, device, epoch, 
          scaler, #! 传递 scaler 参数
          gradient_accumulation_steps, 
          tokens_per_optimizer_step_local=tokens_per_optimizer_step_local)

    current_time = time.time()
    elapsed_time = current_time - total_start_time
    # total_tokens_processed is updated globally in train()
    overall_avg_tokens_per_sec_global = total_tokens_processed / elapsed_time if elapsed_time > 0 else 0
    
    # 使用SwanLab记录总体指标
    if rank == 0 and swanlab_run:
        swanlab.log({
            "summary/epoch": epoch + 1, # 1-based epoch
            "summary/total_time_seconds": elapsed_time,
            "summary/total_tokens_processed_global": total_tokens_processed,
            "summary/overall_avg_tokens_per_second_global": overall_avg_tokens_per_sec_global,
        })
    
    logger.info(
        f"Epoch Summary: {epoch + 1}, "
        f"总用时: {elapsed_time:.2f}s, "
        f"总处理tokens (Global): {total_tokens_processed:,}, "
        f"整体平均吞吐量 (Global): {overall_avg_tokens_per_sec_global:.0f} tokens/s"
    )

    if rank == 0:
        # 计算最后一个全局优化器步数 (用于最终检查点名称和记录)
        # epoch is 0-based here
        final_global_optimizer_step = (epoch + 1) * num_optimizer_steps_per_epoch

        save_checkpoint_helper(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler, #! 传递 scaler 参数
            config_to_save=config_to_save, # qwen_config
            epoch=epoch + 1, # Pass 1-based epoch for filename and logging
            # For final checkpoint, current_optimizer_step_in_epoch can be num_optimizer_steps_per_epoch
            current_optimizer_step_in_epoch=num_optimizer_steps_per_epoch, 
            global_optimizer_step=final_global_optimizer_step,
            args_namespace=args,
            is_final_checkpoint=True
        )

# 训练完成后的清理
if rank == 0 and swanlab_run:
    logger.info("训练完成，正在结束SwanLab实验记录...")
    # SwanLab会自动在进程结束时完成实验

if is_distributed:
    dist.destroy_process_group()


