import torch
# 我们不再从 loguru 直接导入 logger，而是期望它作为参数传入
# from loguru import logger 

def get_gpu_peak_flops(logger):
    """
    获取单个GPU理论峰值FLOPS
    针对混合精度训练优化，优先使用Tensor Core性能
    """
    if not torch.cuda.is_available():
        return 0
    
    # 获取GPU信息
    gpu_name = torch.cuda.get_device_name()
    
    # GPU FLOPS估算（优先使用Tensor Core性能用于混合精度训练）
    gpu_flops_dict = {
        'V100': 125e12,     # 125 TFLOPS (Tensor Core FP16)
        'A100': 312e12,     # 312 TFLOPS (Tensor Core BF16/FP16)
        'H100': 989e12,     # 989 TFLOPS (Tensor Core FP16)
        '3090': 142e12,     # 142 TFLOPS (Tensor Core FP16)
        '4090': 330e12,     # 330 TFLOPS (Tensor Core BF16/FP16)
        'T4': 130e12,       # 130 TFLOPS (Tensor Core FP16)
    }
    
    # 尝试匹配GPU型号
    for gpu_type, flops in gpu_flops_dict.items():
        if gpu_type in gpu_name:
            logger.info(f"检测到GPU: {gpu_name}, 单卡Tensor Core峰值FLOPS: {flops/1e12:.1f} TFLOPS")
            return flops
    
    # 如果未找到匹配的GPU，使用保守估算
    logger.warning(f"未识别的GPU型号: {gpu_name}, 使用默认估算值")
    return 100e12  # 默认值100 TFLOPS

def estimate_model_flops(model, logger):
    """
    估算Transformer模型每个token的理论FLOPs
    使用更精确的公式: 6 * N * (1 + s/(12*h) + V/(16*L*h))
    
    参数:
        model: 模型实例
        logger: 日志记录器实例
    
    返回:
        每个token的理论FLOPs数值
    """
    # 1. 获取模型参数数量
    if hasattr(model, 'module'):
        # 处理DDP（DistributedDataParallel）包装的模型
        n_params = sum(p.numel() for p in model.module.parameters())
        model_config = model.module.config
    else:
        n_params = sum(p.numel() for p in model.parameters())
        model_config = model.config

    # 2. 尝试从模型配置中获取详细参数
    try:
        # s: 序列长度, 根据用户提供的值
        s = 2048
        # V: 词汇表大小
        V = model_config.vocab_size
        # L: 模型层数
        L = model_config.num_hidden_layers
        # h: 在此公式中, h 代表模型的隐藏层维度 (hidden_size), 而不是单个头的维度
        h = model_config.hidden_size
        
        logger.info(f"使用详细公式计算FLOPs, 参数为: V={V}, L={L}, h(hidden_size)={h}, s={s}")
        
        # 3. 根据新公式计算每个token的FLOPs
        # N 在这里是 n_params
        # flops_per_token = 6 * n_params * (1 + s/(12*h) + V/(16*L*h))
        flops_per_token = 6 * n_params * (1 + s / (12 * h) + V / (16 * L * h))
        logger.info(f"估算的每个token的训练计算量 (FLOPs per token, based on detailed formula): {flops_per_token:.0f}")

    except AttributeError as e:
        # 如果无法从config中获取所需参数，则回退到简单估算
        logger.warning(f"无法从模型配置中获取详细参数 ({e})。将使用 6 * n_params 近似计算FLOPs。")
        flops_per_token = 6 * n_params
        logger.info(f"估算的每个token的训练计算量 (FLOPs per token, based on 6 * n_params approximation): {flops_per_token:.0f}")

    return flops_per_token

def calculate_mfu_distributed(tokens_processed, step_time, flops_per_token, single_gpu_peak_flops, world_size):
    """
    计算分布式训练的MFU（Model FLOPS Utilization）
    
    参数:
        tokens_processed: 单个进程处理的token数量
        step_time: 处理这些token所用的时间 (秒)
        flops_per_token: 每个token的理论FLOPs
        single_gpu_peak_flops: 单个GPU的理论峰值FLOPS/秒
        world_size: 参与训练的GPU数量 (虽然在此函数中未直接使用world_size来调整MFU，但保留它以明确上下文)
    
    返回:
        MFU利用率（0-1之间的数值）
    """
    if step_time <= 0 or single_gpu_peak_flops <= 0 or flops_per_token <= 0:
        return 0
    
    # 计算单个GPU的实际FLOPS
    single_gpu_actual_flops = (tokens_processed * flops_per_token) / step_time
    
    # 计算单个GPU的MFU
    single_gpu_mfu = single_gpu_actual_flops / single_gpu_peak_flops
    
    # 分布式训练中，MFU是基于单个GPU计算的，因为每个GPU处理相同的计算量
    # 确保MFU不超过100%
    return min(single_gpu_mfu, 1.0)
