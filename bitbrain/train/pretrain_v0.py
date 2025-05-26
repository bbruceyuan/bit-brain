#！ 使用FP32版本训练的最小代码版本
import os
import sys
import argparse
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch.utils.data import DataLoader
from modelscope import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from bitbrain.dataset.pretrain_dataset import PretrainDataset
from loguru import logger


train_dataset = PretrainDataset()

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


model_id = "Qwen/Qwen2-0.5B"

#! (1) 加载模型配置
logger.info(f"Loading configuration for {model_id} from ModelScope...")
qwen_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
logger.info(f"Configuration for {model_id} loaded successfully.")

#* 打印配置中的最大位置嵌入（通常是最大上下文长度）
logger.info(f"Model's max_position_embeddings from config: {qwen_config.max_position_embeddings}")

#! (2) 从配置初始化新模型（权重随机初始化）
logger.info(f"Initializing a new model from configuration: {model_id} (training from scratch)...")
model = AutoModelForCausalLM.from_config(config=qwen_config)
logger.info(f"New model initialized successfully with random weights based on {model_id} configuration.")

#! (3) 加载与模型配置匹配的分词器
logger.info(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
logger.info(f"Tokenizer for {model_id} loaded successfully.")

#* 用于保存检查点 
config_to_save = qwen_config

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Total parameters: {total_params / 1e6:.2f} M")

#! (4) 设置优化器和学习率调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000)


#! (5) 训练循环
def train(model, optimizer, scheduler, train_loader, val_loader, device, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        # 将数据移到设备上
        x, y = x.to(device), y.to(device)

        #! 前向传播 - Qwen2模型的调用方式
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（推荐用于大模型训练）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        # 调整学习率
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            logger.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    return total_loss


def eval(model, val_loader, device):
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            # Qwen2模型的验证调用
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            val_loss += loss.item()
    return val_loss


# 训练主循环
num_epochs = 2
logger.info(f"Starting pretraining for {num_epochs} epochs...")

for epoch in range(num_epochs):
    logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
    
    train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device, epoch)
    val_loss = eval(model, val_loader, device)
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    logger.info(
        f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
    )

    # 保存模型检查点
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config_to_save,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    }
    
    # 确保checkpoints目录存在
    import os
    os.makedirs("checkpoints", exist_ok=True)
    
    # 保存每个epoch的模型
    checkpoint_path = f"checkpoints/qwen2_pretrain_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

logger.info("Pretraining completed!")

