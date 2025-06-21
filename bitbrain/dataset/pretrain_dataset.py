import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
import os
from loguru import logger


class HFLoadedDataset(Dataset):
    """一个包装类，用于处理从磁盘加载的Hugging Face数据集"""
    def __init__(self, data_path, tokenizer):
        # 调用父类的初始化方法，这是良好编程习惯的一部分
        super().__init__()
        
        # 检查 data_path 是否是一个实际存在的目录
        # os.path.isdir() 会告诉我们这个路径是不是一个文件夹
        if not os.path.isdir(data_path):
            # 如果 data_path 不是一个目录，我们就认为它是一个单独的数据集文件夹，按原方式加载
            logger.info(f"Loading single pre-tokenized dataset from {data_path}...")
            self.hf_dataset = load_from_disk(data_path)
        else:
            # 如果 data_path 是一个目录，我们就遍历它的子目录来加载所有数据集
            logger.info(f"Loading and concatenating datasets from subdirectories in {data_path}...")
            
            # 使用列表推导式高效地找到所有子目录的完整路径
            # 1. os.listdir(data_path) 列出父文件夹下的所有文件和文件夹名
            # 2. for d in ... 遍历每一个名字
            # 3. os.path.join(data_path, d) 将父文件夹路径和名字拼接成完整路径
            # 4. if os.path.isdir(...) 只保留那些本身也是文件夹的路径
            sub_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
            
            if not sub_dirs:
                # 如果没有找到任何子目录，说明没有数据，抛出错误并提醒用户
                raise ValueError(f"No subdirectories found in {data_path} to load datasets from.")

            # 创建一个空列表，像一个空篮子，用来存放从每个子目录加载的数据集
            datasets_to_concat = []
            for sub_dir in sub_dirs:
                try:
                    # 打印日志，让我们知道程序正在做什么
                    logger.info(f"  -> Loading dataset from {sub_dir}...")
                    # 加载数据集
                    dataset = load_from_disk(sub_dir)
                    # 将加载好的数据集放进我们的"篮子"里
                    datasets_to_concat.append(dataset)
                    logger.info(f"  -> Successfully loaded {len(dataset)} samples from {sub_dir}.")
                except Exception as e:
                    # 如果某个子目录加载失败（比如文件夹是空的或格式不对），
                    # 我们只打印一个警告信息并跳过它，而不是让整个程序崩溃。这让代码更健壮。
                    logger.warning(f"  -> Could not load dataset from {sub_dir}. Skipping. Error: {e}")
            
            if not datasets_to_concat:
                # 如果"篮子"还是空的，说明一个有效的数据集都没加载到，也抛出错误
                raise ValueError(f"No valid datasets could be loaded from subdirectories of {data_path}.")

            logger.info(f"Concatenating {len(datasets_to_concat)} datasets...")
            # 使用 concatenate_datasets 函数将"篮子"里的所有数据集合并成一个大的数据集
            self.hf_dataset = concatenate_datasets(datasets_to_concat)
            logger.info("All datasets concatenated successfully.")

        self.tokenizer = tokenizer
        
        # 关键优化：在这里一次性设置输出格式为 PyTorch 张量
        # 这一步不需要改变，它会对我们最终合并好的大数据集生效
        self.hf_dataset.set_format(
            type='torch', 
            columns=['input_ids', 'attention_mask']
        )
        
        logger.info("Dataset loaded and formatted for PyTorch successfully.")
        logger.info(f"Total dataset size after concatenation: {len(self.hf_dataset)}")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # 现在 hf_dataset[idx] 会直接返回 PyTorch 张量，无需手动转换
        item = self.hf_dataset[idx]
        
        # input_ids 和 attention_mask 已经是 torch.Tensor 了
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        
        # 创建 labels
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return input_ids, labels, attention_mask
