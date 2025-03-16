import torch
from torch.utils.data import Dataset
import tiktoken
from datasets import load_dataset
from loguru import logger


class PretrainDataset(Dataset):
    def __init__(self, max_lines=1000000000, block_size=512):
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.max_lines = max_lines

        self.eos_token = self.enc.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]
        logger.info("Loading dataset from fjcanyue/wikipedia-zh-cn")
        dataset = load_dataset("fjcanyue/wikipedia-zh-cn")

        self.encoded_data = []

        raw_data = []

        for i, item in enumerate(dataset["train"]):
            if i >= self.max_lines:
                break
            raw_data.append(item["text"])

        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])

        # 将长文本分割成训练样本
        for i in range(0, len(full_encoded), self.block_size):
            # 多取一个 Token 作为目标
            chunk = full_encoded[i : i + self.block_size + 1]
            # 如果长度不够，用 eos_token 填充
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
