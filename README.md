# BitBrain🧠
最少使用 3090 即可训练自己的比特大脑🧠（进行中）. Train your own BitBrain with just an RTX 3090 minimum.(Stay tuned)

## 运行
- step0: 使用 python3.12 （可选）
- step1: 首先安装 uv。见[链接](https://docs.astral.sh/uv/getting-started/installation/)
- step2: 初始化虚拟环境 `uv venv`
- step3: `uv install`
- **step4** run: `uv run python -m bitbrain.train.pretrain` 
```shell
# 以下仅仅是演示
2025-03-16 10:06:35.098 | INFO     | __main__:<module>:23 - Total parameters: 120.116736 M
2025-03-16 10:07:01.647 | INFO     | __main__:train:51 - Epoch: 0, Batch: 0, Loss: 10.9204
2025-03-16 10:07:18.789 | INFO     | __main__:train:51 - Epoch: 0, Batch: 100, Loss: 4.0074
2025-03-16 10:07:36.940 | INFO     | __main__:<module>:70 - Epoch: 0, Train Loss: 4.3673, Val Loss: 3.5040
2025-03-16 10:07:39.148 | INFO     | __main__:train:51 - Epoch: 1, Batch: 0, Loss: 3.5021
2025-03-16 10:07:56.363 | INFO     | __main__:train:51 - Epoch: 1, Batch: 100, Loss: 3.5192
2025-03-16 10:08:14.195 | INFO     | __main__:<module>:70 - Epoch: 1, Train Loss: 3.4028, Val Loss: 3.2691
```

## 备注
本项目还在施工中，目前仅支持pretrain，项目结构还会重构~


> 最后欢迎大家使用 [AIStackDC](https://aistackdc.com/phone-register?invite_code=D872A9) 算力平台，主打一个便宜方便（有专门的客服支持），如果你需要的话可以使用我的邀请链接: [https://aistackdc.com/phone-register?invite_code=D872A9](https://aistackdc.com/phone-register?invite_code=D872A9)
