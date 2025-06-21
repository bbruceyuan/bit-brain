import torch

#! 模型预训练时测试
def test_model_on_prompts(model, tokenizer, device, max_seq_len=200):
    model.eval()
    prompts = [
        '马克思主义基本原理',
        '人类大脑的主要功能',
        '万有引力定律是',
        '世界上最高的山是',
        'The cat sat on the',
        'The capital of the United States is',
        'The color of grass is usually',
    ]
    model.eval()
    # 兼容 DDP
    if hasattr(model, "module"):
        gen_model = model.module
    else:
        gen_model = model
    
    test_results = {}  # 新增：存储测试结果
    
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(device)
        with torch.no_grad():
            generated_ids = gen_model.generate(
                inputs["input_ids"],
                max_new_tokens=max_seq_len,
                num_return_sequences=1,
                do_sample=True,
                attention_mask=inputs["attention_mask"],
                pad_token_id=151643,
                top_p=0.95,
                top_k=20,
                temperature=0.6,
                bos_token_id=151643,
                eos_token_id=[151645, 151643],
            )
            response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"\n[测试] 输入: {prompt}\n[测试] 输出: {response}\n")
            test_results[f"测试/输入_{prompt[:10]}"] = response  # 记录测试结果
    
    model.train()
    torch.cuda.empty_cache()  # 显式释放显存



