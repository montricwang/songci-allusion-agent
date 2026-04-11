from transformers import BertModel, BertTokenizer
import torch
import os

# 获取当前脚本所在目录的父目录，定位到 models 文件夹
# 这样无论你在哪里执行脚本，路径都不会错
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../models/bert-ccpoem")

# 核心加载代码（基本保持原样，只改路径变量）
tokenizer = BertTokenizer.from_pretrained(model_path) 
model = BertModel.from_pretrained(model_path)

# 推理部分
text = "一行白鹭上青天"
input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0) 

# 开启 no_grad 可以让你的集成显卡压力小一点
with torch.no_grad():
    outputs = model(input_ids)
    # 这里的 outputs 通常是一个对象，取它的第一个元素（last_hidden_state）
    last_hidden_state = outputs[0] 
    sen_emb = torch.mean(last_hidden_state, 1)[0] 

print("-" * 30)
print(f"输入文本: {text}")
print(f"向量维度: {sen_emb.shape}")
print(f"向量预览 (前3位): {sen_emb[:3].tolist()}")
print("-" * 30)