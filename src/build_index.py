import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm  # 这是一个带百分比进度条的库，建议 pip install tqdm
from pathlib import Path

# --- 1. 环境准备与路径定位 ---
# Path(__file__) 拿到当前文件
# .resolve() 拿到绝对路径
# .parent 拿到 src 文件夹，再点一次 .parent 拿到项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent

# 2. 直接使用斜杠 / 拼接路径（这是 pathlib 的魔法，非常直观）
MODEL_PATH = ROOT_DIR / "models" / "bert-ccpoem"
INPUT_CSV = ROOT_DIR / "data" / "tang_sentences.csv"
OUTPUT_NPY = ROOT_DIR / "data" / "tang_vectors.npy"

# --- 2. 加载“翻译官” (BERT) ---
print("正在加载模型，请稍候...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertModel.from_pretrained(MODEL_PATH)
model.eval()  # 设置为评价模式（不训练，只用它来转换向量）

# --- 3. 准备数据 ---
df = pd.read_csv(INPUT_CSV)
sentences = df["sentence"].tolist()


# --- 4. 批量处理函数 (核心) ---
def get_batch_vectors(text_list):
    # 将文本转为模型能看懂的数字 ID
    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        padding=True,  # 长度不够的补齐
        truncation=True,  # 太长的截断
        max_length=64,  # 诗句一般不长，64 足够了
    )

    with torch.no_grad():  # 告诉电脑不要算梯度（省显存/内存）
        outputs = model(**inputs)

    # 按照清华 README 的建议：取 mean pooling
    # 即把所有汉字的特征取平均值，代表整句话的意思
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings.numpy()


# --- 5. 开始循环处理 ---
batch_size = 32  # 每次处理 32 句，就像去超市结账一次刷 32 件商品
all_vectors = []

print(f"开始生成向量，总计 {len(sentences)} 句...")
for i in tqdm(range(0, len(sentences), batch_size)):
    # 截取这一批的句子
    batch_sentences = sentences[i : i + batch_size]
    # 转换成向量
    batch_vecs = get_batch_vectors(batch_sentences)
    # 存入列表
    all_vectors.append(batch_vecs)

# --- 6. 缝合与保存 ---
# 把零散的小数组堆叠成一个巨大的矩阵
final_matrix = np.vstack(all_vectors)
# 保存到硬盘，格式是 .npy
np.save(OUTPUT_NPY, final_matrix)

print(f"🎉 成功！向量库已存至: {OUTPUT_NPY}")
print(f"最终矩阵形状: {final_matrix.shape}")  # 应该是 [总行数, 512]
