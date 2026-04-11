import os
from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F

# 1. 获取当前脚本 (find_verse.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
# 2. 获取 src 文件夹的路径
src_dir = os.path.dirname(current_script_path)
# 3. 获取项目根目录 (src 的上一层)
root_dir = os.path.dirname(src_dir)
# 4. 准确定位到 models 文件夹
model_path = os.path.join(root_dir, "models", "bert-ccpoem")

# 打印一下，方便排查
print(f"正在尝试从这里加载模型: {os.path.abspath(model_path)}")

# 接下来再加载，就不会报错了
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)


def get_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # 取平均值作为句子向量
    return torch.mean(outputs[0], dim=1)


# --- 模拟“大海” (我们的数据库) ---
database_poems = [
    "白鹭一行登碧霄",
    "明月照大江",
    "举杯邀明月",
    "一双白鹭向南飞",
    "花间一壶酒",
]

# 预先算好库里诗句的向量
database_vectors = torch.cat([get_vector(p) for p in database_poems])

# --- 模拟“捞针” (用户输入) ---
query = "一行白鹭上青天"
query_vector = get_vector(query)

# 计算相似度 (余弦相似度)
# F.cosine_similarity 会计算 query 和库里每一句的得分
scores = F.cosine_similarity(query_vector, database_vectors)

# 找出得分最高的前 3 名
top_k_values, top_k_indices = torch.topk(scores, k=3)

print(f"查询句：{query}\n")
print("搜索结果：")
for i in range(len(top_k_indices)):
    idx = top_k_indices[i]
    print(f"第{i + 1}名: {database_poems[idx]} (相似度: {top_k_values[i]:.4f})")
