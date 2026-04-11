import faiss
import numpy as np
from pathlib import Path

# 路径设置
ROOT_DIR = Path(__file__).resolve().parent.parent
NPY_PATH = ROOT_DIR / "data" / "tang_vectors.npy"
INDEX_SAVE_PATH = ROOT_DIR / "data" / "tang_poetry.index"

# 1. 加载向量 (float32 是 FAISS 的标配)
print("正在读取向量数据...")
vectors = np.load(NPY_PATH).astype("float32")

# 2. 创建索引
# IndexFlatL2 指的是使用“欧式距离”进行精确搜索
# 512 是我们 BERT 模型的维度
d = 512
index = faiss.IndexFlatL2(d)

# 3. 把向量塞进索引
print("正在构建 FAISS 索引...")
index.add(vectors)

# 4. 保存索引文件到硬盘
faiss.write_index(index, str(INDEX_SAVE_PATH))
print(f"🎉 向量库构建完成！已存至: {INDEX_SAVE_PATH}")
