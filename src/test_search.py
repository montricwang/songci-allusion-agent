import faiss
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from pathlib import Path


class PoetrySearcher:
    def __init__(self):
        # 1. 设置路径 (使用我们之前定下的 pathlib 信仰)
        self.root_dir = Path(__file__).resolve().parent.parent
        self.model_path = self.root_dir / "models" / "bert-ccpoem"
        self.csv_path = self.root_dir / "data" / "tang_sentences.csv"
        self.index_path = self.root_dir / "data" / "tang_poetry.index"

        # 2. 加载模型与分词器
        print("正在加载 BERT 模型...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertModel.from_pretrained(self.model_path)
        self.model.eval()

        # 3. 加载 FAISS 索引
        print("正在加载 FAISS 向量库...")
        self.index = faiss.read_index(str(self.index_path))

        # 4. 加载 CSV 账本 (只需读取，建议指定类型省内存)
        print("正在加载 CSV 账本...")
        self.df = pd.read_csv(self.csv_path)

    def search(self, query_text, top_k=5):
        # 第一步：把用户说的话变成 512 维向量
        inputs = self.tokenizer(
            query_text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 同样使用 mean pooling 保证维度一致
            query_vec = (
                torch.mean(outputs.last_hidden_state, dim=1).numpy().astype("float32")
            )

        # 第二步：去向量库里“瞬移”寻找最接近的邻居
        # D 是距离，I 是索引（对应 CSV 的行号）
        distances, indices = self.index.search(query_vec, top_k)

        # 第三步：去 CSV 账本里按 ID 拿人
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            row = self.df.iloc[idx]
            results.append(
                {
                    "score": float(dist),
                    "sentence": row["sentence"],
                    "author": row["author"],
                    "title": row["title"],
                }
            )
        return results


# --- 简单测试 ---
if __name__ == "__main__":
    searcher = PoetrySearcher()
    while True:
        text = input("\n请输入你想搜的意境 (输入 q 退出): ")
        if text == "q":
            break

        hits = searcher.search(text)
        print("\n为你找到以下匹配:")
        for h in hits:
            print(
                f"[{h['score']:.4f}] {h['sentence']} —— {h['author']}《{h['title']}》"
            )
