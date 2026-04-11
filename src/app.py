import streamlit as st
import faiss
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from pathlib import Path

# --- 页面配置 ---
st.set_page_config(page_title="唐诗语义搜索引擎", layout="wide")
st.title("🏮 唐诗语义搜索引擎")
st.markdown("基于 `BERT-ccpoem` 与 `FAISS` 的向量检索系统")

# --- 常量与路径 ---
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "bert-ccpoem"
CSV_PATH = ROOT_DIR / "data" / "tang_sentences.csv"
INDEX_PATH = ROOT_DIR / "data" / "tang_poetry.index"


# --- 缓存加载逻辑 (避免每次刷新页面都重载模型) ---
@st.cache_resource
def load_resources():
    st.info("正在加载语义模型与向量库，请稍候...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertModel.from_pretrained(MODEL_PATH)
    model.eval()
    index = faiss.read_index(str(INDEX_PATH))
    df = pd.read_csv(CSV_PATH)
    return tokenizer, model, index, df


tokenizer, model, index, df = load_resources()


# --- 搜索核心逻辑 ---
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).numpy().astype("float32")


# --- 侧边栏设置 ---
with st.sidebar:
    st.header("搜索设置")
    top_k = st.slider("展现结果数量", 1, 20, 5)
    st.divider()
    st.write("当前索引容量: ", len(df), "条诗句")

# --- 主界面交互 ---
query = st.text_input(
    "输入你想搜的意境、词句或心情：", placeholder="例如：红蜡泪、深夜失眠、金戈铁马"
)

if query:
    query_vec = get_embedding(query)
    distances, indices = index.search(query_vec, k=top_k)

    st.subheader(f"🔍 为您找到以下最接近『{query}』的诗句：")

    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx]
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                # 距离越小，相似度越高，这里简单转换一下显示
                st.metric("相似度得分", f"{dist:.2f}")
            with col2:
                st.markdown(f"### {row['sentence']}")
                st.caption(f"—— {row['author']} · 《{row['title']}》")
            st.divider()
