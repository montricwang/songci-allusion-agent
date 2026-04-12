import pandas as pd
import re
from pathlib import Path

# --- 常量与路径 ---
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "raw_data" / "Poetry" / "唐.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "tang_sentences.csv"


def split_sentences(text: str):
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"[，。！？；]", text)
    return [p.strip() for p in parts if p.strip()]


def main():
    df = pd.read_csv(INPUT_FILE)
    rows = []
    sentence_id = 0

    for _, row in df.iterrows():
        title = str(row["题目"]).strip()
        dynasty = str(row["朝代"]).strip()
        author = str(row["作者"]).strip()
        content = row["内容"]

        sentences = split_sentences(content)
        for sent in sentences:
            rows.append(
                {
                    "sentence_id": sentence_id,
                    "sentence": sent,
                    "title": title,
                    "dynasty": dynasty,
                    "author": author,
                }
            )
            sentence_id += 1

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"运行成功，保存 {len(out_df)} 句到 {OUTPUT_FILE} 中")


if __name__ == "__main__":
    main()
