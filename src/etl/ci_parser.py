import json
import re
import hashlib
from typing import List, Dict
import opencc


class CiJSONParser:
    def __init__(self):
        # t2s: 繁体转简体（中国大陆标准）
        self.converter = opencc.OpenCC("t2s")
        # 正则：匹配非标点字符序列 + 可选的一个标点
        self.split_pattern = re.compile(r"([^，、。！？；]+)([，、。！？；]?)")

    def _generate_id(self, author: str, title: str, full_content: str) -> str:
        """使用作者+标题+全文生成唯一MD5指纹"""
        fingerprint = f"{author}_{title}_{full_content}"
        return hashlib.md5(fingerprint.encode("utf-8")).hexdigest()[:16]

    def parse_entry(self, entry: Dict) -> List[Dict]:
        raw_author = entry.get("author", "未知")
        raw_title = entry.get("rhythmic", "无题")
        paragraphs = entry.get("paragraphs", [])

        # 1. 转换与对齐
        author = self.converter.convert(raw_author)
        title = self.converter.convert(raw_title)

        # 2. 生成全文（简体）用于哈希和切分
        raw_full_text = "".join(paragraphs)
        simplified_full_text = self.converter.convert(raw_full_text)

        # 3. 生成父级 ID
        parent_id = self._generate_id(author, title, simplified_full_text)

        # 4. 原子切分
        matches = self.split_pattern.findall(simplified_full_text)

        atoms = []
        for idx, (text_content, punctuation) in enumerate(matches):
            text_content = text_content.strip()
            if not text_content:
                continue

            atoms.append(
                {
                    "content": text_content,  # 纯净文本：Embedding 核心
                    "author": author,
                    "title": title,
                    "full_content": simplified_full_text,  # 完整简体内容，方便溯源
                    "genre": "ci",
                    "meta": {
                        "punc": punctuation,  # 标点符号
                        "parent_id": parent_id,  # 哈希指纹
                        "line_idx": idx,  # 句序
                    },
                }
            )
        return atoms


# --- 完整样例测试 ---
if __name__ == "__main__":
    # 使用你之前给出的钱惟演经典词作作为样例
    test_entry = {
        "author": "钱惟演",
        "paragraphs": [
            "城上风光莺语乱。",
            "城下烟波春拍岸。",
            "绿杨芳草几时休，泪眼愁肠先已断。",
        ],
        "rhythmic": "木兰花",
    }

    parser = CiJSONParser()
    results = parser.parse_entry(test_entry)

    print(f"\n{'=' * 20} ETL 解析完整输出 {'=' * 20}\n")

    # 我们打印前两个原子结果来看看结构
    for i, atom in enumerate(results):
        print(f"--- 原子句 [{i}] ---")
        # 使用 json.dumps 只是为了让输出格式更漂亮（带缩进）
        print(json.dumps(atom, indent=4, ensure_ascii=False))
        print()

    print(f"{'=' * 60}")
