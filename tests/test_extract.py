# test_extract.py
from openai import OpenAI
from src.schema import AgentState
from src.skills import extract_allusions

client = OpenAI(
    api_key="sk-fef335f3ddcd4adbae61d85c033facc7",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

state: AgentState = {
    "input_text": "纵有魏珠照乘，未买得、流年住。争如盛饮流霞，醉偎琼树。",
    "extracted_data": [],
    "validation_results": [],
    "final_report": None,
    "error": None,
}

result = extract_allusions(state, client)

print(f"错误：{result['error']}")
print(f"找到典故数：{len(result['extracted_data'])}")
for a in result["extracted_data"]:
    print(f"\n【{a['phrase']}】{a['source']}")
    print(f"  {a['original'][:50]}...")
