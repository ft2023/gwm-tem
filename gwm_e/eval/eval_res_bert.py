from bert_score import score
import pandas as pd
import json

# 读取数据
with open('hopimage-5/AgentClinic.jsonl') as f:
    lines = f.readlines()

# 解析数据
preds = []
refs = []
for line in lines:
    data = json.loads(line)
    preds.append(data['text'])
    refs.append(data['gt'])

# 计算BERT score
P, R, F1 = score(preds, refs, lang='en', verbose=True)

# 计算平均分数
avg_P = P.mean().item()
avg_R = R.mean().item()
avg_F1 = F1.mean().item()

print(f'Precision: {avg_P:.4f}')
print(f'Recall: {avg_R:.4f}')
print(f'F1: {avg_F1:.4f}')