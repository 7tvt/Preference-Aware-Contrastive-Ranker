import json, random

# 加载原始问答数据和政策库
with open("pacr_training_data.json", 'r', encoding='utf-8') as f:
    samples = json.load(f)
with open("policy_library.json", 'r', encoding='utf-8') as f:
    policies = json.load(f)
policy_dict = {p['doc_no']: p['full_text'] for p in policies}

pairs = []
for s in samples:
    question = s['question']
    good_answer = s['answer']  # 标准答案，作为最优
    # 构造3-4个劣后答案：用随机别的政策摘要
    other_policies = [p for p in policies if p['doc_no'] != s['positive_policy']['id']]
    if not other_policies:
        continue
    negative_pols = random.sample(other_policies, min(4, len(other_policies)))
    for npol in negative_pols:
        bad_answer = f"依据《{npol['title']}》：{npol['full_text'][:300]}"
        pairs.append({
            "prompt": question,
            "chosen": good_answer,
            "rejected": bad_answer
        })

with open("dpo_preference_pairs_fixed.json", 'w', encoding='utf-8') as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)
print(f"生成 {len(pairs)} 个有区分度的偏好对")