import json
from collections import defaultdict

with open("dpo_preference_pairs.json", 'r', encoding='utf-8') as f:
    pairs = json.load(f)

# 按prompt分组
prompt_answers = defaultdict(list)
for pair in pairs:
    prompt = pair['prompt']
    chosen = pair['chosen']
    rejected = pair['rejected']
    prompt_answers[prompt].append((chosen, rejected))

# 构建排序链（通过传递闭包）
ranked_data = []
for prompt, pair_list in prompt_answers.items():
    # 收集所有答案
    answers_set = set()
    for c, r in pair_list:
        answers_set.add(c)
        answers_set.add(r)
    answers = list(answers_set)

    # 定义比较函数
    from functools import cmp_to_key


    def compare(a, b):
        for c, r in pair_list:
            if c == a and r == b:
                return -1  # a排在b前
            if c == b and r == a:
                return 1
        return 0


    sorted_answers = sorted(answers, key=cmp_to_key(compare))
    # 确保每个列表至少有2个答案
    if len(sorted_answers) >= 2:
        ranked_data.append({
            "prompt": prompt,
            "ranked_answers": sorted_answers
        })

with open("dpo_ranked_answers.json", 'w', encoding='utf-8') as f:
    json.dump(ranked_data, f, ensure_ascii=False, indent=2)

print(f"构建了 {len(ranked_data)} 个问题的排序列表")