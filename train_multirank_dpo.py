import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# ================== 配置 ==================
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "dpo_preference_pairs_fixed.json"   # 新数据，区分度更高
OUTPUT_DIR = "./dpo_multirank_corrected"
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 1e-5
BETA = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 辅助函数 ==================
def compute_logprob(model, tokenizer, prompt, answer, max_length=256):
    """计算给定答案在模型下的对数概率（总和）"""
    text = prompt + " " + answer
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    # loss 是平均交叉熵，乘以 token 数得到总对数似然（负数），我们取绝对值
    return -outputs.loss * inputs["input_ids"].size(1)

def build_ranked_samples(data_path, tokenizer, model):
    """读取二元对数据，用 model 打分并为每个问题构建排序列表"""
    with open(data_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    prompt_pairs = defaultdict(list)
    for p in pairs:
        prompt_pairs[p['prompt']].append((p['chosen'], p['rejected']))

    ranked_samples = []
    for prompt, pair_list in prompt_pairs.items():
        # 收集所有唯一答案
        answers_set = set()
        for c, r in pair_list:
            answers_set.add(c)
            answers_set.add(r)
        answers = list(answers_set)

        if len(answers) < 2:
            continue

        # 用当前模型（或 ref_model）为每个答案打分
        logprobs = [compute_logprob(model, tokenizer, prompt, a).item() for a in answers]
        # 按 logp 降序排列（模型更喜欢的排在前面）
        sorted_answers = [ans for _, ans in sorted(zip(logprobs, answers), key=lambda x: x[0], reverse=True)]

        # 保留至少2个答案才能形成偏好对
        if len(sorted_answers) >= 2:
            ranked_samples.append({
                'prompt': prompt,
                'ranked_answers': sorted_answers
            })

    print(f"从 {len(pairs)} 个二元对中构建了 {len(ranked_samples)} 个有效排序样本（基于模型偏好）")
    return ranked_samples

# ================== 数据集 ==================
class RankedAnswersDataset(Dataset):
    def __init__(self, ranked_data):
        self.data = ranked_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ================== 多阶 DPO 损失 ==================
def multi_rank_dpo_loss(model, ref_model, tokenizer, prompt, ranked_answers, beta=0.1):
    if len(ranked_answers) < 2:
        return torch.tensor(0.0, requires_grad=True, device=DEVICE)

    def get_log_prob(text):
        inputs = tokenizer(prompt + " " + text, return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
        outputs = model(**inputs, labels=inputs["input_ids"])
        return -outputs.loss * inputs["input_ids"].size(1)

    # 待优化模型的 logp
    logprobs = [get_log_prob(ans) for ans in ranked_answers]
    # 参考模型 logp（冻结，不更新梯度）
    with torch.no_grad():
        ref_logprobs = [get_log_prob(ans).detach() for ans in ranked_answers]

    loss = 0.0
    K = len(ranked_answers)
    for i in range(K - 1):
        # 模型对数概率比率差
        pi_ratio = logprobs[i] - ref_logprobs[i]
        pj_ratio = logprobs[i+1] - ref_logprobs[i+1]
        loss += -F.logsigmoid(beta * (pi_ratio - pj_ratio))
    return loss / (K - 1)

# ================== 训练 ==================
def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 使用参考模型的偏好构建排序列表（也可以使用 ref_model，这里就用 ref_model 保证一致性）
    ranked_data = build_ranked_samples(DATA_PATH, tokenizer, ref_model)

    dataset = RankedAnswersDataset(ranked_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            prompt = batch['prompt'][0]
            answers = batch['ranked_answers'][0]
            optimizer.zero_grad()
            loss = multi_rank_dpo_loss(model, ref_model, tokenizer, prompt, answers, beta=BETA)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.6f}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"模型已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    train()