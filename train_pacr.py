import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import ndcg_score
import numpy as np

# ================== 配置 ==================
DATA_PATH = "pacr_training_data.json"
MODEL_NAME = "./bert-base-chinese" # 也可用其他中文BERT变体
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 数据集 ==================
class PACRDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample['question']
        pos_policy = sample['positive_policy']['text']
        neg_policies = [np['text'] for np in sample['negative_policies']]

        q_input = self.encode_text(question)
        p_pos = self.encode_text(pos_policy)
        # 将所有负例政策编码成 batch
        p_negs = [self.encode_text(ng) for ng in neg_policies]

        return {
            'q_input_ids': q_input['input_ids'].squeeze(0),
            'q_attention_mask': q_input['attention_mask'].squeeze(0),
            'p_pos_input_ids': p_pos['input_ids'].squeeze(0),
            'p_pos_attention_mask': p_pos['attention_mask'].squeeze(0),
            'p_neg_input_ids': torch.stack([p['input_ids'].squeeze(0) for p in p_negs]),
            'p_neg_attention_mask': torch.stack([p['attention_mask'].squeeze(0) for p in p_negs]),
            'question': question  # 仅用于日志
        }

# ================== 模型 ==================
class PACR(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.question_encoder = BertModel.from_pretrained(model_name)
        self.policy_encoder = BertModel.from_pretrained(model_name)
        # 简单融合层：拼接两个 [CLS] 向量后接 MLP 打分
        self.scorer = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def encode_question(self, input_ids, attention_mask):
        out = self.question_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output  # [batch, 768]

    def encode_policy(self, input_ids, attention_mask):
        out = self.policy_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output

    def forward(self, q_ids, q_mask, p_ids, p_mask):
        q_emb = self.encode_question(q_ids, q_mask)
        p_emb = self.encode_policy(p_ids, p_mask)
        combined = torch.cat([q_emb, p_emb], dim=-1)
        score = self.scorer(combined).squeeze(-1)  # [batch]
        return score

# ================== 损失函数 ==================
def info_nce_loss(pos_scores, neg_scores, temperature=0.07):
    """
    pos_scores: [batch_size]
    neg_scores: [batch_size, num_negatives]
    """
    batch_size = pos_scores.size(0)
    # 将正例得分与所有负例得分拼接，形成 [batch, 1 + num_neg]
    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [B, 1+N]
    labels = torch.zeros(batch_size, dtype=torch.long, device=pos_scores.device)  # 正例在第0列
    loss = F.cross_entropy(logits / temperature, labels)
    return loss

# ================== 训练函数 ==================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        q_ids = batch['q_input_ids'].to(device)
        q_mask = batch['q_attention_mask'].to(device)
        pos_ids = batch['p_pos_input_ids'].to(device)
        pos_mask = batch['p_pos_attention_mask'].to(device)
        neg_ids = batch['p_neg_input_ids'].to(device)   # [B, N, L]
        neg_mask = batch['p_neg_attention_mask'].to(device)

        # 正例得分
        pos_score = model(q_ids, q_mask, pos_ids, pos_mask)

        # 负例得分：需要将负例展平计算，再恢复形状
        B, N, L = neg_ids.shape
        neg_ids_flat = neg_ids.view(B*N, L)
        neg_mask_flat = neg_mask.view(B*N, L)
        # 复制问题嵌入 N 次
        q_ids_rep = q_ids.unsqueeze(1).repeat(1, N, 1).view(B*N, L)
        q_mask_rep = q_mask.unsqueeze(1).repeat(1, N, 1).view(B*N, L)

        neg_score = model(q_ids_rep, q_mask_rep, neg_ids_flat, neg_mask_flat).view(B, N)

        loss = info_nce_loss(pos_score, neg_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_ranking(model, dataloader, device):
    """计算 NDCG@5 指标（简单版：每个问题一个正例，4个负例计算NDCG）"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            q_ids = batch['q_input_ids'].to(device)
            q_mask = batch['q_attention_mask'].to(device)
            pos_ids = batch['p_pos_input_ids'].to(device)
            pos_mask = batch['p_pos_attention_mask'].to(device)
            neg_ids = batch['p_neg_input_ids'].to(device)
            neg_mask = batch['p_neg_attention_mask'].to(device)

            B, N, L = neg_ids.shape
            # 拼接正例和负例
            all_ids = torch.cat([pos_ids.unsqueeze(1), neg_ids], dim=1).view(B*(N+1), L)
            all_mask = torch.cat([pos_mask.unsqueeze(1), neg_mask], dim=1).view(B*(N+1), L)
            q_ids_rep = q_ids.unsqueeze(1).repeat(1, N+1, 1).view(B*(N+1), L)
            q_mask_rep = q_mask.unsqueeze(1).repeat(1, N+1, 1).view(B*(N+1), L)

            scores = model(q_ids_rep, q_mask_rep, all_ids, all_mask).view(B, N+1)
            # 标签：第一列正例相关性为1，其余为0
            labels = torch.zeros_like(scores)
            labels[:, 0] = 1
            all_preds.append(scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    # 计算 NDCG@5（此数据固定为5个候选）
    ndcg = ndcg_score(labels, preds, k=5)
    return ndcg

# ================== 主流程 ==================
def main():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    dataset = PACRDataset(DATA_PATH, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PACR(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_ndcg = 0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, dataloader, optimizer, DEVICE)
        ndcg = evaluate_ranking(model, dataloader, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | NDCG@5: {ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(model.state_dict(), "pacr_model.pt")
            print(f"  ✓ 模型保存（最佳NDCG）")

    print(f"\n训练完成！最佳 NDCG@5: {best_ndcg:.4f}")

if __name__ == "__main__":
    main()