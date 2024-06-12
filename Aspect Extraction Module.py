import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from transformers import BertConfig


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label_to_int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_to_int = label_to_int

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        row = self.texts[index]
        text = row
        label = self.labels[index]  # 假设labels是列表
        label_int = self.label_to_int[label]  # 使用预先定义好的label_to_int字典进行转换
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_int, dtype=torch.long),  # 确保这里使用转换后的整数标签
        }

# 加载CSV数据
data = pd.read_csv("./0506.csv", encoding='gbk')  # 替换为你的CSV文件路径
texts = data["title"].tolist()
labels = data["category"].tolist()

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 初始化tokenizer
config_path = './config.json'
weights_path = './pytorch_model.bin'

config = BertConfig.from_json_file(config_path)
tokenizer = BertTokenizer.from_pretrained('./')
model = BertForSequenceClassification.from_pretrained(weights_path, config=config)

# 定义一个标签到整数的映射字典
label_to_int = {"财经": 0,"宠物": 1,"动漫": 2,"房产": 3,"教育": 4,"军事": 5,"科技": 6,"科普": 7,"历史": 8,"旅游": 9,"汽车": 10,"情感": 11,"生活": 12,"时事": 13,"数码": 14,"思想": 15,"文化": 16,"娱乐": 17,"育儿": 18,"运动": 19,"证券": 20}

# 创建数据集和DataLoader
max_len = 128  # 可根据实际情况调整
train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_len, label_to_int)
val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_len, label_to_int)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
num_labels = len(set(labels))
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)
model = BertForSequenceClassification.from_pretrained(
    './',  # 修改为你的本地模型目录
    num_labels=num_labels
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
import torch.optim as optim



# 定义训练和评估函数
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_epoch(model, data_loader, device):
    model.eval()
    fin_loss, fin_acc = 0, 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            acc = torch.sum(preds == labels).item() / len(labels)

            fin_loss += loss.item()
            fin_acc += acc
    return fin_loss / len(data_loader), fin_acc / len(data_loader)


# 设置训练参数
epochs = 30  # 训练轮数
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
from torch import save

best_val_loss = float('inf')  # 初始化最佳验证损失为正无穷大
best_model_path = "./best_model0601.pth"  # 最佳模型保存路径
# 开始训练
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print("Training...")
    train_epoch(model, train_loader, optimizer, scheduler, device)

    print("Validating...")
    val_loss, val_acc = eval_epoch(model, val_loader, device)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

    # 检查并保存最佳模型
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model...")
        best_val_loss = val_loss
        save(model.state_dict(), best_model_path)  # 仅保存模型参数


print("Training completed. Best model saved at:", best_model_path)
