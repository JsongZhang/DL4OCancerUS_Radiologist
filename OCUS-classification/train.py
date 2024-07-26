import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import train_dataset, val_dataset, test_dataset
import os
import random
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(1)  # Default is no weighting
        else:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert inputs to probabilities
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        at = self.alpha.gather(0, targets.data)
        F_loss = at * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


Pretrained = False
Pretrained_path = './CDOA_r50_300.pth.tar'

model_name = "resnet50"
def worker_init_fn(worker_id):
    random.seed(1234 + worker_id) #seed_num + work_id
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, dynamic_ncols=True, ascii=True):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return avg_loss, accuracy, precision, recall, f1

def train_model(model, criterion, optimizer, train_data, val_data, num_epochs=25):
    best_acc = 0
    scaler = GradScaler()  # 初始化 GradScaler

    Train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    Val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(Train_dataloader, dynamic_ncols=True, ascii=True):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 全精度运算
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
            # 使用 autocast 来运行前向传播

            # 半精度运算
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 使用 scaler 来缩放 loss，使得反向传播可以在不溢出的情况下使用半精度
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_data)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, Val_dataloader, criterion)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}')

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            if best_acc < 85.00:
                print('saving best model at epoch {}'.format(epoch+1))
                torch.save(model.state_dict(), 'tt_newest_best_model_{}_acc.pth'.format(model_name))
            elif best_acc > 85.00:
                break
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    print("add in save checkpoint, best acc is ", best_acc)


# Initialize model
model = timm.create_model(model_name, pretrained=True, num_classes=3)
if Pretrained and os.path.isfile(Pretrained_path):
    print(f"=> loading checkpoint '{Pretrained}'")
    checkpoint = torch.load(Pretrained_path, map_location="cpu")

    state_dict = checkpoint.get('state_dict', checkpoint)  # 支持直接的state_dict或包装过的checkpoint
    new_state_dict = {}

    # 例如，假设我们不想加载最后的线性层权重
    linear_keyword = 'fc'  # 这需要根据你模型的具体情况来设定

    for k in list(state_dict.keys()):
        if k.startswith('module.base_encoder') and not k.startswith(f'module.base_encoder.{linear_keyword}'):
            new_key = k.replace('module.base_encoder.', '')  # 删除前缀以匹配模型中的命名
            new_state_dict[new_key] = state_dict[k]

    # 加载处理过的权重到模型的encoder部分
    # net.load_state_dict(new_state_dict, strict=False)
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    print('Loaded pretrained(encoder) ----->', Pretrained_path)
else:
    print('=> no pretrained for net')
model.to(device)
print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
Alpha = [0.3, 0.6, 0.1]

# criterion = FocalLoss(Alpha, gamma=2, reduction='mean')
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)


train_model(model, criterion, optimizer, test_dataset, test_dataset, num_epochs=200)
