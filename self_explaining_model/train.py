import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW
from preprocess_data import MyDataset
from model import ExplainableModel
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score


class Config:
    model_name = 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    training_batch_size = 8
    validation_batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    epochs = 5
    learning_rate = 2e-5
    max_length = 64
    lam = 1


def set_random_seed(seed=318):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed()

model = ExplainableModel(Config.model_name, 5).to(Config.device)

train_dataset = MyDataset('dataset/train.txt', Config.max_length, Config.tokenizer)
validate_dataset = MyDataset('dataset/dev.txt', Config.max_length, Config.tokenizer)
test_dataset = MyDataset('dataset/test.txt', Config.max_length, Config.tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=Config.training_batch_size, shuffle=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=Config.validation_batch_size)

optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
total_steps = len(train_dataloader) // Config.training_batch_size * Config.epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0.1 * total_steps,
                                            num_training_steps=total_steps)

for epoch in range(Config.epochs):
    model.train()
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    running_loss = 0
    data_size = 0

    start_indexs, end_indexs = [], []
    for i in range(1, Config.max_length - 1):
        for j in range(i, Config.max_length - 1):
            start_indexs.append(i)
            end_indexs.append(j)

    for step, batch in bar:
        input_ids, attention_mask, label, span_masks = batch
        input_ids = input_ids.to(Config.device)
        attention_mask = attention_mask.to(Config.device)
        label = label.to(Config.device)
        start_indexs = start_indexs.to(Config.device)
        end_indexs = end_indexs.to(Config.device)

        batch_size = input_ids.size(0)

        label = label.view(-1)
        y_hat, a_ij = model(input_ids, attention_mask, start_indexs, end_indexs, span_masks)

        loss_fn = nn.CrossEntropyLoss()
        ce_loss = loss_fn(y_hat, label)

        reg_loss = Config.lam * a_ij.pow(2).sum(dim=1).mean()

        loss = ce_loss - reg_loss

        predict_scores = F.softmax(y_hat, dim=-1)
        predict_labels = torch.argmax(predict_scores, dim=-1)

        acc = precision_score(predict_labels, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        data_size += batch_size

        running_loss += (loss.item() * batch_size)

        epoch_loss = running_loss / data_size

        bar.set_postfix(epoch=epoch, epoch_loss=epoch_loss, learning_rate=optimizer.param_groups[0]['lr'])

    model.eval()
    bar = tqdm(enumerate(validate_dataloader), total=len(validate_dataloader))
    running_loss = 0
    data_size = 0
    for step, batch in bar:
        input_ids, attention_mask, label, start_indexs, end_indexs, span_masks = batch
        input_ids = input_ids.to(Config.device)
        attention_mask = attention_mask.to(Config.device)
        label = label.to(Config.device)
        start_indexs = start_indexs.to(Config.device)
        end_indexs = end_indexs.to(Config.device)

        batch_size = input_ids.size(0)

        label = label.view(-1)
        y_hat, a_ij = model(input_ids, attention_mask, start_indexs, end_indexs, span_masks)

        loss_fn = nn.CrossEntropyLoss()
        ce_loss = loss_fn(y_hat, label)

        reg_loss = Config.lam * a_ij.pow(2).sum(dim=1).mean()

        loss = ce_loss - reg_loss

        predict_scores = F.softmax(y_hat, dim=-1)
        predict_labels = torch.argmax(predict_scores, dim=-1)

        acc = precision_score(predict_labels, label)

        data_size += batch_size

        running_loss += (loss.item() * batch_size)

        epoch_loss = running_loss / data_size

        bar.set_postfix(epoch=epoch, epoch_loss=epoch_loss, learning_rate=optimizer.param_groups[0]['lr'])
