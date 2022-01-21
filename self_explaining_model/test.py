import torch
from sklearn.metrics import precision_score
from preprocess_data import MyDataset
from model import ExplainableModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TestConfig:
    model_name = 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    max_length = 64
    lam = 1


def set_random_seed(seed=318):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed()
test_dataset = MyDataset('dataset/test.txt', TestConfig.max_length, TestConfig.tokenizer)
model = ExplainableModel(TestConfig.model_name, 5).to(TestConfig.device)
model.eval()
model.load_state_dict(torch.load('model.bin'))

test_dataloader = DataLoader(test_dataset, batch_size=TestConfig.test_batch_size)

bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
with torch.no_grad():
    running_loss = 0
    data_size = 0

    start_indexs, end_indexs = [], []
    for i in range(1, TestConfig.max_length - 1):
        for j in range(i, TestConfig.max_length - 1):
            start_indexs.append(i)
            end_indexs.append(j)

    start_indexs = torch.tensor(start_indexs).to(TestConfig.device)
    end_indexs = torch.tensor(end_indexs).to(TestConfig.device)
    total_labels, total_predicted_labels = [], []

    for step, batch in bar:
        input_ids, attention_mask, label, span_masks = batch
        input_ids = input_ids.to(TestConfig.device)
        attention_mask = attention_mask.to(TestConfig.device)
        label = label.to(TestConfig.device)
        span_masks = span_masks.to(TestConfig.device)

        batch_size = input_ids.size(0)

        label = label.view(-1)
        y_hat, a_ij = model(input_ids, attention_mask, start_indexs, end_indexs, span_masks)

        loss_fn = nn.CrossEntropyLoss()
        ce_loss = loss_fn(y_hat, label)

        reg_loss = TestConfig.lam * a_ij.pow(2).sum(dim=1).mean()

        loss = ce_loss - reg_loss

        predict_scores = F.softmax(y_hat, dim=-1)
        predict_labels = torch.argmax(predict_scores, dim=-1)

        total_labels.extend(label.cpu())
        total_predicted_labels.extend(predict_labels.cpu())
        data_size += batch_size

        running_loss += (loss.item() * batch_size)

        epoch_loss = running_loss / data_size

    print("test epoch loss: {}".format(epoch_loss))
    acc = precision_score(total_labels, total_predicted_labels, average='micro')
    print('test accuracy: {}'.format(acc))
