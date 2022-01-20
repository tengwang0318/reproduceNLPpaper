import os
from transformers import AutoModel, AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch


class MyDataset(Dataset):
    def __init__(self, file_name, max_length, tokenizer):
        super(MyDataset, self).__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer

        with open(file_name) as f:
            lines = f.readlines()
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        label, sentence = line.split('\t', 1)

        sentence = sentence.strip()
        if sentence.endswith('.'):
            sentence = sentence[:-1]
        output = self.tokenizer.encode_plus(sentence,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            truncation=True,
                                            add_special_tokens=True)

        input_ids = output['input_ids']
        # print(input_ids)
        attention_mask = output['attention_mask']

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        label = torch.LongTensor([int(label)])

        # generate start index and end index
        start_indexs, end_indexs = [], []
        for i in range(1, self.max_length - 1):
            for j in range(i, self.max_length - 1):
                start_indexs.append(i)
                end_indexs.append(j)
        # generate span_mask used to distinguish which index is correct.
        span_masks = []
        # middle_index means the end of original sentence
        middle_index = input_ids.tolist().index(2)
        # print(middle_index)
        length = middle_index - 1
        for start_index, end_index in zip(start_indexs, end_indexs):
            if 1 <= start_index <= length and 1 <= end_index <= length and \
                    (start_index > middle_index or end_index < middle_index):
                span_masks.append(0)
            # start_index <= middle_index <= end_index
            else:
                span_masks.append(1e6)

        start_indexs = torch.LongTensor(start_indexs)
        end_indexs = torch.LongTensor(end_indexs)
        span_masks = torch.LongTensor(span_masks)

        return input_ids, attention_mask, label, start_indexs, end_indexs, span_masks


def test():
    dataset = MyDataset('dataset/train.txt', 15)
    datalaoder = DataLoader(dataset, batch_size=1)
    for input_ids, attention_mask, label, start_indexs, end_indexs, span_masks in datalaoder:
        print(input_ids)
        print(attention_mask)
        # print(label)
        print(start_indexs)
        print(end_indexs)
        print(span_masks)
        break



