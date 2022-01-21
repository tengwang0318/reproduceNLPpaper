import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class SICModel(nn.Module):
    def __init__(self, hidden_size):
        super(SICModel, self).__init__()
        self.hidden_size = hidden_size

        self.W_1 = nn.Linear(hidden_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, hidden_size)
        self.W_3 = nn.Linear(hidden_size, hidden_size)
        self.W_4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, start_indexs, end_indexs):
        W1_h = self.W_1(hidden_state)
        W2_h = self.W_2(hidden_state)
        W3_h = self.W_3(hidden_state)
        W4_h = self.W_4(hidden_state)

        W1_hi_emb = torch.index_select(W1_h, 1, start_indexs)
        W2_hj_emb = torch.index_select(W2_h, 1, end_indexs)
        W3_hi_start_emb = torch.index_select(W3_h, 1, start_indexs)
        W3_hi_end_emb = torch.index_select(W3_h, 1, end_indexs)
        W4_hj_start_emb = torch.index_select(W4_h, 1, start_indexs)
        W4_hj_end_emb = torch.index_select(W4_h, 1, end_indexs)

        span = W1_hi_emb + W2_hj_emb + (W3_hi_start_emb - W3_hi_end_emb) + torch.mul(W4_hj_start_emb, W4_hj_end_emb)
        hij = torch.tanh(span)
        return hij


class InterpretationModel(nn.Module):
    def __init__(self, hidden_size):
        super(InterpretationModel, self).__init__()
        self.h_t = nn.Linear(hidden_size, 1)

    def forward(self, h_ij, span_masks):
        o_ij = self.h_t(h_ij).squeeze(-1)
        # (bs, span_num)
        o_ij = o_ij - span_masks
        # (bs, span_num) to get the possibility of every pair of sub-words.
        a_ij = nn.functional.softmax(o_ij, dim=1)
        # h_ij size is (bs, span_num, hidden_size)
        H = (a_ij.unsqueeze(-1) * h_ij).sum(dim=1)

        return H, a_ij


class ExplainableModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(ExplainableModel, self).__init__()
        self.num_labels = num_labels
        self.bert_config = AutoConfig.from_pretrained(model_name)
        self.intermediate = AutoModel.from_pretrained(model_name)
        self.span_info_collect = SICModel(self.bert_config.hidden_size)
        self.interpretation = InterpretationModel(self.bert_config.hidden_size)

        self.output = nn.Linear(self.bert_config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask, start_indexs, end_indexs, span_masks):
        output = self.intermediate(input_ids, attention_mask)
        hidden_state = output['last_hidden_state']
        h_ij = self.span_info_collect(hidden_state, start_indexs, end_indexs)
        H, a_ij = self.interpretation(h_ij, span_masks)
        output = self.output(H)
        return output, a_ij
