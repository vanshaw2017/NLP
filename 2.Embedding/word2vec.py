import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.input_weight = nn.Embedding(self.vocab_size, self.embed_dim)  # 输入侧的参数embedding
        self.output_weight = nn.Embedding(self.vocab_size, self.embed_dim)  # 输出测得embedding
        init_range = 1.0 / self.embed_dim
        init.uniform_(self.input_weight.weight, -init_range, init_range)
        init.constant(self.output_weight.weight, 0)

    def forward(self, pos_input, pos_output, neg_v):
        """
        前向传播
        :param pos_input: 中心词的下标
        :param pos_output: Context词的下标
        :param neg_v:
        :return:
        """
        emb_input = self.input_weight(pos_input)
        emb_output = self.output_weight(pos_output)
        emb_neg = self.output_weight(neg_v)
        score = torch.sum(torch.matmul(emb_input, emb_output), dim=1)
        pos_loss = -F.logsigmoid(score)

        # 负采样样本
        neg_score = torch.bmm(emb_neg, emb_input.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, -10, 10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        neg_loss = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(pos_loss + neg_loss)
