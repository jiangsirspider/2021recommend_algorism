import torch.nn as nn
import torch.nn.functional as F
import config
import torch
class TextModel(nn.Module):

    def __init__(self):
        super(TextModel, self).__init__()
        # self.embedding = nn.Embedding(num_embeddings=len(config.convert_to_num),embedding_dim=config.EMBEDDING_DIM,padding_idx=config.convert_to_num.PAD) #[N,200]
        # self.layer = nn.Sequential(
        #     nn.Linear(1, 1),
        #
        # )
        self.l1 = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """
        # print(input)
        # print(input.size())  # torch.Size([100, 2])
        # input_embeded = self.embedding(input)  # input embeded :[batch_size,max_len,200]

        # output = input_embeded.contiguous().view(input_embeded.size(0), -1)
        # print(output.size())
        out = self.l1(input)
        out = self.sm (out)
        # out = self.l2(out)
        # print(out.size())

        # out = self.lr(output)
        # out_relu = F.relu(out)
        # out = self.lr2(out_relu)
        # print(out.size())
        # print(F.log_softmax(out))

        return out






