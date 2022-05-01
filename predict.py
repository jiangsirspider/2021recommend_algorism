# from convert_to_num2 import ConvertToNum

from torch import optim
from dataset import NumDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
from config import *
from model import TextModel

model = TextModel().to(device)
optimizer = optim.Adam(model.parameters())


def predict(_input):
    model.eval()
    model.load_state_dict(torch.load("./models/comment_net.pkl"))
    features = torch.FloatTensor(
        [int(i) for i in list(_input)])
    with torch.no_grad():
        data = features.to(device)
        output = model(data)
        # print(output.data)
        print(output.cpu().detach().item())
    return output.cpu().detach().item()
        # pred = output.data.max(-1, keepdim=False)[-1]  # 获取最大值的位置,[batch_size,1]
    # print(pred)

if __name__ == '__main__':
    #[bd_level: 1 有预算 -1：没有预算
    # ,star_level :商店人气：1-4,
    # ]0 -1
    predict(['1','2']) # 0.9120562076568604





