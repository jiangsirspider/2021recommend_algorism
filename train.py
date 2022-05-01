from convert_to_num2 import ConvertToNum
import torch.nn as nn
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
# optimizer = optim.Adam(model.parameters())

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
def train(epoch):
    model.train()
    if os.path.exists("./models/comment_net.pkl"):
        print(True)
        model.load_state_dict(torch.load("./models/comment_net.pkl"))
        optimizer.load_state_dict(torch.load("./models/comment_optimizer.pkl"))
    dataset = NumDataset(train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, (data, label) in bar:
        data = data.to(device)
        # label = label.squeeze(-1).to(device)
        label = label.to(device)
        optimizer.zero_grad()
        out = model.forward(data)
        # print(out.size())
        # print(label.size())
        # loss = F.nll_loss(out, label)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        bar.set_description("epcoh:{}  idx:{}   loss:{:.6f}".format(epoch, idx, loss.item()))
        if idx % 100 == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, idx * len(data), len(train_dataloader.dataset),
            #            100. * idx / len(train_dataloader), loss.item()))
            torch.save(model.state_dict(), "./models/comment_net.pkl")
            torch.save(optimizer.state_dict(), './models/comment_optimizer.pkl')

def test():
    test_loss = 0
    correct = 0
    model.eval()
    model.load_state_dict(torch.load("./models/comment_net.pkl"))
    my_dataset = NumDataset(train=False)
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            # target = target.squeeze(-1).to(device)
            target = target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()
            test_loss += criterion(output, target).item()
            # print(output.cpu().detach().numpy()>0.5)
            pred = output.cpu().detach().numpy()>0.5 # 获取最大值的位置,[batch_size,1]
            pred = torch.tensor(pred).to(device)
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(dataloader)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))

if __name__ == '__main__':
    for i in range(10):
        train(i)
        test()