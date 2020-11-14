import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms,datasets,models
import torch.utils.data as data
import PIL.Image as pimg
import matplotlib.pyplot as plt
transf_data = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]
)
# nn.Sigmoid()
# nn.BCELoss()
#
# nn.BCEWithLogitsLoss()
# nn.CrossEntropyLoss()
# nn.MSELoss()
train_data = datasets.CIFAR10("../cifar",train=True,transform=transf_data,download=True)
test_data = datasets.CIFAR10("../cifar",train=False,transform=transf_data,download=False)
train_loader = data.DataLoader(train_data,100,shuffle=True)
test_loader = data.DataLoader(test_data,100,shuffle=True)

net = models.densenet121(pretrained=False)
#如果加载了预训练模型，就遍历所有的预训练权重，不使用梯度更新

'''
for param in net.parameters():
    #梯度默认为True，改成False，表示使用预训练权重
    param.requires_grad = False
'''
# print(net)
#更改最后一层（分类层）,只训练更改后的那层的参数
"第一种方式"
# net.classifier = nn.Linear(1024, 10)
"第二种方式"
num_cls_ftr = net.classifier.in_features
#增加网络层数
net.classifier = nn.Linear(num_cls_ftr, 128)
net.out = nn.Linear(128, 10)
print(net)

if torch.cuda.is_available():
    # net = net.cuda()
    device = torch.device("cuda")
else:
    # net = net
    device = torch.device("cpu")

net = net.to(device)
# net.load_state_dict(torch.load("./cifar_params.pth"))# 只恢复保存的网络参数
# net = torch.load("./cifar_net.pth")# 恢复保存的整个网络和参数
# loss_function = nn.MSELoss()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

net.train()
for epoch in range(2):
    for i,(x,y) in enumerate(train_loader):

        x = x.to(device)
        y = y.to(device)
        # y = torch.zeros(y.cpu().size(0), 10).scatter_(1, y.cpu().reshape(-1, 1), 1).to(device)
        output = net(x)
        # print(y.shape)
        # print(output.shape)

        loss = loss_function(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%10 == 0:

            print("Epoch:{},Loss:{:.3f}".format(epoch,loss.item()))
    torch.save(net.state_dict(),"./cifar_params.pth")#只保存网络参数
    torch.save(net,"./cifar_net.pth")#保存整个网络和参数

net.eval()
eval_loss = 0
eval_acc = 0
for i,(x,y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    out = net(x)
    # y = torch.zeros(y.cpu().size(0),10).scatter_(1,y.cpu().reshape(-1,1),1).cuda()
    loss = loss_function(out,y)
    print("Test_Loss:{:.3f}".format(loss.item()))
    eval_loss += loss.item()*y.size(0)

    arg_max= torch.argmax(out,1)
    eval_acc += (arg_max==y).sum().item()

mean_loss = eval_loss/len(test_data)
mean_acc= eval_acc/len(test_data)
print("Loss:{:.3f},Acc:{:.3f}".format(mean_loss,mean_acc))



