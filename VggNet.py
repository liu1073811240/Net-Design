import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
import torch.utils.data as data
import PIL.Image as pimg
import matplotlib.pyplot as plt

transf_data = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, ], std=[0.5, ]),
     ]
)

train_data = datasets.CIFAR10("./cifar", train=True, transform=transf_data, download=True)
test_data = datasets.CIFAR10("./cifar", train=False, transform=transf_data, download=False)

train_loader = data.DataLoader(train_data, 512, shuffle=True)
test_loader = data.DataLoader(test_data, 128, shuffle=True)

# 第一种情况
# net = models.resnet50(pretrained=True)
# fc_features = net.fc.in_features
# net.out = nn.Linear(fc_features, 10)
# print(net)

# 第二种情况
net = models.vgg19_bn(pretrained=True)

# 如果加载了预训练模型，就遍历所有的预训练权重，不使用梯度更新参数
for param in net.parameters():
    # 梯度默认为True, 改成False
    param.requires_grad = False

# 重新定义某一层的网络形状，如果是加载了预训练参数，被重新定义了形状的那层参数需要重新学习。
# 更改VGG的指定层（最后一层）的输出形状，只更新最后一层的参数
# net.classifier._modules['6'] = nn.Linear(4096, 10)

# 增加VGG的层数，在原模型最后一层的基础上再增加一层，只更新增加一层的参数。
# net.classifier._modules['7'] = nn.Sequential(
#     nn.ReLU(inplace=True),
#     nn.Dropout(0.5),
#     nn.Linear(1000, 10)
# )
# num_fc_ftr = net.classifier._modules['6'].in_features
# net.classifier = nn.Linear(num_fc_ftr, 10)

# net.features._modules['0'] = nn.Conv2d(1, 64, 3, 1, 1)
net.avgpool = nn.AdaptiveAvgPool2d(7)
net.classifier._modules['7'] = nn.ReLU(inplace=True)
net.classifier._modules['8'] = nn.Dropout(0.5)
net.classifier._modules['9'] = nn.Linear(1000, 10)
print(net)
exit()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

net = net.to(device)
# net.load_state_dict(torch.load("./cifar_params.pth"))  # 只恢复保存的网络参数
# net = torch.load("./cifar_net.pth")  # 恢复保存的整个网络的参数
# loss_function = nn.MSELoss()
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters())
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-5, momentum=0)
# 用Adam训练两轮以后，再用SGD优化器训练，将两者参数进行微调，温和的进行训练，步长给1e-5,不加动量

net.train()
for epoch in range(2):
    for i, (x, y) in enumerate(train_loader):
        '''
        print(x.shape)  # torch.Size([100, 3, 32, 32])
        xs = x[0].data.numpy()
        print(xs.shape)  # (3, 32, 32)
        xs = xs.transpose(1, 2, 0)
        print(xs.shape)  # (32, 32, 3)
        xs = (xs*0.5 + 0.5) * 255
        img = pimg.fromarray(np.uint8(xs))
        plt.imshow(img)
        plt.pause(0)
        '''
        x = x.to(device)
        y = y.to(device)
        # y = torch.zeros(y.cpu().size(0), 10).scatter_(1, y.cpu().reshape(-1, 1), 1)

        output = net(x)
        # print(y.shape)  # torch.Size([100, 10])
        # print(output.shape)  # torch.Size([100, 10])

        loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch:{}, loss:{:.3f}".format(epoch, loss.item()))

    torch.save(net.state_dict(), "./cifar_param.pth")  # 只保存网络参数
    # torch.save(net, "./cifar_net.pth")  # 保存整个网络模型和参数

net.eval()
eval_loss = 0
eval_acc = 0
for i, (x, y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    out = net(x)
    # y = torch.zeros(y.cpu().size(0), 10).scatter(1, y.cpu().reshape(-1, 1), 1).cuda()
    loss = loss_function(out, y)
    print("Test_Loss:{:.3f}".format(loss.item()))

    eval_loss += loss.item()*y.size(0)

    arg_max = torch.argmax(out, 1)
    eval_acc += (arg_max == y).sum().item()

mean_loss = eval_loss / len(test_data)
mean_acc = eval_acc / len(test_data)
print("Loss:{:.3f}, Acc:{:.3f}".format(mean_loss, mean_acc))
