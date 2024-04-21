import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import time

start_time = time.time()
# 宏定义
BATCH_SIZE = 512
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ERROR_RATE = 0


# normalization
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

#   实例化dataset
train_dataset = datasets.MNIST(root='../dataset', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='../dataset', train=False, download=True, transform=transform)

#   载入数据集
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# test
# fig = plt.figure()
# for i in range(12):
#     plt.subplot(3,4,i+1)
#     plt.tight_layout()
#     plt.imshow(train_dataset.train_data[i],cmap='gray',interpolation='none')
#     plt.title("Labels:{}".format(train_dataset.train_labels[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


# 构建CNN模型
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28
        self.conv1 = torch.nn.Conv2d(1, 10, 5)
        self.conv2 = torch.nn.Conv2d(10, 20, 3)
        self.fc1 = torch.nn.Linear(20*10*10, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    # 前馈
    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        # pooling
        out = F.max_pool2d(out, 2, 2)
        out = self.conv2(out)
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

# 创建model和optimizer
CNN_model = Net().to(DEVICE)
optimizer = torch.optim.Adam(CNN_model.parameters())

# train
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        # 梯度清零
        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)

        # 反向传播
        loss.backward()

        # 参数更新
        optimizer.step()

        # 结果打印
        if(batch_idx+1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    global ERROR_RATE
    ERROR_RATE += test_loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

for epoch in range(1, EPOCHS+1):
    train(CNN_model, DEVICE, train_loader, optimizer, epoch)
    test(CNN_model, DEVICE, test_loader)

end_time = time.time()

programming_time = end_time-start_time

print(f"this program takes {programming_time}s")
print(f"error rate is {ERROR_RATE/20}")
print(f"accuracy is{100-ERROR_RATE/20}")
