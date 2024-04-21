import torch
from torchvision import datasets
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import time

# 记录开始时间
start_time = time.time()

# Load Dataset
train_dataset = datasets.MNIST(root="../dataset", train=True, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST(root="../dataset", train=False, transform=transforms.ToTensor())

# Initialize parameters
BATCH_SIZE = 100
n_iters = 3000
epochs = n_iters / (len(train_dataset) / BATCH_SIZE)
input_dim = 784
output_dim = 10
lr_rate = 0.01




train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model building
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        outputs = self.linear(x)
        return outputs

LR_model = LogisticRegression(input_dim, output_dim)

loss_criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(LR_model.parameters(), lr=lr_rate)

# Model trianing
iter = 0

ERROR_RATE = 0;

for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = LR_model(images)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        iter+=1
        if iter%500==0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28*28))
                outputs = LR_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                correct += (predicted == labels).sum()
            accuracy = 100 * correct/total
            ERROR_RATE += loss.item()
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))


end_time = time.time()

programming_time = end_time-start_time

print(f"error rate is {ERROR_RATE/6}")
print(f"accuracy is{100-ERROR_RATE/6}")

print(f"it takes {programming_time}")


