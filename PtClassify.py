###################################
#                                 #
#             PyTorch             #
#                                 #
###################################

###################################
#         Classification          #
###################################

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Загрузка и преобразование данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                               transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                              transform=transform)

# Создание загрузчиков данных
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,
                                          shuffle=False)

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

model = CNN()
criterion = nn.NLLLoss() # Функция потерь для логарифмических вероятностей
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
  model.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if not batch_idx % 200:
      print('Эпоха обучения: {} [{}/{} ({:.0f}%)]\tПотери: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

def test():
  model.eval()
  test_loss = 0
  correct = 0

  with torch.no_grad():
    for data, target in test_loader:
      output = model(data)
      test_loss += criterion(output, target).item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  accuracy = 100. * correct / len(test_loader.dataset)
  print('\nТестовый сет: Средняя потеря: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset), accuracy))

for epoch in range(1, 11):
  train(epoch)
  test()
###################################
#         Classification          #
###################################

###################################
#                                 #
#             PyTorch             #
#                                 #
###################################