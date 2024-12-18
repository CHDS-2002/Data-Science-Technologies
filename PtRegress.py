###################################
#                                 #
#             PyTorch             #
#                                 #
###################################

###################################
#           Regression            #
###################################

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Генерация синтетических данных
num_samples = 100
X1 = torch.randn(num_samples, 1)
X2 = torch.randn(num_samples, 1)
X = torch.cat((X1, X2, torch.ones(num_samples, 1)), dim=-1)
k1 = 2
k2 = 3
k3 = 0.3
b = 1
y = k1 * X1 + k2 * X2 + b + torch.randn_like(X1) * k3

# Визуализация данных
plt.scatter(X[:, 0].detach().numpy(), y.detach().numpy(), marker='o',
            label='Данные')
plt.title('Исходные данные')
plt.xlabel('X1')
plt.ylabel('y')
plt.legend()
plt.show()

# Создание модели
class MultiLinearRegression(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(MultiLinearRegression, self).__init__()
    self.linear = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    return self.linear(x)

# Параметры модели
input_dim = 3
output_dim = 1
model = MultiLinearRegression(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Тренировка модели
epochs = 100
correct = 0

for epoch in range(epochs):
  y_pred = model(X)
  loss = criterion(y_pred.squeeze(1), y)
  print(f'Эпоха {epoch}: Потеря: {loss.item()}')
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# Визуализация результата
pred = model(X)
plt.scatter(X[:, 0].detach().numpy(), pred.detach().numpy(), marker='.',
            label='Прогнозы')
plt.title('Фактические и прогнозируемые значения')
plt.xlabel('X1')
plt.ylabel('y')
plt.legend()
plt.show()

# Оценка модели
mse = nn.MSELoss()
accuracy = nn.Acc
print(f'MSE: {mse(pred.squeeze(1), y)}')

###################################
#           Regression            #
###################################

###################################
#                                 #
#             PyTorch             #
#                                 #
###################################