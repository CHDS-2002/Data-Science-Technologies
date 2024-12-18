###################################
#                                 #
#           Tensorflow            #
#                                 #
###################################

###################################
#           Regression            #
###################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Генерация синтетических данных
num_samples = 200
X = tf.random.normal((num_samples, 1))
noise = tf.random.normal((num_samples, 1))
k = 3
b = 2
y = k * X + b + noise

# Визуализация данных
plt.scatter(X.numpy(), y.numpy(), marker='o', label='Данные')
plt.title('Исходные данные')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

# Компиляция модели
model.compile(optimizer='Adam', loss='mse')

# Обучение модели
history = model.fit(X, y, epochs=100, verbose=0)

# Вывод результатов обучения
plt.plot(history.history['loss'], label='Потеря')
plt.title('Динамика потерь во время обучения')
plt.xlabel('Эпоха')
plt.ylabel('Средняя ошибка на мини-батче')
plt.legend()
plt.show()

# Прогнозирование
predictions = model.predict(X)

# Визуализация прогнозируемых значений
plt.scatter(X, predictions, marker='.', label='Прогнозы')
plt.title('Фактические и прогнозируемые значения')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Оценка модели
mse = tf.keras.losses.MSE(y, predictions)
print(f'MSE: {mse}')

###################################
#           Regression            #
###################################

###################################
#                                 #
#           Tensorflow            #
#                                 #
###################################