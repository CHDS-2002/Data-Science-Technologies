###################################
#                                 #
#          scikit-learn           #
#                                 #
###################################

###################################
#         Classification          #
###################################

# Importing libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist

# Создание модели
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Сборка модели
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Загружаем данные MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Приведение данных к формату, необходимому для CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Нормализация данных
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Преобразование меток в категориальны формат
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Обучение модели
history = model.fit(X_train, y_train, batch_size=128,
                    epochs=10,
                    validation_data=(X_test, y_test))

# Прогнозирование на тестовом наборе
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# Отчёт о классификации
print(classification_report(y_test, y_pred))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанная метка')
plt.ylabel('Истинная метка')
plt.title('Матрица ошибок')
plt.show()

# Построение графиков истории обучения
plt.plot(history.history['loss'], label='Потери обучения')
plt.plot(history.history['val_loss'], label='Потери тестовые')
plt.legend()
plt.title('Кривая потерь')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.grid(True)
plt.show()

plt.plot(history.history['accuracy'], label='Точность обучения')
plt.plot(history.history['val_accuracy'], label='Точность тестовая')
plt.legend()
plt.title('Кривая точности')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.grid(True)
plt.show()

###################################
#         Classification          #
###################################

###################################
#                                 #
#          scikit-learn           #
#                                 #
###################################