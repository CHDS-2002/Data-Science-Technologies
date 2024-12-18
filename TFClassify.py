###################################
#                                 #
#           Tensorflow            #
#                                 #
###################################

###################################
#         Classification          #
###################################

import tensorflow as tf
from keras import layers, models
from keras.datasets import cifar10

# Определение архитектуры модели
def create_model(input_shape):
  # Входной слой
  inputs = layers.Input(shape=input_shape)

  # Свёрточный блок 1
  x = layers.Conv2D(filters=16, kernel_size=(3, 3),
                    padding='same', activation='relu')(inputs)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)

  # Свёрточный блок 2
  x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                   activation='relu')(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)

  # Полносвязный слой
  x = layers.Flatten()(x)
  x = layers.Dense(units=64, activation="relu")(x)

  # Выходной слой
  outputs = layers.Dense(units=1)(x)

  # Сборка модели
  model = models.Model(inputs=inputs, outputs=outputs)

  return model

# Форма изображения
input_shape = (32, 32, 3)

# Создаём модель
model = create_model(input_shape)

# Компилируем модель
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(), # MSE для регрессии
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# Данные для обучения и валидации модели
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Обучаем модель
history = model.fit(train_images, train_labels,
                    validation_data=(test_images, test_labels),
                    epochs=10, batch_size=32)

###################################
#         Classification          #
###################################

###################################
#                                 #
#           Tensorflow            #
#                                 #
###################################