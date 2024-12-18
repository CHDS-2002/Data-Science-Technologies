###################################
#                                 #
#          scikit-learn           #
#                                 #
###################################

###################################
#           Regression            #
###################################

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Генерация синтетических данных
np.random.seed(42)
X = np.random.rand(100, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(size=X.shape[0], scale=0.3)

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Создание и обучение модели
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), activation='relu',
                             solver='adam', max_iter=500, random_state=42)
mlp_regressor.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = mlp_regressor.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

coefficients = mlp_regressor.coefs_
intercepts = mlp_regressor.intercepts_

report_data = {
    'Признак': ['X{}'.format(i+1) for i in range(len(coefficients) + len(intercepts))],
    'Коэффициент': list(intercepts) + list(coefficients),
    'Стандартная ошибка': [np.std(intercept) for intercept in intercepts] +
     [np.std(coeff) for coeff in coefficients],
    't-статистическая': [0] * len(intercepts) + [coeff/err for coeff, err in zip(coefficients, [np.std(coeff) for coeff in coefficients])],
    'p-значение': [0] * len(intercepts) + [0] * len(coefficients)
}

# СОздаём таблицу коэффициентов
report_df = pd.DataFrame(report_data)

print(report_df)
print(f'MSE: {mse:.3f}, R2: {r2:.3f}')

###################################
#           Regression            #
###################################

###################################
#                                 #
#          scikit-learn           #
#                                 #
###################################