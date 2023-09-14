# LinearRegression线性回归
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.metrics import mean_squared_error
diabetes = datasets.load_diabetes()
diabetes_x = diabetes.data[:, np.newaxis, 2]

diabetes_x_train = diabetes_x[:-20]
diabetes_x_test = diabetes_x[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# 核心代码
regr = linear_model.LinearRegression()
regr.fit(diabetes_x_train, diabetes_y_train)

print('Input Values')
print(diabetes_x_test)

# 核心代码
diabetes_y_pred = regr.predict(diabetes_x_test)
print('Predicted Output Values')
print(diabetes_y_pred)

# 模型评估
mse = mean_squared_error(diabetes_y_test,diabetes_y_pred)
print('均方误差（MSE）:', mse)

# 结果可视化
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

plt.scatter(diabetes_x_test, diabetes_y_test, color='black')
plt.plot(diabetes_x_test, diabetes_y_pred, color='red', linewidth=1)

plt.xlabel('体质指数',fontsize=12)
plt.ylabel('一年后患病的定量指标', fontsize=12)

plt.show()

