from sklearn import neural_network
from sklearn.datasets import load_iris
import numpy as np
import sys
import warnings
from sklearn.metrics import classification_report,confusion_matrix

warnings.filterwarnings('ignore')
iris = load_iris()
mlp = neural_network.MLPClassifier(hidden_layer_sizes=(10, 20),
                                   # （10，20)层数10+每层单元数20
                                   activation='relu',  # 激活函数
                                   solver='adam',
                                   alpha=0.00001,
                                   batch_size='auto',
                                   learning_rate='constant',
                                   learning_rate_init=0.001,
                                   power_t=0.5,
                                   max_iter=200,
                                   tol=1e-4
                                   )
mlp.fit(iris.data, iris.target)
print(mlp.predict([[1, 2, 3, 4]]))  # 预测结果
print('类别数：\n', mlp.n_outputs_)   # 输出类别数
print('所有类别： \n', mlp.classes_)  # 所有类别
print('损失函数损失值： \n', mlp.loss_)  # 损失函数的损失值
print('偏移量： \n', mlp.intercepts_)  # 偏移量

print('权重：\n', mlp.coefs_)
print('迭代论述：\n', mlp.n_iter_)
print('网络层次： \n', mlp.n_layers_)  # 只有一层隐藏层时 = 3
print('输出层的激活函数名称：\n', mlp.out_activation_)
# 用训练集预测数据（练习）
predictions = mlp.predict(iris.data)
print(confusion_matrix(iris.target, predictions))
print(classification_report(iris.target,predictions))
