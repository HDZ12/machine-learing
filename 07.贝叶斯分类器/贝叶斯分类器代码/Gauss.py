from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 1.加载数据
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)

# 2.定义分类器
clf = GaussianNB()

# 3.模型训练
clf.fit(X_train, y_train)

# print(clf.class_count_)
# print(clf.classes_)
# print(clf.sigma_) 返回每个类别下每个特征分量的均值
# print(clf.theta_) 返回每个类别下每个特征分量的方差
print(X.shape)
print(clf.theta_.shape)

# 4.模型结果
print("训练集分数：", clf.score(X_train, y_train))
print("测试集分数：", clf.score(X_test, y_test))

# 如果我们的数据集较为大时，一次性不能够全部读入内存，此时就可以用partial_fit方法进行分批进行训练
clf.partial_fit(X_train, y_train)

print("训练集分数：", clf.score(X_train, y_train))
print("测试集分数：", clf.score(X_test, y_test))
