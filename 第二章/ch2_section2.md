# 使用Python实现感知机算法


在前一节，我们学习了Rosenblatt的感知机如果工作；这一节我们用Python对其进行实现，并且应用于Iris数据集。关于代码的实现，我们使用面向对象的编程思想，定义一个感知机接口作为Python类，类中的方法主要有初始化方法，fit方法和predict方法。






```
import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta:float
        Learning rate (between 0.0 and 1.0)
    n_iter:int
        Passes over the training dataset.

    Attributes
    -------------
    w_: 1d-array
        Weights after fitting.
    errors_: list
        Numebr of misclassifications in every epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ------------
        X: {array-like}, shape=[n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_featuers is the number of features.
        y: array-like, shape=[n_smaples]
            Target values.

        Returns
        ----------
        self: object
        """

        self.w_ = np.zeros(1 + X.shape[1]) # Add w_0
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1) #analoge ? : in C++



```

有了以上的代码实现，我们可以初始化一个新的Perceptron对象，并且对学习率eta和迭代次数n\_iter赋值，fit方法先对权重参数初始化，然后对训练集中每一个样本循环，根据感知机算法对权重进行更新。类别通过predict方法进行预测。除此之外，self.errors\_ 还记录了每一轮中误分类的样本数，有助于接下来我们分析感知机的训练过程。
