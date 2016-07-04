# 通过嵌套交叉验证选择算法

结合k折交叉验证和网格搜索是调参的好手段。可是如果我们想从茫茫算法中选择最合适的算法，用什么方法呢？这就是本节要介绍的嵌套交叉验证(nested cross validation)。 Varma和Simon在论文*Bias in Error Estimation When Using Cross-validation for Model Selection*中指出使用嵌套交叉验证得到的测试集误差几乎就是真实误差。


嵌套交叉验证外层有一个k折交叉验证将数据分为训练集和测试集。还有一个内部交叉验证用于选择模型算法。下图演示了一个5折外层交叉沿则和2折内部交叉验证组成的嵌套交叉验证，也被称为5*2交叉验证：


![](https://ooo.0o0.ooo/2016/06/28/57727696128be.png)



sklearn中可以如下使用嵌套交叉验证：



![](https://ooo.0o0.ooo/2016/06/28/57727759c2d45.png)

我们使用嵌套交叉验证比较SVm和决策树分类器：

![](https://ooo.0o0.ooo/2016/06/28/577277f19857f.png)
