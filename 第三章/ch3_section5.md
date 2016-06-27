# 支持向量机

另一个经常使用的机器学习算法是支持向量机(support vector machine, SVM)，SVM可以看做是感知机的扩展。在感知机算法中，我们最小化错误分类误差。在SVM中，我们的优化目标是最大化间隔(margin)。间隔定义为两个分隔超平面(决策界)的距离，那些最靠近超平面的训练样本也被称为支持向量(suppor vectors)。可以看下图：


![](https://ooo.0o0.ooo/2016/06/20/5767ae1337091.png)


## 最大化间隔

 最大化决策界的间隔，这么做的原因是间隔大的决策界趋向于含有更小的泛化误差，而间隔小的决策界更容易过拟合。为了更好地理解间隔最大化，我们先认识一下那些和决策界平行的正超平面和负超平面，他们可以表示为：
 
 
 ![](https://ooo.0o0.ooo/2016/06/20/5767b4c0a36f8.png)
 
 
 用(1)减去(2)，得到：
 
 ![](https://ooo.0o0.ooo/2016/06/20/5767b505b8b47.png)
 
 对上式进行归一化，
 
 ![](https://ooo.0o0.ooo/2016/06/20/5767b57928a93.png)
 
其中，![](https://ooo.0o0.ooo/2016/06/20/5767b59ada318.png)。

上式等号左边可以解释为正超平面和负超平面之间的距离，也就是所谓的间隔。

现在SVM的目标函数变成了最大化间隔$$\frac{2}{||w||}$$,限制条件是样本被正确分类，可以写成：

![](https://ooo.0o0.ooo/2016/06/20/5767b6164a8ac.png)


上面两个限制条件说的是所有负样本要落在负超平面那一侧，所有正样本要落在正超平面那侧。我们用更简洁的写法代替：


![](https://ooo.0o0.ooo/2016/06/20/5767b6675272a.png)

实际上，使用二次规划(quadratic programming)最小化$$\frac{1}{2}||w||^{2}$$很容易，但是二次规划显然超出了本书的内容，如果你对SVM感兴趣，推荐阅读Vladimir Vapnik写的 The Nature of Statistical Learning Theory, Springer Science&Business Media或Chris J.C. Burges写很棒的解释A Tutorial on Support Vector Machines for Pattern Recognition.














 
 
 
 
 
 

