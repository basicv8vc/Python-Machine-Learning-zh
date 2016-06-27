# 理解sklearn中estimator的API

上一节我们调用了sklearn中Imputer类来处理缺失值。Imputer属于sklearn中所谓的transformer类，专门用于数据转换。此类estimator的两个必不可少的方法是fit和transform。fit方法用于从训练集中学习模型参数，transform用学习到的参数转换数据。

任何要进行转换的数据的特征维度必须和fit时的数据特征维度相同。

下图演示了fit和transform的过程：


![](https://ooo.0o0.ooo/2016/06/22/576a5b856e73c.png)




我们在第三章用到的各类分类器属于sklearn中的estimator，它的API和transformer非常像。Estimator还有一个predict方法，大部分也含有transform方法。同样Estimator含有fit方法来学习模型参数。只不过不同的是，在监督学习时，我们还像fit方法提供每个样本的类别信息。

![](https://ooo.0o0.ooo/2016/06/22/576a5cc2bd8b0.png)( 图中应该是est.fit(X_train,y_train))










