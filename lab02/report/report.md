# AI LAB 02

吴昊 PB16001800

# 数据预处理

## 1. 频段分类

根据 EEG 特征的频段对160维度的特征进行分类，分为 $\theta$ 波，慢 $\alpha$ 波，$\alpha$波，$\beta$波和$\gamma$波。

## 2. 特征降维

经过调研，常用的比较有效的 EEG 降维方式主要有：PCA (Principal Component Analysis)，ICA(Independent Component Analysis) 和 LDA (Linear Discriminant Analysis)。其中，LDA 要求是有标签数据，其余两者不需要标签。根据文献[1]指出，相比 PCA 和 ICA 而言，LDA 对 EEG 的降维处理效果最好，因此我们使用 LDA 分别对上述 5 个频段进行降维处理，将每个频段的 32 维向量降到 3 维（LDA 降维的最大维数为分类数 - 1，而 valence arousal label 共有 4 个分类）。

## 3. K 折划分 (k = 5)

完成降维之后，对数据进行随机打乱（shuffle），并进行 5 折划分，便于在之后的测试中进行交叉检验。

交叉检验的方式为：轮流选取这 5 份数据中的一份，每份有且仅有一次作为测试数据的机会，其余作为训练数据，计算出准确率。

 对这 5 个准确率取平均，作为最终准确率。

注意：在划分时，我们保证了每一个 id 在各个数据集中出现的次数是相同的。

## 4. 存储

将划分好的数据以对象形式存储到本地，以便于在调整模型，输入的数据不会有改变。

存储方式使用了 Python 的 内置库 Pickle，可以直接保存 / 读取 Python 的各类对象。

# 模型训练与评价

## 1. SVM

软边界 SVM 使用了 kbf 核函数，然后通过循环尝试找出0.01 ~ 50.00之间的最优 C 取值（具体代码为函数 svm_tuning），该取值为 0.81

最终在 DEAP 数据集上准确率为 0.44765625

在 MAHNOB-HCI 数据集上准确率为 0.493388062723277

运行方式：`python svm.py`

值得注意的是，训练数据中的标签样本共有 4 类，而 SVM 仅仅支持二元分类。

为了实现 K 类的 SVM，这里才用了 1 v 1 的实现方式，即：

任意两个类之间都建立一个 SVM，同时训练数据只选取标签为这两类的数据。

这样可以训练得到 $K(K-1)/2$ 个 SVM。

在测试集上验证时，输入数据会被发送到每一个 SVM 上，每个 SVM 计算其预测的分类结果，并根据自己的分类结果进行投票，得票最高的分类为最终分类结果。

这样实现比较简单，在工程上也被证明是可靠的实践（LibSVM是通过1v1实现的），但是存在的问题是，如果分类数量变多，计算量是指数级别增长的。

## 2.MLP

MLP 部分关键代码如下：

```python
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_size))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
```

解释：

Keras 的代码描述的是层与层之间的连接关系

第一层神经元个数为 input_size （即输入向量的维数），与第二层全联接，激活函数为 relu。

第二层神经元个数为 64，与第三层全联接，激活函数为relu，并且将0.5（50%）的数据丢弃（dropout）。

第三层神经元个数为 64，与第四层全联接，激活函数为relu，并且将0.5（50%）的数据丢弃（dropout）。

第四层神经元个数为输出数据的维度（分类数）。

使用随机梯度下降作为优化器，初始学习率为 0.1，衰减率为$1\times10^6$，动量为0.9，使用nesterov修正。

动量是对于之前计算出的梯度的一个积累值，使用动量可以加快模型的收敛速度，同时，在模型接近局部极值点，梯度方向发生突变的时候，动量会帮助抵消这个突变，使得模型训练能跨过该局部极值点。

nesterov 修正是指，先根据动量计算出下一步的梯度，然后以下一步的梯度进行梯度下降。这样计算梯度可以在模型接近收敛时候减少因为动量积累导致的在收敛点附近来回震荡的时间。

----

在训练模型时，为确定训练次数（epoch），使用了回调函数（callback）之提前中止（earlystop），它保持对 loss 的监视，并且在 loss 函数不增反降的时候停止训练。此时训练的次数为最优训练次数。

最终训练结果为：

| 数据集 | Loss | 准确率 |
| ------ | ---- | ------ |
|DEAP valence arousal|1.5289123807634626|0.4156250017029898|
|MAHNOB valence arousal|1.9570359230041503|0.4596707046031952|
|DEAP subject id|0.8102078029087612|0.7849330357142856|
|MAHNOB subject id|1.1388960599899292|0.7275553941726685|
|MAHNOB emotion|2.0676942586898805|0.19152380228042604|

## 3. Naïve Bayes

对于 DEAP 和 MAHNOB-HCI 的 subject id 数据集，使用基于高斯分布的 NB(Naïve Bayes) 分类器，取 Laplace Smoothing 系数为 1.0，得到接近100%的准确率。

DEAP 准确率为 0.984375

MAHNOB-HCI 准确率为 0.9756816682667502

对于MAHNOB-HCI 情感分类 数据集，采用基于多项分布的 NB 分类器，Laplace Smoothing 系数为 1.0。

准确率为 0.2195811046454672，效果有限。

## 4. 感兴趣的分类器(Decision Tree)

简单调用 sci-kit learn 中已有的 decision tree 实现，计算准确率

DEAP: 0.3400669642857143
MAHNOB: 0.40338140115075405

# 参考文献

[1]. Subasi, Abdulhamit, and M. Ismail Gursoy. "EEG signal classification using PCA, ICA, LDA and support vector machines." *Expert systems with applications* 37.12 (2010): 8659-8666.