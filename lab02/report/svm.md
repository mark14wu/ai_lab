# SVM 对 valence arousal label 分类

吴昊 PB16001800

# 数据预处理

## 1. 频段分类

根据 EEG 特征的频段对160维度的特征进行分类，分为 $\theta$ 波，慢 $\alpha$ 波，$\alpha$波，$\beta$波和$\gamma$波

## 2. 特征降维

经过调研，常用的比较有效的 EEG 降维方式主要有：PCA (Principal Component Analysis)，ICA(Independent Component Analysis) 和 LDA (Linear Discriminant Analysis)。其中，LDA 要求是有标签数据，其余两者不需要标签。根据文献[1]指出，相比 PCA 和 ICA 而言，LDA 对 EEG 的降维处理效果最好，因此我们使用 LDA 分别对上述 5 个频段进行降维处理，将每个频段的 32 维向量降到 3 维（LDA 降维的最大维数为分类数 - 1，而 valence arousal label 共有 4 个分类）

## 3. K 折划分 (k = 5)

完成降维之后，对数据进行随机打乱（shuffle），并进行 5 折划分，便于在之后的测试中进行交叉检验

交叉检验的方式为：轮流选取这 5 份数据中的一份作为测试数据，其余作为训练数据，计算出准确率

 5 份数据计算出 5 个准确率，取平均，即为最终准确率

注意：在划分时，我们保证了每一个 id 在各个数据集中出现的次数是相同的

# 模型训练与评价



# 参考文献

[1]. Subasi, Abdulhamit, and M. Ismail Gursoy. "EEG signal classification using PCA, ICA, LDA and support vector machines." *Expert systems with applications* 37.12 (2010): 8659-8666.