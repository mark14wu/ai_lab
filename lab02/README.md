首先进入虚拟环境：

`source venv/bin/activate`

然后使用 pip 安装所需包：

`pip install -r requirements.txt`

然后分别运行：

`python svm.py`

`python MLP.py`

`python NaiveBayes.py`

`python decisiontree.py`

来实现 4 个模型

数据一旦随机划分好就不会更改划分，若想要重新划分数据：

可以输入`make clean`来删除缓存在本地的数据