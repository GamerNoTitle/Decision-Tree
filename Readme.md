## Decision-Tree
这是一个正在学习各种算法的Noob的笔记，这一Part是关于决策树（Decision-Tree）的。
首先，我们调用的数据还是使用鸢尾花的数据，就是文件里面的Iris_Data.csv
当然，需要pandas来读取这个文件。
```python
import pandas as pd
data=pd.read_csv('.\Iris_Data.csv')
```
这样就可以把文件的内容转入到一个data变量。
前面各种什么读取表格之类的操作跟昨天的[KNN算法](https://github.com/GamerNoTitle/KNN-Calculation)一样，这里不再赘述。
读入数据以后，就要调用DSTree的算法了。
使用下面这个代码，我们可以得到信息熵，把信息熵的值代入到DSTree中。
```python
DSTree = tree.DecisionTreeClassifier(criterion='entropy')
```
与KNN一样，机器都需要样本进行训练，我们可以使用
```python
DSTree.fit(feature_train,label_train)
```
来调用前面打乱的数据样本对其进行训练，让其得到它自己的算法。
最后再使用predict函数，让它进行预测，使用numpy的mean计算出预测的准确值
```python
DSTree.fit(feature_train,label_train)
acc=np.mean(DSTree.predict(feature_test)==label_test)
print(acc)
```
最后输出acc的值，就是这个算法的准确率了。