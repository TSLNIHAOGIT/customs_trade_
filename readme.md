1.机器学习步骤


2.模型自动化调参及训练主要包括一下几个部分：
（1）特征选择模块
（2）参数调优模块:小数量grid_search 、大数据量bayes、随机搜索相结合
（3）模型集成模块：(每个模型调优之后进行)stacking、stacknet


对数据还不是很了解，数据具体形式是：
一单是包含所有excel表格中的所有字段还是以某种方式关联这些表格？
这些表格或者字段之间有没有比较重要的时间顺序？

###
1.特征工程
对于类别变量或者数值变量
（1）特征选择
用lightgbm或xgboost中的feature_importance 选取不同的特征重要性（如split gain cover）
每种特征重要性选取一定比例(如80%)的重要特征，最后取并集(或者交集)。
（2）特征构建
  用特征选择中排在前几（或者人工挑选的比较重要的,例如运输工具、数量、价格等）的特征计算各种统计特征（一级和二级）：如count、unique
  关联特征：例如两个原始特征做差，根据实际情况而定
2.模型建立
  对纯数值特征、类别特征、新增特征以及三者混合分别建立分类模型(其中类别特征可以取one_hot转为数值特征或者指定该列为类别变量)
3.参数调优
   对每个模型用gridsearchcv、randomsearchcv形式寻找最优参数，如以auc（或者自定义标准，可能召回率高一些更好）最大化为目标时，可以保留auc较大时的多组参数
4.模型集成
  （1）对参数调优时的多组参数，每组参数就是一个新模型，用这些模型进行stacking集成
  （2）对各种特征类型的stacking模型，做简单的加权作为最终的模型
   
  
特征工程这一块整体上还要再加强理解以及对模型的理解

1.特征选择的各种方式（如何分类的，优缺点广义上做到充分理解）
2.特征提取这一块更要加强：pca、lda等等
3.特征构建：具有物理意义的特征构建，统计特征构建，等等