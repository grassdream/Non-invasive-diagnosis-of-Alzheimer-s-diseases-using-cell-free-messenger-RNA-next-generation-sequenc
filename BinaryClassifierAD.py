# _*_ coding: utf-8 _*_
# @Time : 2023-04-21 16:29 
# @Author : YingHao Zhang(池塘春草梦)
# @Version：v0.1
# @File : BinaryClassifierAD.py
# @desc : 对AD和NCI病人实现二分类,RF,AUC=0.9,binbin

from useful_functions import *
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier as RF
import time


start_time = time.time()
# %%
# 首先读取数据
# 数据的文件夹，包含了所有要用的数据
filepath = "D:/Bingo/项目/data/各个分类数据结果/"
# 标志物、训练集和测试集
MoleculeTypes = ['lncrna', 'mirna', 'SNP', 'mrna','WEIbilireads']
# MoleculeType = 4
resig = 0
# 选择前top_n的基因
top_n = 300
for MoleculeType in range(len(MoleculeTypes)):
    try:
        MoleculeTypeName = MoleculeTypes[MoleculeType]
        sig_file = 'ttest_deg_238'+MoleculeTypes[MoleculeType]+'RF.csv'
        file_train = '238'+MoleculeTypes[MoleculeType]+'RF.csv'
        file_test = '100'+MoleculeTypes[MoleculeType]+'RF.csv'
        # 读取数据
        train = pd.read_csv(filepath + file_train)
        test = pd.read_csv(filepath + file_test)
        #%%
        X_train = train.iloc[:, 2:].T
        y_train = train.iloc[:, 1].T
        X_test = test.iloc[:, 2:].T
        y_test = test.iloc[:, 1].T

        if resig==1:
            # 这一部分注释的是直接从一个基因排名的文件开始的
            sig = pd.read_csv(filepath + sig_file)
            sig.columns = ['Gene']
            # 选择基因排名前top_n的基因
            sig_top = sig.loc[0:top_n - 1]
            sig_top = set(list(sig_top['Gene']))
            genes = set(list(X_test.index))
            sig_genes = list(sig_top & genes)
            X_train = X_train.loc[sig_genes]
            X_test = X_test.loc[sig_genes]
        else:
            pass

        X_train = X_train.T
        X_test = X_test.T
        # 标准化
        X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
        X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))

        # 分类器构建
        rf = RF(random_state=0,max_depth=5,n_estimators=230)
        print(MoleculeTypeName)
        rf.fit(X_train, y_train)
        clf_evaluate(rf, MoleculeTypeName, X_test, y_test, CM=0, ROC=1)
    except:
        pass

end_time = time.time()
run_time = end_time - start_time
print("Done (/▽＼)")
print("代码运行时间为：{}分{}秒".format(int(run_time / 60), round(run_time % 60)))
