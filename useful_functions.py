# _*_ coding: utf-8 _*_
# @Time : 2023-04-02 11:20 
# @Author : YingHao Zhang(池塘春草梦)
# @Version：v0.1
# @File : useful_functions.py
# @desc : 一些有用的函数，减少代码复用
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import VotingClassifier


def clf_evaluate(clf, label, X_test, y_test, CM=1, ROC=1):
    """
    This function is used to evaluate the classifier.
    The evaluation contains: accuracy, f1_score, precision_score, recall, auc and confusion matrix.
    This function can plot the confusion matrix and ROC curve.

    Parameters:
      clf - sklearn.ensemble
      label - the classfier's name
      X_train - the training data
      y_train - the training labels
      X_test - the testing data
      y_test - the testing labels
      CM - confusion matrix, When CM = 1, we plot the confusion matrix
      ROC - ROC curve, When ROC = 1, we plot the ROC curve

    Returns:
        accuracy, precision_score, f1_score, recall, auc in dataframe format

    """
    # clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    y_predict_probe = clf.predict_proba(X_test)
    y_predict_probe = y_predict_probe[:, 1]
    from sklearn import metrics
    scores = []
    scores.append((round(metrics.accuracy_score(y_test, y_predict), 4),
                   round(metrics.precision_score(y_test, y_predict), 4),
                   round(metrics.f1_score(y_test, y_predict), 4),
                   round(metrics.recall_score(y_test, y_predict), 4),
                   round(metrics.matthews_corrcoef(y_test, y_predict), 4),
                   round(metrics.roc_auc_score(y_test, y_predict_probe), 4),
                   metrics.confusion_matrix(y_test, y_predict)
                   ))
    # scores = pd.DataFrame(scores,
    #                              columns=['准确率', '精确率', 'F1分数', '召回率', 'AUC'])
    scores = pd.DataFrame(scores,
                          columns=['Accuracy', 'Precision', 'F1', 'Recall','MCC' , 'AUC', 'Confusion Matrix'])
    print(label + '\'s evaluation:')
    from IPython.display import display
    display(scores)

    # print(scores_result)
    if (CM == 1):
        cm = metrics.confusion_matrix(y_test, y_predict, labels=clf.classes_)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=clf.classes_)
        disp.plot()
        plt.title(label + ' Confusion Matrix')
        plt.show()
    # P = clf.predict_proba(X_test)[:, 1]
    # fpr, tpr, threshold = metrics.roc_curve(y_test, P)
    if (ROC == 1):

        fpr, tpr, threshold = metrics.roc_curve(y_test, y_predict_probe)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(6, 4), dpi=250)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.stackplot(fpr, tpr, color='steelblue', alpha=0.4, edgecolor='black')
        # plt.plot(fpr, tpr, color='black', lw=1)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.text(0.5, 0.3, 'ROC curve (area = %0.4f)' % roc_auc, fontsize=10)
        plt.xlabel('False positive rate')
        plt.ylabel('Talse positive rate')
        # plt.savefig('D:/test.jpg')
        plt.title(label + ' ROC curve')
        plt.show()
    return scores, y_predict


def SoftVote(X_train, y_train, X_test, y_test, clfs, labels, soft=True):
    '''

    :description: 给定训练集、测试集以及sklearn分类器，实现软投票，并返回评估指标
    :param X_train: 训练集X
    :param y_train: 训练集y
    :param X_test: 测试集或验证集X
    :param y_test: 测试集或验证集y
    :param clfs: 分类器模型（sklearn）
    :param labels: 分类器标签
    :return: 每个基分类器以及软投票的ROC curve, accuracy, precision_score, f1_score, recall, auc in dataframe format
    '''
    X_validation = X_test
    y_validation = y_test
    weights = []
    estimators = []
    for clf, label in zip(clfs, labels):
        estimators.append((label, clf))
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_validation)
        y_predict_probe = clf.predict_proba(X_validation)
        y_predict_probe = y_predict_probe[:, 1]
        locals()['evaluation_' + label] = clf_evaluate(clf, label, X_validation, y_validation, 0, 1)
        weight = metrics.roc_auc_score(y_validation, y_predict_probe)
        weights.append(weight)
        # estimators.append((label, clf))
    if soft:
        vote2 = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
        clf = vote2
        clf.fit(X_train, y_train)
        label = "SoftVote"
        locals()['evaluation_' + label] = clf_evaluate(clf, label, X_validation, y_validation)
    else:
        pass

def fit_evaluate_sm(X_train, y_train, X_test, y_test, clf, label):
    '''

    :description: 给定训练集、测试集以及sklearn分类器，实现软投票，并返回评估指标
    :param X_train: 训练集X
    :param y_train: 训练集y
    :param X_test: 测试集或验证集X
    :param y_test: 测试集或验证集y
    :param clfs: 分类器模型（sklearn）
    :param labels: 分类器标签
    :return: 每个基分类器以及软投票的ROC curve, accuracy, precision_score, f1_score, recall, auc in dataframe format
    '''
    X_validation = X_test
    y_validation = y_test
    locals()['evaluation_' + label] = clf_evaluate(clf, label, X_validation, y_validation, 0, 1)

