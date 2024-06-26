# Results Comparing

## Introduction

在本项目中我选择CNN、逻辑回归、决策树这三种方法来实现基于MNIST数据集的手写数字识别，其中前两种方法基于pytorch实现，而决策树基于sklearn包实现。

## Result Table

|   | train time(s) | average error |
| --- | --- | --- |
|  CNN | 342.6289610862732 | 0.03925608344852924 |
|  Logistic Regression（Iteration times：3000） |  36.5039758682251 | 0.5400924980640411 |
|  Decision Tree | 54.461730003356934 | 0.13038095238095238 |

## Analysis

显然可以看到CNN模型具有最好的预测准确率，但同时CNN模型训练所需的时间是三种模型中最长的一个。逻辑回归和决策树模型效果表现较差，主要原因是因为这两种模型相较CNN来说，参数数量较小，难以捕捉图像数据的复杂关系；同时，由于图像数据的高维性，每一个像素都是一个特征，决策树和逻辑回归在训练中可能遇到维数灾难的问题，因此实际运行效果不如CNN模型。