# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:37:03 2021

@author: ujoshi
"""

#       Classification Metrics
def accuracy(y_true,y_pred):
    
    correct_counter=0 #counter for correct predictions
    for yt,yp in zip(y_true,y_pred):
        if yt==yp:
            correct_counter+=1
    return correct_counter/len(y_true)

from sklearn import metrics

l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]
print(metrics.accuracy_score(l1, l2))

print(accuracy(l1, l2),'\n')


def true_positive(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp

def true_negative(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn

def false_positive(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn

print(true_positive(l1, l2))
print(true_negative(l1, l2))
print(false_positive(l1, l2))
print(false_negative(l1, l2),'\n')


#Precision = TP / (TP + FP)
#Our model is correct % times when it’s trying to identify positive samples
def precision(y_true,y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    
    return (tp/(tp+fp))

print(metrics.precision_score(l1,l2))
print(precision(l1,l2),'\n')



#Recall = TP / (TP + FN)
#our model identified % of positive samples correctly
def recall(y_true,y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    
    return (tp/(tp+fn))

print(metrics.recall_score(l1,l2))
print(recall(l1,l2),'\n')


#F1 = 2TP / (2TP + FP + FN)
# OR       F1 = 2PR / (P + R)

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    
    return ((2*p*r)/(p+r))

print(metrics.f1_score(l1, l2))
print(f1(l1,l2),'\n')


#TPR = TP / (TP + FN)   (same as RECALL)
#FPR = FP / (FP + TN)
# SPECIFICITY = 1-FPR   (also known as TNR)

def tpr(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return (tp/(tp+fn))

def fpr(y_true, y_pred):
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    return (fp/(fp+tn))


##   Receiver Operating Characteristic (ROC)
## Area under Curve (AUC)

tpr_list = []
fpr_list = []

y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]

y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]

for thresh in thresholds: 
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
 
    temp_tpr = tpr(y_true, temp_pred)
 
    temp_fpr = fpr(y_true, temp_pred)
 
    tpr_list.append(temp_tpr)
    fpr_list.append(temp_fpr)

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
plt.fill_between(fpr_list, tpr_list, alpha=0.4)
plt.plot(fpr_list, tpr_list, lw=3)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)
plt.show()

