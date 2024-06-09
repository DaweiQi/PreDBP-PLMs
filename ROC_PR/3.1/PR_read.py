import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data1 = pd.read_csv('../../ROC_PR/3.1/PR_data/ProtBert_0.870_pr_data.csv')
fpr1 = data1.iloc[:, 0]
tpr1 = data1.iloc[:, 1]
roc_auc1 = "0.875"
data2 = pd.read_csv('../../ROC_PR/3.1/PR_data/ProtAlbert_0.865_pr_data.csv')
fpr2 = data2.iloc[:, 0]
tpr2 = data2.iloc[:, 1]
roc_auc2 = "0.872"
data3 = pd.read_csv('../../ROC_PR/3.1/PR_data/ProtXLNet_0.858_pr_data.csv')
fpr3 = data3.iloc[:, 0]
tpr3 = data3.iloc[:, 1]
roc_auc3 = "0.858"
data4 = pd.read_csv('../../ROC_PR/3.1/PR_data/0.912_pr_data.csv')
fpr4 = data4.iloc[:, 0]
tpr4 = data4.iloc[:, 1]
roc_auc4 = "0.912"
data5 = pd.read_csv('../../ROC_PR/3.1/PR_data/electra_0.638_pr_data.csv')
fpr5 = data5.iloc[:, 0]
tpr5 = data5.iloc[:, 1]
roc_auc5 = "0.649"
data6 = pd.read_csv('../../ROC_PR/3.1/PR_data/esm1b_0.864_pr_data.csv')
fpr6 = data6.iloc[:, 0]
tpr6 = data6.iloc[:, 1]
roc_auc6 = "0.864"
data7 = pd.read_csv('../../ROC_PR/3.1/PR_data/esm2_0.809_pr_data.csv')
fpr7 = data7.iloc[:, 0]
tpr7 = data7.iloc[:, 1]
roc_auc7 = "0.809"
data8 = pd.read_csv('../../ROC_PR/3.1/PR_data/tape_0.795_pr_data.csv')
fpr8 = data8.iloc[:, 0]
tpr8 = data8.iloc[:, 1]
roc_auc8 = "0.795"
data9 = pd.read_csv('../../ROC_PR/3.1/PR_data/unirep_0.842_pr_data.csv')
fpr9 = data9.iloc[:, 0]
tpr9 = data9.iloc[:, 1]
roc_auc9 = "0.842"

# plt.figure()

plt.figure(figsize=(8, 6))

# 使用不同的颜色和线型来绘制每条曲线
plt.plot(fpr1, tpr1, color='#E74C3A', linewidth=2, label='ProtBert (AUPRC = {})'.format(roc_auc1))
plt.plot(fpr2, tpr2, color='#4DBCD7', linewidth=2, label='ProtAlbert (AUPRC = {})'.format(roc_auc2))
plt.plot(fpr3, tpr3, color='#1CA189', linewidth=2, label='ProtXLNet (AUPRC = {})'.format(roc_auc3))
plt.plot(fpr4, tpr4, color='#3D578A', linewidth=2, label='ProtT5 (AUPRC = {})'.format(roc_auc4))
plt.plot(fpr5, tpr5, color='#AD85B1', linewidth=2, label='ProtElectra (AUPRC = {})'.format(roc_auc5))
plt.plot(fpr6, tpr6, color='#86C65E', linewidth=2, label='ESM-1b (AUPRC = {})'.format(roc_auc6))
plt.plot(fpr7, tpr7, color='#54809D', linewidth=2, label='ESM2 (AUPRC = {})'.format(roc_auc7))
plt.plot(fpr8, tpr8, color='#456990', linewidth=2, label='Tape (AUPRC = {})'.format(roc_auc8))
plt.plot(fpr9, tpr9, color='#D61C7E', linewidth=2, label='Unirep (AUPRC = {})'.format(roc_auc9))

# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall', fontdict={'weight': 'normal', 'size': 14})
plt.ylabel('Precision', fontdict={'weight': 'normal', 'size': 14})
plt.legend(loc="lower right", prop={'weight': 'normal', 'size': 10})
plt.grid(True)
plt.show()