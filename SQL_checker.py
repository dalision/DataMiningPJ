#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# In[126]:


import sys
import pandas as pd
import math
import seaborn as sns
import collections
import urllib
from urllib import parse
malicious = pd.read_csv("data/data59399/malicious.txt",names=["payload"])
benign = pd.read_csv("data/data59399/benign.txt",names=["payload"])
# benign.apply(parse.urlencode)
benign["payload"] = benign["payload"].apply(lambda x: (parse.unquote(x)).lower())
malicious.head(10)
malicious = malicious.drop(index=[815,878])
malicious["payload"] = malicious["payload"].apply(lambda x: (parse.unquote(x)).lower())
malicious["label"] =1
malicious["len"] = malicious["payload"].apply(lambda x: len(x))
benign["len"] = benign["payload"].apply(lambda x: len(x))
benign["label"] =0
totol = pd.concat([malicious,benign],axis=0)
totol.head(1)


# In[127]:


malicious.iloc[:50000,]


# # 基于统计

# In[128]:


import seaborn as sns
import pandas
ax1 = sns.kdeplot(totol.loc[totol["label"]==1,"len"],shade=True,color='r',label="m")
ax1 = sns.kdeplot(totol.loc[totol["label"]==0,"len"],shade=True,color='b',label="Normal")


# In[129]:


def getshan(x):#熵
	tmp_dict = {}
	payload_len = len(x["payload"])
	for i in range(0,len(x["payload"])):
		if x["payload"][i] in tmp_dict.keys():
			tmp_dict[x["payload"][i]] = tmp_dict[x["payload"][i]] + 1
		else:
			tmp_dict[x["payload"][i]] = 1
	shannon = 0
	for i in tmp_dict.keys():
		p = float(tmp_dict[i]) / payload_len
		shannon = shannon - p * math.log(p,2)
	return shannon


# In[130]:


totol["shan"] = totol.apply(getshan,axis=1)


# In[131]:


ax1 = sns.kdeplot(totol.loc[totol["label"]==1,"shan"],shade=True,color='r',label="m")
ax1 = sns.kdeplot(totol.loc[totol["label"]==0,"shan"],shade=True,color='b',label="Normal")


# # 基于经验

# In[132]:


totol["white"] = totol["payload"].apply(lambda x:x.count(" ")/len(x))
def key_num(line):
    key_num = line.count('and ')+line.count('or% ')+line.count('xor% ')+line.count('sysobjects% ')+line.count('version ')+line.count('substr ')+line.count('len ')+line.count('substring ')+line.count('exists ')
    key_num=key_num+line.count('mid% ')+line.count('asc% ')+line.count('inner join% ')+line.count('xp_cmdshell ')+line.count('version ')+line.count('exec ')+line.count('having ')+line.count('unnion ')+line.count('order ')+line.count('information schema')
    key_num=key_num+line.count('load_file ')+line.count('load data infile ')+line.count('into outfile ')+line.count('into dumpfile ')+line.count('waitfor delay')
    return key_num
totol["key"] = totol["payload"].apply(key_num)
######white 和 key 都挺不错


# In[133]:


def special(line):
    spe = "( ) * < > ; @ - .".split(" ") 
    spe.append("#")
    spe.append("'") 
    s_count = 0
    for symbol in spe:
        s_count += line.count(symbol)
    return s_count

#闭合
def dan(line):
    count = 0
    # if line.count('(')-line.count(')')!=0:
    #     count +=1
    # if line.count('<')-line.count('>')!=0:
    #     count +=1
    if line.count("'")%2==1:
        count +=1
    if line.count("\"")%2==1:
        count +=1
    return count==0

def fj(line):
    return ";" in line #or ")" in line
totol["fj"] = totol["payload"].apply(fj)
totol["sb"] = totol["payload"].apply(special)
totol["dan"] = totol["payload"].apply(dan)


# In[134]:


ax1 = sns.kdeplot(totol.loc[totol["label"]==1,"key"],shade=True,color='r',label="m")
ax1 = sns.kdeplot(totol.loc[totol["label"]==0,"key"],shade=True,color='b',label="Normal")


# In[135]:


import seaborn as sns
import pandas
sns.countplot(x='key', hue='label',data=totol)


# In[136]:


totol.head()


# In[ ]:





# 

# In[148]:


features = totol.iloc[:,1:]
lable = totol["label"]
features = totol.drop(["label","payload"],axis=1)
features.head(5)


# In[110]:





# In[149]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sklearn.ensemble as ek
from sklearn import  tree, linear_model
from sklearn.model_selection import cross_validate as cross_validation
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# 新增
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[161]:


X=StandardScaler().fit_transform(features)#实现均方差值
X_train, X_test, y_train, y_test=train_test_split(X,lable)#划分数据集
model = { "DecisionTree":tree.DecisionTreeClassifier(max_depth=10),
         "RandomForest":ek.RandomForestClassifier(),
         "Adaboost":ek.AdaBoostClassifier(n_estimators=50),
         "GradientBoosting":ek.GradientBoostingClassifier(n_estimators=50),
         "GNB":GaussianNB(),
         "LogisticRegression":LogisticRegression()   
}

results = {}
for algo in model:
    clf = model[algo]
    clf.fit(X_train, y_train)
    print("{}:".format(model[algo]))
    res = clf.predict(X_test)
    mt = confusion_matrix(y_test, res)
    results[algo] = mt[1][0] / float(sum(mt[1]))*100
    print(classification_report(y_true=y_test,y_pred=clf.predict(X_test),digits=4))
    
  

# import time 
# t1 = time.time()
# X=StandardScaler().fit_transform(features)#实现均方差值
# X_train, X_test, y_train, y_test=train_test_split(X,lable)#划分数据集
# rf=RandomForestClassifier().fit(X_train,y_train)
# t2 = time.time()
# print("it costs {}s".format(t2-t1))
# import pickle
# print(classification_report(y_true=y_test,y_pred=rf.predict(X_test),digits=4))
# with open('/home/aistudio/work/lrmodel.pickle', 'wb') as fr: 
#     pickle.dump(rf, fr)


# In[162]:


for i in results:
    print("{}：{}".format(i,results[i]))


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[163]:


clf = model["RandomForest"]
res = clf.predict(X_test)
mt = confusion_matrix(y_test, res)
print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))
print(classification_report(res,y_test,digits=4))


# In[167]:


for i in model:
    imp_dic = zip(features.columns.values,model[i].feature_importances_)
# {(features.columns.values[i],clf.feature_importances_[i])for i in range(len(clf.feature_importances_))}
# for i in range(len(clf.feature_importances_)):
#     print("feature:{},importance:{}".format(features.columns.values[i],clf.feature_importances_[i]))
    imp_dic = dict(imp_dic)
    print(sorted(imp_dic.items(),key=lambda x :x[1],reverse = True ))


# In[166]:


imp_dic = dict(imp_dic)
print(sorted(imp_dic.items(),key=lambda x :x[1],reverse = True ))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




