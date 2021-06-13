#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, you need to use the persistence path as the following:
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code, so that every time the environment (kernel) starts, just run the following code:
import sys
sys.path.append('/home/aistudio/external-libraries')


# # Load Data

# In[2]:


import sys
import pandas as pd
import math
import seaborn as sns
import collections
dga = pd.read_csv("work/dg.csv",names=["domain"])
normal = pd.read_csv("data/data57259/umbrella-top-1m.csv",usecols=[1],names=["domain"])
dga["domain"] = dga["domain"].str.replace("\\t","")
normal["label"] = 0
dga["label"] = 1
print(dga.shape)
print(normal.shape)
dga = dga.iloc[:100000,:]
normal = normal.iloc[:100000,:]
sf = pd.concat([dga,normal],axis=0)
sf.head()


# In[12]:


get_ipython().system('pip install nltk')
import nltk
from nltk.collocations import  BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
domain = ["11","22","33",'33']
Bigram_measures = nltk.collocations.BigramAssocMeasures()
Bigram_finder = BigramCollocationFinder.from_words(domain)
scored = Bigram_finder.score_ngrams(Bigram_measures.poisson_stirling)
scored


# # Features engeneering

# 

# In[ ]:


get_ipython().system('pip install progressbar')


# In[ ]:


#提取good中的常用域名
get_ipython().system('pip install progressbar')
import urllib
from progressbar import *


#判断是否在good的topTLD中
common_root={}
progress = ProgressBar()
good_list=list(normal['domain'])
# common_root = ['cn','com','cc','net','org','gov','info']

for i in progress(range(len(good_list))):
  extraction=good_list[i].split(".")
  
  ext=extraction[-1]
  if ext in common_root.keys():
    common_root[ext]+=1
  else:
    common_root[ext]=1

print(common_root)
common_root_sorted=sorted(common_root.items(), key = lambda kv:(kv[1], kv[0]),reverse = True)
print(common_root_sorted[:])

# 选择排名前1%的后缀名
common_root_list=[]
for i in common_root_sorted:
  if i[1]>len(common_root_sorted)/100:
    common_root_list.append(i[0])
print(common_root_list)



def cal_ext(x):
  extraction=x["domain"].split(".")
  ext=extraction[-1]
  if ext in common_root_list:
    return 1
  else:
    return 0


# In[ ]:


#len
def length(x):
    return len(x["domain"])
sf["len"] = normal.apply(length,axis=1)


# In[ ]:


def count_digits(x):#有多少数字
    digits=list('0123456789')
    return sum(digits.count(i) for i in x["domain"].lower())/len(x["domain"])

def count_repeat_letter(x):#重复的字母
    count = collections.Counter(i for i in x["domain"].lower() if i.isalpha()).most_common()
    cnt = 0
    for letter,ct in count:
        if ct>1:
            cnt+=1
    return cnt



# In[ ]:


from itertools import groupby
def consecutive_consonant(x):#找到连清辅音
    cnt = 0
    #consonant = set(['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'z'])
    consonant = set(['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w','x', 'y', 'z'])
    digit_map = [int(i in consonant) for i in x["domain"]]
    consecutive=[(k,len(list(g))) for k, g in groupby(digit_map)]
    count_consecutive = sum(j for i,j in consecutive if j>1 and i==1)
    return count_consecutive


# In[ ]:


def getroot(x):#得到最后的域名
    common = ['cn','com','cc','net','org','gov','info']
    temp = x["domain"].split('.')[-1]
    return temp in common

def getshan(x):#熵
	tmp_dict = {}
	domain_len = len(x["domain"])
	for i in range(0,len(x["domain"])):
		if x["domain"][i] in tmp_dict.keys():
			tmp_dict[x["domain"][i]] = tmp_dict[x["domain"][i]] + 1
		else:
			tmp_dict[x["domain"][i]] = 1
	shannon = 0
	for i in tmp_dict.keys():
		p = float(tmp_dict[i]) / domain_len
		shannon = shannon - p * math.log(p,2)
	return shannon

def getyuanyin(x):#原音占比
	yuan_list = ['a','e','i','o','u']
	domain = x["domain"].lower()
	count_word = 0
	count_yuan = 0
	yuan_ratio = 0
	for i in range(0,len(domain)):
		if ord(domain[i]) >= ord('a') and ord(domain[i]) <= ord('z'):
			count_word = count_word + 1
			if domain[i] in yuan_list:
				count_yuan = count_yuan + 1
	if count_word == 0:
		return yuan_ratio
	else:
		yuan_ratio = float(count_yuan) / count_word
		return yuan_ratio
#bigram特征
def get_bigram(x):
    finder = BigramCollocationFinder.from_words(x["domain"])#定义finder
    bigram_measures = nltk.collocations.BigramAssocMeasures()#
    scored = finder.score_ngrams(bigram_measures.pmi)#排名方式
    result = 0
    for i in scored:
        result = result + i[1]
    return int(result)
#trigram特征
def get_trigram(x):
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    trigram_finder = TrigramCollocationFinder.from_words(x["domain"])
    scored = trigram_finder.score_ngrams(trigram_measures.poisson_stirling)
    result = 0
    for i in scored:
        result = result + i[1]
    return int(result)


# In[ ]:


get_ipython().system('pip install tldextract')
import tldextract
import nltk
from nltk.collocations import *
from nltk.metrics import  BigramAssocMeasures


# In[ ]:


sf["bigram"]= sf.apply(get_bigram,axis=1)
sf["trigram"]= sf.apply(get_trigram,axis=1)
sf["len"] = sf.apply(length,axis = 1)
print("!!!!!!!!!!")
sf["cal_ext"] = sf.apply(cal_ext,axis =1)
print("!!!!!!!!!!")
sf["count_digits"] = sf.apply(count_digits,axis = 1)
print("!!!!!!!!!!")
sf["count_repeat_letter"] = sf.apply(count_repeat_letter,axis = 1)
print("!!!!!!!!!!")
sf["consecutive_consonant"] = sf.apply(consecutive_consonant,axis = 1)
print("!!!!!!!!!!")
sf["getroot"] = sf.apply(getroot,axis = 1)
print("!!!!!!!!!!")
sf["getshan"] = sf.apply(getshan,axis = 1)
sf["getyuanyin"] = sf.apply(getyuanyin,axis = 1)
sf["getcommonurl"]= sf.apply(cal_ext,axis=1)



# In[ ]:


import seaborn as sns
import pandas
sns.countplot(x='cal_ext', hue='label',data=sf)


# In[ ]:


import seaborn as sns
import pandas
sns.countplot(x='getcommonurl', hue='label',data=sf)


# In[ ]:


ax1 = sns.kdeplot(sf.loc[sf["label"]==1,"bigram"],shade=True,color='r',label="DGA")
ax1 = sns.kdeplot(sf.loc[sf["label"]==0,"bigram"],shade=True,color='b',label="Normal")


# In[ ]:


ax2 = sns.kdeplot(sf.loc[sf["label"]==1,"trigram"],shade=True,color='r',label="DGA")
ax2 = sns.kdeplot(sf.loc[sf["label"]==0,"trigram"],shade=True,color='b',label="Normal")


# In[ ]:


sns.regplot(x="trigram", y="bigram", data=sf)


# In[ ]:


features = sf.iloc[:,2:]
lable = sf["label"]
features = features.drop(["trigram","cal_ext","getcommonurl"],axis=1)


# # 模型选择

# In[ ]:


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


# In[ ]:





# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[ ]:


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
    results[algo] = clf.score(X_test,y_test)
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


# In[ ]:





# In[ ]:


for i in results:
    print("{}：{}".format(i,results[i]))


# # 单独训练rf

# In[25]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,auc,make_scorer,f1_score
X=StandardScaler().fit_transform(features)#实现均方差值
X_train, X_test, y_train, y_test=train_test_split(X,lable)#划分数据集
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [20,50,100,200, 500],
    # 'max_features': ['auto', 'sqrt', 'log2'],
    # 'max_depth' : [4,5,6,7,8],
    # 'criterion' :['gini', 'entropy']
}
f1sc = make_scorer(f1_score, average='micro')
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3,return_train_score=True,scoring=f1sc,)
CV_rfc.fit(X_train, y_train)
print(classification_report(y_true=y_test,y_pred=CV_rfc.best_estimator_.predict(X_test),digits=4))


# In[24]:


clf = model[max(results)]
res = clf.predict(X_test)
mt = confusion_matrix(y_test, res)
print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))
print(classification_report(res,y_test,digits=4))


# In[ ]:


imp_dic = zip(features.columns.values,clf.feature_importances_)
# {(features.columns.values[i],clf.feature_importances_[i])for i in range(len(clf.feature_importances_))}
# for i in range(len(clf.feature_importances_)):
#     print("feature:{},importance:{}".format(features.columns.values[i],clf.feature_importances_[i]))
imp_dic


# In[ ]:


imp_dic = dict(imp_dic)
print(sorted(imp_dic.items(),key=lambda x :x[1],reverse = True ))


# In[ ]:




