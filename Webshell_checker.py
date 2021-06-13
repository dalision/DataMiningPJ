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
get_ipython().system('ls ')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[38]:


import os 
import zlib
dir1 =  r"/home/aistudio/work/data/php-webshell"
dir2 =  r"/home/aistudio/work/data/php-benign"
l1 = os.listdir(dir1)
l2 = os.listdir(dir2)
print(len(l1),len(l2))


# In[5]:


import sys
import pandas as pd
import math
import seaborn as sns
import collections
import urllib
from urllib import parse
import os
from tqdm import tqdm
import pandas as pd
tqdm.pandas(desc='pandas bar')

def openf1(path):
    path = os.path.join("/home/aistudio/work/data/php-benign",path)
    # print(path)
    f = open(path, errors='ignore')
    temp  = f.read()
    # print(temp)
    f.close()
    return temp

def openf2(path):
    path = os.path.join("/home/aistudio/work/data/php-webshell",path)
    # print(path)
    f = open(path, errors='ignore')
    temp  = f.read()
    # print(temp)
    f.close()
    return temp

# print(openf(r"article_allowurl_edit.php"))
def read1(path):
    filelist = os.listdir(path)
    # print(filelist[1])
    hhh  =[openf1(path) for path in filelist]
    # print(hhh)
    # print(filelist[1])
    dic = {"data":hhh,"filename":filelist,"label":[0 for _ in  range(len(filelist))]}
    return dic

def read2(path):
    filelist = os.listdir(path)
    # print(filelist[1])
    hhh  =[openf2(path) for path in filelist]
    # print(hhh)
    # print(filelist[1])
    dic = {"data":hhh,"filename":filelist,"label":[1 for _ in  range(len(filelist))]}
    return dic
bengin = read1(r"/home/aistudio/work/data/php-benign")
evil = read2("/home/aistudio/work/data/php-webshell")
bf = pd.DataFrame(bengin)
ef = pd.DataFrame(evil)
totol = pd.concat([bf,ef],axis=0)


# In[44]:


import numpy as np 
def linelen_stats(f):
    lines = f.splitlines()
    linelens = np.array([len(i) for i in lines])
    stats_f = [np.mean,np.median,np.max,np.var]
    return [f(linelens) for f in stats_f]

def samelenline_cnt(f):
    string = f.splitlines()
    if len(string) == 1:
        return 0
    sub = 0
    i = 0
    while i < len(string):
        if i == 0:
            if string[i] != string[i + 1]:
                sub += 1
        elif i == len(string) - 1:
            if string[i] != string[i - 1]:
                sub += 1
        else:
            if string[i] != string[i - 1] and string[i] != string[i + 1]:
                sub += 1
        i += 1
    return len(string) - sub

def samelenline_cnt(f):
    string = f.splitlines()
    if len(string) == 1:
        return 0
    sub = 0
    i = 0
    while i < len(string):
        if i == 0:
            if string[i] != string[i + 1]:
                sub += 1
        elif i == len(string) - 1:
            if string[i] != string[i - 1]:
                sub += 1
        else:
            if string[i] != string[i - 1] and string[i] != string[i + 1]:
                sub += 1
        i += 1
    return len(string) - sub
#longest
import re
def longest(data):
    longest = 0
    longest_word = ""
    words = re.split("[\s,\n,\r]", data)
    if words:
        for word in words:
            length = len(word)
            if length > longest:
                longest = length
                longest_word = word
    return longest

#正则敏感函数
def SignatureNasty(data):
    valid_regex = re.compile('(eval\(|file_put_contents|proc_exec|base64_decode|pfsockopen|ini_set|proc_get_status|python_eval|ini_restore|ini_alter|exec\(|passthru|popen|proc_open|pcntl|assert\(|system\(|chroot|shell|chgrp|chown)', re.I)
    matches = re.findall(valid_regex, data)
    return len(matches)

#匹配特殊符号
def match1(data):
    valid_regex = re.compile('(@\$_\[\]=|\$_=@\$_GET|\$_\[\+""\]=)', re.I)
    matches = re.findall(valid_regex, data)
    return len(matches)    

#匹配eval
def match2(data):
    valid_regex = re.compile('(eval\(\$(\w|\d))', re.I)
    matches = re.findall(valid_regex, data)
    return len(matches)

##Index of Coincidence
def ic(data):
    """Calculate the Index of Coincidence for a file"""
    char_count = 0
    total_char_count = 0
    for x in range(256):
        char = chr(x)
        charcount = data.count(char)
        char_count += charcount * (charcount - 1)
        total_char_count += charcount
    ic = float(char_count)/(total_char_count * (total_char_count - 1))
    return ic

#计算熵值
def entropy(data):
    entropy = 0
    stripped_data =data.replace(' ', '')
    for x in range(256):
        p_x = float(stripped_data.count(chr(x)))/len(stripped_data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def ratio(data):
    data = bytes(data, encoding = "utf8")
    compressed = zlib.compress(data)
    ratio = float(len(compressed)) / float(len(data))
    # self.results.append({"filename":filename, "value":ratio})
    return ratio



# # 除去注释+提取特征：

# In[7]:


def filterf(f):
    comm_regex_1 = '(\/\*.*?\*\/)'
    comm_regex_2 = '(//.*|#.*)'
    f = re.sub(comm_regex_1, '', f, flags=re.DOTALL | re.M | re.I | re.UNICODE)
    f = re.sub(comm_regex_2, '', f, flags=re.M | re.I | re.UNICODE)
    return f

labels = "linelen_stats samelenline_cnt longest signatureNasty match1 match2 ic entropy ".split()
funlist = [linelen_stats,samelenline_cnt,longest,SignatureNasty,match1,match2,ic,entropy]
totol["data"] = totol["data"].apply(filterf)
for i in range(len(labels)):
    totol[labels[i]] = totol["data"].progress_apply(funlist[i])


# In[16]:


totol["l1"] = totol["linelen_stats"].apply(lambda x:x[0])
totol["l2"] = totol["linelen_stats"].apply(lambda  x:x[1])
totol["l3"] = totol["linelen_stats"].apply(lambda  x:x[2])
totol["l4"] = totol["linelen_stats"].apply(lambda  x:x[3])


# In[41]:


def openfb(path):
    path = os.path.join("/home/aistudio/work/data/php-benign",path)
    # print(path)
    f = open(path, 'rb')
    temp  = f.read()
    # print(temp)
    f.close()
    return temp

def readb(path):
    filelist = os.listdir(path)
    # print(filelist[1])
    hhh  =[openf1(path) for path in filelist]
    # print(hhh)
    # print(filelist[1])
    dic = {"data":hhh,"filename":filelist,"label":[0 for _ in  range(len(filelist))]}
    return dic


# In[45]:


totol["ratio"] = totol["data"].apply(ratio)


# In[46]:


X = totol.iloc[:,4:]
Y = totol.iloc[:,2]


# In[63]:


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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X=StandardScaler().fit_transform(X)#实现均方差值
X_train, X_test, y_train, y_test=train_test_split(X,Y)#划分数据集
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
    # print("{}:".format(model[algo]))
    res = clf.predict(X_test)
    mt = confusion_matrix(y_test, res)
    results[algo] = mt[1][0] / float(sum(mt[1]))*100
    # print(classification_report(y_true=y_test,y_pred=clf.predict(X_test),digits=4))
    
  


# In[64]:


for i in results:
    print("{}：{}".format(i,results[i]))


# In[65]:


clf = model["RandomForest"]
res = clf.predict(X_test)
mt = confusion_matrix(y_test, res)
print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))
print(classification_report(res,y_test,digits=4))


# In[67]:


i = "RandomForest"
imp_dic = zip(totol.iloc[:,4:].columns.values,model[i].feature_importances_)
imp_dic = dict(imp_dic)
print(sorted(imp_dic.items(),key=lambda x :x[1],reverse = True ))
imp_dic = dict(imp_dic)
# print(sorted(imp_dic.items(),key=lambda x :x[1],reverse = True ))


# In[70]:


import time
t1 = time.time()
clf = model["RandomForest"]
res = clf.predict(X_test)
t2 = time.time()
print("it costs {}s".format((t2-t1)))


# In[8]:


import numpy as np 
ay = np.array([[1,4,5,7,34,32],[1,32,24,5,35,3],[1,32,24,5,35,3]])
ay = np.delete(ay,[1,0],0)
ay


# 

# 

# 

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
