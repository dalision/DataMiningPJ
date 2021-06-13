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


# # 读取数据

# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code, so that every time the environment (kernel) starts, just run the following code:
import sys
sys.path.append('/home/aistudio/external-libraries')
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
npath = "/home/aistudio/data/data52101/dmzo_nomal.csv"
xpath ="/home/aistudio/data/data52101/xssed.csv"
#index_col=1 names= columns是对之后的定义
nf = pd.read_csv(npath,names=["params","label"])
xf = pd.read_csv(xpath,names=["params","label"])#注意是names,colums是之后的事情
nf = pd.DataFrame(nf)
xf = pd.DataFrame(xf)
nf["label"] = 0
xf["label"] = 1
sf = pd.concat([nf,xf])
sf


# # 分词

# In[ ]:


import nltk
import re
from urllib.parse import unquote
import logging

def GeneSe(payload):
    #数字泛化为"0"

    payload=payload["params"].lower()
    
    payload=unquote(unquote(payload))
    payload,num=re.subn(r'\d+',"0",payload)
    #替换url为”http://u
    payload,num=re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    #分词
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    '''
    return nltk.regexp_tokenize(payload, r)
sf["parsed"]=sf.apply(GeneSe,axis = 1)
sf.head(5)


# 

# # 生成词表

# In[ ]:


ans = []
sf['parsed'].apply(lambda x:[ans.append(i) for i in x])
word = {"word":ans}
wtable = pd.DataFrame(word)
words = wtable["word"].value_counts()[:300]
wordt = {"word":words.index,"num":words.values}
wnum = pd.DataFrame(wordt)
wnum.head(5)


# # 泛化

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[ ]:


global  filterlist
filterlist  = wnum["word"].values
def filterfunc(x):
    global  filterlist
    return x in filterlist

def dele(x):
    temp = []
    for i in x["parsed"]:
        temp.append(i if filterfunc(i) else "WORD")
    return temp
sf['words'] = sf.apply(dele,axis=1)
sf.head(5)


# # 生成词向量

# In[ ]:


get_ipython().system('pip install gensim')
from gensim.models.word2vec import Word2Vec


# In[6]:


#不要加载
embedding_size=128
skip_window=5
num_sampled=64
num_iter=100
data_set = sf['words']
model=Word2Vec(data_set,size=embedding_size,window=skip_window,negative=num_sampled,iter=num_iter)


# In[ ]:


#不要加载
model.save('/home/aistudio/work/model_word2vec')


# In[7]:


global model_new
model_new = Word2Vec.load('/home/aistudio/work/model_word2vec')
type(model_new["WORD"])
import numpy as np
model_new["WORD"].shape


# # 分词向量化

# In[8]:


embeddings=model_new.wv
def out(x):
    global model_new
    sum1 = np.zeros((128))
    for i in x["words"]:
        sum1 +=model_new[i]
    return sum1/len(x["words"])
sf["features"] =sf.apply(out,axis=1)
sf.head(10)


# In[ ]:


#不要加载
labels = sf["label"]
features = sf["features"].values[:10000]
final = np.empty(shape=[0, 128])
for feature in features:
    feature=feature.reshape((1,128))
    final =np.append(final, feature, axis = 0)


# In[ ]:


#不要加载
labels = sf["label"]
features = sf["features"].values[10000:20000]
final1 = np.empty(shape=[0, 128])
for feature in features:
    feature=feature.reshape((1,128))
    final1 =np.append(final1, feature, axis = 0)


# In[ ]:


#不要加载
labels = sf["label"]
features =[ sf["features"].values[10000*i:10000*(i+1)] for i in range(6)]
features.append(sf["features"].values[10000*6:])
lf = []
for feature in features:
    final2 = np.empty(shape=[0, 128])
    for f in feature:
        f=f.reshape((1,128))
        final2 =np.append(final2, f, axis = 0)
    lf.append(final2)


# In[ ]:


#不要加载
final = np.vstack(lf)


# In[ ]:


#不要加载
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC
# del dig["url"]
# labels = dig["label"]
# del dig["label"]
final = final.astype(np.float32)
ff = pd.DataFrame(final)



# In[ ]:


#不要加载
ff.dropna(inplace=True)
ff.fillna(0)
ff.shape


# # 模型训练与验证

# In[ ]:


#不要加载
X=StandardScaler().fit_transform(ff)#实现均方差值
X_train, X_test, y_train, y_test=train_test_split(X,labels[:-1])#划分数据集
rf=RandomForestClassifier().fit(X_train,y_train)
import pickle
print(classification_report(y_true=y_test,y_pred=rf.predict(X_test)))
with open('/home/aistudio/work/lrmodel.pickle', 'wb') as fr: 
    pickle.dump(rf, fr)


# In[9]:


import pickle
def test(payload):
    dic = {"params":payload}
    t = pd.Series(dic)
    t = GeneSe(t)
    dic = {"parsed":t}
    print(payload)
    t = pd.Series(dic)
    t = dele(t)
    sum1 = np.zeros(shape=128)
    model_new = Word2Vec.load('/home/aistudio/work/model_word2vec')
    for word in t:
        sum1 += model_new[word]
    sum1 = sum1/float(len(t))
    # rf=RandomForestClassifier()
    with open("/home/aistudio/work/lrmodel.pickle","rb") as f:
        rf = pickle.load(f)
    sum1 = np.expand_dims(sum1, 0)
    print(sum1.shape)
    return rf.predict(sum1)


# In[10]:


k = input()
if test(k):
    print("eval one")
else:
    print("normal one")


# In[ ]:


q = get_ipython().run_line_magic('3Cscript%3Ealert%28%22m%22%29%3C/script%3E', '')
QueryText="><script>alert('X')</script>
ring=snape_slash_fleet;action=info;b= 	
ring=harrypotterforad

