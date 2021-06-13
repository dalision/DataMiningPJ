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


# # **LOAD DATA**

# In[1]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code, so that every time the environment (kernel) starts, just run the following code:
import sys
sys.path.append('/home/aistudio/external-libraries')
#read csv
import pandas as pd
import ipaddress as ip #works only in python 3
from urllib.parse import urlparse
import math
import tldextract
import requests
import re
#pd.read_csv('data.csv',usecols=[0,1,2,3],names = ["id_index","url"])
es = pd.read_csv("./data/data54433/phishing_verified_online.csv")#evil url 
ns = pd.read_csv("./data/data54434/top-1m.csv",names = ["id_index","url"])#normal url 
#将csv规范化
def turndome(x):
    return str(urlparse(x["url"]).netloc)
del ns["id_index"]
ns["label"] = 1
ns["tld"] = ns.apply(lambda x: tldextract.extract(x["url"]).suffix,axis =1)
evil = es.drop(["phish_id",'phish_detail_url', "submission_time","verified","verification_time","online","target"], axis=1) 
evil["label"] = 0
evil["url"] = evil.apply(turndome,axis=1)
evil["tld"] = evil.apply(lambda x: tldextract.extract(x["url"]).suffix,axis = 1)
ns = ns[:14989]
data = pd.concat([ns,evil])
dic = dict(data["tld"].value_counts())
dic[""]=0
del data["tld"]
data.to_csv("/home/aistudio/work/data.csv")
dic
# print(data.shape)


# In[ ]:


print(ns.shape)
evil.shape


# In[ ]:


new = pd.read_csv("/home/aistudio/work/data.csv",index_col=0)
new.head(5)


# # feature engineering

# # define functions 

# In[ ]:


#特征分析
#确认域名是否为ip
def isip(uri):
    try:
        if ip.ip_address(uri):
            return 1
    except:
        return 0

# 特殊字符计数
def countdelim(url):
    count = 0
    delim=[';','_','?','=','&']
    for each in url:
        if each in delim:
            count = count + 1
    return count

# 记录点数
def countdots(url):  
    return url.count('.')

#是否存在连字符
def isPresentHyphen(url):
    return url.count('-')

#判断 @
def isPresentAt(url):
    return url.count('@')

#黑名单过滤
Suspicious_TLD=['zip','cricket','link','work','party','gq','kim','country','science','tk']
Suspicious_Domain=['luckytime.co.kr','mattfoll.eu.interia.pl','trafficholder.com','dl.baixaki.com.br','bembed.redtube.comr','tags.expo9.exponential.com','deepspacer.com','funad.co.kr','trafficconverter.biz']


#对子域名进行计数
def countSubDomain(subdomain):
    if not subdomain:
        return 0
    else:
        return len(subdomain.split('.'))

#记录字符串的信息熵

def range_bytes(): return list(range(256))
def H(data, iterator=range_bytes):
    if not data:
        return 0
    entropy = 0
    for x in iterator():
        p_x = float(data.count(chr(x))) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def cal_entropy(text):
    h = 0.0
    sum = 0
    letter = [0] * 26
    text = text.lower()
    for i in range(len(text)):
        if text[i].isalpha():
            letter[ord(text[i]) - ord('a')] += 1
            sum += 1
    # print('\n', letter)
    for i in range(26):
        p = 1.0 * letter[i] / sum
        if p > 0:
            h += -(p * math.log(p, 2))
    return h

def shortening_service( url):
    """Tiny URL -> phishing otherwise legitimate"""
    match=re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net',url)
    if match:
        return 1               
    else:
        return 0 

def ifurlcode(url):
    if "%" in url:
        return 1
    else:
        return 0 

def gettlds(url):
    num = dic.get(tldextract.extract(url).suffix)
    if num == None:
        return 0
    else:
        return num


featureSet = pd.DataFrame(columns=('url','no of dots','presence of hyphen','len of url','presence of at','presence of double slash','no of subdir','no of subdomain','len of domain','is IP','presence of Suspicious_TLD','presence of suspicious domain','label'))
dic


# # get features and prepare database 

# In[ ]:



def getFeatures(url, label): 
    result = []
    
    #add the url to feature set
    result.append(url)
    #parse the URL and extract the domain information
    ext = tldextract.extract(url)
    #counting number of dots in subdomain   
    # print(ext.subdomain) 
    # result.append(countdots(ext.subdomain))
    #checking hyphen in domain   
    result.append(int(isPresentHyphen(url)))
    #length of URL    
    result.append(int(len(url)))
    #checking @ in the url    
    result.append(int(isPresentAt(url)))
    #checking presence of double slash    
    #Count number of subdir    
    #number of sub domain    
    result.append(int(countSubDomain(ext.subdomain)))
    #length of domain name    
    #count number of queries   
    #Adding domain informati
    #if IP address is being used as a URL     
    # result.append(isip(ext.domain))
    #presence of Suspicious_TLD
    result.append(1 if ext.suffix in Suspicious_TLD else 0)
    #presence of suspicious domain
    result.append(1 if '.'.join(ext[1:]) in Suspicious_Domain else 0 )
    result.append(H(url))
    result.append(int(shortening_service(url)))
    result.append(int(ifurlcode(url)))
    result.append(gettlds(url))
    result.append(int(label))
    return result


#记录顶级域名前面数目个数；
sample = "scu.edu.cn"
# label = 1
# if tldextract.extract(sample).suffix=="":
#     print(1)
print(gettlds(sample))
# result = getFeatures(sample, label)
# print(result)
#url/url中有没有-/url的长度/检查@是否在url中/子域名的分段/黑名单1/黑名单2/信息熵/urlcode/判断是否为短连接/tld/label


# In[ ]:


df =new


# In[ ]:


# temp = pd.Series(df.apply(lambda pd:tldextract.extract(pd["url"]),axis=1))
# temp
keys = ["url","line","len","@","sub_numbers","blacklist1","blacklist2","entropy","urlcode","shortlen","tldss","label"]
results = pd.DataFrame(columns=keys)
# print(len(["url","line","len","@","sub_numbers","blacklist1","blacklist2","entropy","urlcode","shortlen","tldss","label"]))


# In[ ]:


for index, row in df.iterrows():
    url = row['url']
    label = row["label"]
    di = dict(zip(keys,getFeatures(url,label)))
    unit = pd.DataFrame(di,index =[0])
    results = pd.concat([results,unit],axis=0)
results


# In[ ]:


import numpy as np
results["line"] = results.line.astype(np.int32)
results["@"] = results["@"].astype(np.int32)
results["sub_numbers"]=results["sub_numbers"].astype(np.int32)
results["blacklist1"]=results["blacklist1"].astype(np.int32)
results["blacklist2"]=results["blacklist2"].astype(np.int32)
results["urlcode"]=results["urlcode"].astype(np.int32)
results["shortlen"]=results["shortlen"].astype(np.int32)
results["tldss"]=results["tldss"].astype(np.int32)
results["label"]=results["label"].astype(np.int32)
results["len"]=results["len"].astype(np.int32)
results.describe()


# In[ ]:


results.describe()


# In[ ]:


results.columns
del results["shortlen"]


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[ ]:


print(results.describe())
results.to_csv("/home/aistudio/work/lastfinaldata.csv")


# # get database

# In[ ]:


dig = pd.read_csv("/home/aistudio/work/lastfinaldata.csv",index_col=False)
# del dig["Unnamed: 0"]
# del dig["@"]
# del dig["blacklist2"]
# del dig["shortlen"]
dig.describe()


# In[ ]:


dig.describe()


# # train

# In[28]:


from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC
# del dig["url"]
# labels = dig["label"]
# del dig["label"]
X=StandardScaler().fit_transform(dig)#实现均方差值
X_train, X_test, y_train, y_test=train_test_split(X,labels)#划分数据集



# In[ ]:





# In[29]:


rf=RandomForestClassifier().fit(X_train,y_train)
print(classification_report(y_true=y_test,y_pred=rf.predict(X_test)))


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,auc,make_scorer,f1_score

rfc=RandomForestClassifier(random_state=42)

param_grid = { 
    'n_estimators': [20,50,100,200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
f1sc = make_scorer(f1_score, average='micro')
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5,return_train_score=True,scoring=f1sc,)
CV_rfc.fit(X_train, y_train)


# In[ ]:


print(classification_report(y_true=y_test,y_pred=CV_rfc.best_estimator_.predict(X_test)))


# In[ ]:




