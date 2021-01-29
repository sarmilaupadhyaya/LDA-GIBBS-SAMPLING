#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from collections import defaultdict
import random
import numpy as np
from collections import Counter



def preprocessing(path):
    """
    read data and preprocess it
    """
    unique_words = set()
    with open(path,"r") as myfile:
        head = []
        for line in myfile:
            words = line.strip().split()
            head.append(words)
            for w in words:
                unique_words.add(w)
        
    unique_words =list(unique_words) 
    head = head[1:]
    id2words = {i:word for i,word in enumerate(unique_words)}
    words2id = {word:i for i,word in enumerate(unique_words)}
    return id2words, words2id,unique_words, head



id2words, words2id, unique_words, head = preprocessing("movies-pp.txt")


# In[3]:




D = len(head)
K = 20
alpha = 0.02
beta = 0.1
N = len(unique_words)


n_d_k = np.zeros([D, K]) + alpha
n_k_w = np.zeros([K, N]) + beta
n_k = np.zeros([K]) + N * beta
z = []

def initialization(same_topic = False):
    """
    function to initialize counts.
    Parameters:
    same_topic (bool): whether to assign same topic to words in same document or not
    """
    new_docs = []
    
    for d, doc in enumerate(head):
        z_current = []
        doc_current = []
        if same_topic:
            topic = random.choice(list(range(K)))
        for word in doc:
            pz = np.divide(np.multiply(n_d_k[d, :], n_k_w[:, words2id[word]]), n_k)
            if not same_topic:
                topic = random.choice(list(range(K)))
            z_current.append(topic)
            n_d_k[d, topic] += 1
            n_k_w[topic, words2id[word]] += 1
            n_k[topic] += 1
            doc_current.append(words2id[word])
        
        z.append(z_current)
        new_docs.append(doc_current)
    return n_d_k, n_k, n_k_w, new_docs, z




def lda_gibbs():
    
    """
    
    lda gibbs sampling code with optimization"""
    
    from datetime import datetime
    a=datetime.now()

    for _ in range(1):
        for i,d in enumerate(new_docs):
            for j,w in enumerate(d):
                topic = z[i][j]
                n_k[topic] -= 1
                n_d_k[i,topic] -= 1
                n_k_w[topic,w] -= 1
                prob = np.divide(np.multiply(n_d_k[i, :], n_k_w[:, w]), np.multiply(n_k, (len(d)-1)+K*alpha) )
                topic = np.random.multinomial(1,np.divide(prob,prob.sum())).argmax()
                z[i][j]= topic
                n_k[topic] += 1
                n_d_k[i,topic] += 1
                n_k_w[topic,w] += 1

    print((datetime.now()-a).total_seconds())
    return n_k_w
        



#lda gibbs sampling by initializing same topic to words of same doc
n_d_k, n_k, n_k_w, new_docs,Z = initialization()
n_k_w = lda_gibbs()



#code to get top words for each topic
topicwords = []
maxTopicWordsNum = 10
for z in range(0, K):
    ids = n_k_w[z, :].argsort()
    topicword = []
    print("Topic ", str(z))
    for j in ids:
        topicword.insert(0, id2words[j])
    print(" ".join(topicword[0 : min(10, len(topicword))]))
    topicwords.append(topicword[0 : min(10, len(topicword))])






