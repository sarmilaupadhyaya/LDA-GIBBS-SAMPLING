#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from collections import defaultdict
import random
import numpy as np
from collections import Counter


# In[2]:


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


# In[ ]:


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


# In[ ]:


def lda_gibbs_without_optimization():
    """
    
    lda gibbs sampling code without optimization"""
    
    from datetime import datetime
    a=datetime.now()

    for _ in range(500):
        for i,d in enumerate(new_docs):
            for j,w in enumerate(d):
                topic = z[i][j]
                n_k[topic] -= 1
                n_d_k[i,topic] -= 1
                n_k_w[topic,w] -= 1
                prob = np.zeros([K])
                for k in range(K):
                    p = (n_d_k[i, k]* n_k_w[k, w])/ (n_k[k]*((len(d)-1)+K*alpha))
                    prob[k] = p
               
                topic = np.random.multinomial(1,np.divide(prob,prob.sum())).argmax()
                z[i][j]= topic
                n_k[topic] += 1
                n_d_k[i,topic] += 1
                n_k_w[topic,w] += 1

    print((datetime.now()-a).total_seconds())
    return n_k_w
        


# In[4]:


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
        


# In[ ]:


#lda gibbs sampling by initializing same topic to words of same doc
n_d_k, n_k, n_k_w, new_docs,Z = initialization()
n_k_w = lda_gibbs()


# In[5]:


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


# Discussion:
#     
# a. Optmization using numpy
#     I tried numpy as the initial method which took nearly 4 hours before. I was using 4 for loops, one the iteration, second through documents, third for words in each document and final through each topic. However, after I removed the loop for topics since it could be optimized using numpys array operation. Now it takes exactly 2 and half hour.
#     

# b. Coeherent SImilarity between topics. 
# 
# Above are the top 10 words printed for each topic. Since, the dataset is movie review we can see the clear distinction between genre of movies. 
# 
# Topic 0: It is about thriller category of movie. There are some trace of number and sequel here since number denoted the sequel of movies.
#     
# Topic 2: This topic is about animated disney movie.
# 
# Topic 3: This topic has hints about movie titanic.
# 
# Topic 4: it definetle indicates some comedy movie.
# 
# Topic 5: thriller and cops makes sense for it to be mvoies with action, thriller, cops.
# 
# Topic 6: This topic indicates the war movie saving private ryan. Words like men, war, soldiers and army indicated the war genre.
# 
# Topic 15: words like vampire and series direct to the series called vampire diaries. SImilarly, the actor playing vampire in one movie is acting as batman in other. Hence, we can see the connection here with word batman.
# 
# Topic 8: These words are related more to the romantic comedy movie with wedding scences.
#     
# Others topics are not that familiar to me since I do not have much knowledge of hollywood movies categories.
# 
# There are few more general and broad topics as well like topic 1.
# 

# I tried assigning same topic to words of same documents. I ran this code in different notebook to run parallelly and hence copied the result of topics in the markdown here.

# In[ ]:


#lda gibbs sampling by initializing same topic to words of same doc
n_d_k, n_k, n_k_w, new_docs,Z = initialization(same_topic=True)
n_k_w = lda_gibbs()


# In[ ]:


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


# Topic  0
# film one like movie story character even characters time two
# 
# Topic  1
# action crime cop police carter murder prison thriller killer jack
# 
# Topic  2
# jackie chan tarantino bond chinese fight martial brown action fiction
# 
# Topic  3
# tarzan shakespeare spice lebowski girls dude joe babe cauldron jane
# 
# Topic  4
# wedding ryan sam julia romantic joe annie max bacon angels
# 
# Topic  5
# west wild rising virus smith cruise hollow depp monster deep
# 
# Topic  6
# truman carrey toy jim disney kids show woody stuart mr
# 
# Topic  7
# school high jay smith patch bulworth mike football silent bob
# 
# Topic  8
# war batman godzilla ryan men battle spielberg private robin soldiers
# 
# Topic  9
# vampire vampires van damme apes blade carpenter planet action seagal
# 
# Topic  10
# star effects special wars science trek space alien aliens planet
# 
# Topic  11
# sex zero sexual house whale helen porter jawbreaker martha haunting
# 
# Topic  12
# movie comedy funny like get good one jokes big really
# 
# Topic  13
# film movie one like even good bad would time get
# 
# Topic  14
# shrek rocky bobby musical sonny hunting krippendorf tribe granger hedwig
# 
# Topic  15
# disney animated mulan animation king voice arnold army songs travolta
# 
# Topic  16
# scream alien killer horror 2 3 williamson ripley lynch urban
# 
# Topic  17
# nights austin boogie dr myers alice joe 54 powers evil
# 
# Topic  18
# flynt simon derek larry ghost black dog melvin norton dora
# 
# Topic  19
# harry titanic ship rose allen elizabeth dicaprio cameron town kate
# 

# If we compare topic 1 here and topic 5 in previous lda method (assigning different topic to each words), we can see that here topic 1 indicates clearer picture of a thriller movie.
# 

# I like the result better here since it gives us supervised start to topic modelling.

# In[ ]:




