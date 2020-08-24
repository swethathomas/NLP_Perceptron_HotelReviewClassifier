import os
import string
import re
from collections import Counter
import sys
import numpy as np

#initialize list of stopwords
stopwords=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", 
 "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
 "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
 "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", 
 "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
 "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
 "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


#function to preprocess each string by 
#1.removing punctuation 
#2.converting each string to lower case 
#3.splitting each string to a list of words
def tokenize(s):
    s=s.translate(str.maketrans('', '', string.punctuation))
    s=s.lower()
    strlist = re.sub("[^\w]", " ",  s).split()
    return strlist

#function to compute weight and bias of the perceptron for positive/negative classification   
def trainSenti(document):
    docs=[]
    wordlist=[]
    
    for place,item in enumerate(document):
        docs.append((tokenize(item[0]),document[place][1]))
        
        for i in docs[place][0]:
            wordlist.append(i)
    #remove least and most frequebt words as well as words in the stop words list
    ctr= Counter(wordlist) 
    least= 6
    most=10000
    vocab=[d for d in wordlist if ctr[d] > least and ctr[d] < most and d not in stopwords]
    vocab=sorted(list(set(vocab)))
    
    
    x_train=[]
    #remove most frequent words(stop words) and least frequent words
    for i in range(0,len(docs)):
        x_list=[]
        for j in vocab:
            x=docs[i][0].count(j)
            x_list.append(x)
        x_train.append((np.array(x_list),docs[i][1]))
    #initialize weights and bias
    w=np.zeros(len(vocab))
    b=0
    u=np.zeros(len(vocab))
    B=0
    c=0
    #Run perceptron updation algorithm for 50 iterations 
    for j in range(0,50):
        x_train=np.random.permutation(x_train)
        for i in range(0,len(x_train)):
            a=np.dot(w,x_train[i][0])+b
            c+=1
             #Run perceptron updation algorithm for 50 iterations 
            if x_train[i][1]*a<=0:
                w=w+x_train[i][1]*x_train[i][0]
                b=b+x_train[i][1]
                u=u+c*x_train[i][1]*x_train[i][0]
                B=B+c*x_train[i][1]
     #return weights calculated for vanialla as well as averaged perceptron
    return vocab,list(w-u/c),(b-B/c),list(w),b

#function to compute weight and bias of the perceptron for truthful/dceptive classification   
def trainTrudec(document):
    
    docs=[]
    wordlist=[]
    
    for place,item in enumerate(document):
        docs.append((tokenize(item[0]),document[place][1]))
    
        for i in docs[place][0]:
            wordlist.append(i)
            
    #remove least and most frequebt words as well as words in the stop words list
    ctr= Counter(wordlist) 
    least = 6
    most=10000
    vocab=[d for d in wordlist if ctr[d] > least and ctr[d] < most and d not in stopwords]
    vocab=sorted(list(set(vocab)))
    
    x_train=[]
    for i in range(0,len(docs)):
        x_list=[]
        for j in vocab:
            x=docs[i][0].count(j)
            x_list.append(x)
        x_train.append((np.array(x_list),docs[i][1]))
    #initialize weights and bias
    w=np.zeros(len(vocab))
    b=0
    c=0
    u=np.zeros(len(vocab))
    B=0
    #Run perceptron updation algorithm for 55 iterations 
    for j in range(0,55):
        x_train=np.random.permutation(x_train)
        for i in range(0,len(x_train)):
            a=np.dot(w,x_train[i][0])+b
            c+=1
            #Run perceptron updation algorithm for 50 iterations 
            if x_train[i][1]*a<=0:
                w=w+x_train[i][1]*x_train[i][0]
                b=b+x_train[i][1]
                u=u+c*x_train[i][1]*x_train[i][0]
                B=B+c*x_train[i][1]
    #return weights calculated for vanialla as well as averaged perceptron
    return vocab,list(w-u/c),(b-B/c),list(w),b


 #file containing train data
train_file = sys.argv[1]

#store training data in lists
document1=[]
document2=[]
for dirpath,sub,files in os.walk(train_file):
    for filename in files:
        if filename.endswith('.txt') and 'README' not in filename:
            if 'positive' in dirpath:
                file = open(os.path.join(dirpath, filename), 'r')
                data = file.read()
                document1.append((data,1))
            if 'negative' in dirpath:
                file = open(os.path.join(dirpath, filename), 'r')
                data = file.read()
                document1.append((data,-1))
            if 'truthful' in dirpath:
                file = open(os.path.join(dirpath, filename), 'r')
                data = file.read()
                document2.append((data,1))
            if 'deceptive' in dirpath:
                file = open(os.path.join(dirpath, filename), 'r')
                data = file.read()
                document2.append((data,-1))


vocab1,u1,B1,w1,b1= trainSenti(document1)
vocab2,u2,B2,w2,b2= trainTrudec(document2)


#write parameters calculated for vanilla model in a txt file
with open('vanillamodel.txt', 'w') as filehandle:
    filehandle.write(str(vocab1)+'\n')
    filehandle.write(str(vocab2)+'\n')
    filehandle.write(str(w1)+'\n')
    filehandle.write(str(b1)+'\n')
    filehandle.write(str(w2)+'\n')
    filehandle.write(str(b2)+'\n')
filehandle.close()

#write parameters calculated for averaged model in a txt file
with open('averagedmodel.txt', 'w') as filehandle:
    filehandle.write(str(vocab2)+'\n')
    filehandle.write(str(vocab2)+'\n')
    filehandle.write(str(u1)+'\n')
    filehandle.write(str(B1)+'\n')
    filehandle.write(str(u2)+'\n')
    filehandle.write(str(B2)+'\n')
filehandle.close()

