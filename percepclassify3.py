import string
import re
import ast
import os
import sys
import numpy as np


#function to preprocess each string by 
#1.removing punctuation 
#2.converting each string to lower case 
#3.splitting each string to a list of words
def tokenize(s):
    s=s.translate(str.maketrans('', '', string.punctuation))
    s=s.lower()
    strlist = re.sub("[^\w]", " ",  s).split()
    return strlist

#funtion that computes class(truthful/deceptive) of review based on weights present in the'averagedmodel.txt' file
#uses weights of an averaged perceptron to compute class of output
def TestTruthAveraged(document,vocab,u,B):
    document=tokenize(document)

    x_list=[]
    for j in vocab:
        x=document.count(j)
        x_list.append(x)
    x_list=np.array(x_list)
    a_averaged=np.sign(np.dot(x_list,np.array(u))+B)
    if a_averaged>0:
        return 'truthful'
    else:
        return 'deceptive'

#funtion that computes class(truthful/deceptive) of review based on weights present in the'vanillamodel.txt' file
#uses weights of an vanilla perceptron to compute class of output
def TestTruthVanilla(document,vocab,w,b):
    document=tokenize(document)
    x_list=[]
    for j in vocab:
        x=document.count(j)
        x_list.append(x)
    x_list=np.array(x_list)
    w=np.array(w)
    a_vanilla=np.sign(np.dot(x_list,np.array(w))+b)
    if a_vanilla>0:
        return 'truthful'
    else:
        return 'deceptive'

#funtion that computes class(positive/negative) of review based on weights present in the'vanillamodel.txt' file
#uses weights of an vanilla perceptron to compute class of output
def TestSentiVanilla(document,vocab,w,b):
    document=tokenize(document)
    x_list=[]
    for j in vocab:
        x=document.count(j)
        x_list.append(x)
    x_list=np.array(x_list)
    w=np.array(w)
    a_vanilla=np.sign(np.dot(x_list,np.array(w))+b)
    if a_vanilla>0:
        return 'positive'
    else:
        return 'negative'
    
#funtion that computes class(positive/negative)of review based on weights present in the'averagedmodel.txt' file
#uses weights of an averaged perceptron to compute class of output
def TestSentiAveraged(document,vocab,u,B):
    document=tokenize(document)
    
    x_list=[]
    for j in vocab:
        x=document.count(j)
        x_list.append(x)
    x_list=np.array(x_list)
    a_averaged=np.sign(np.dot(x_list,np.array(u))+B)
    if a_averaged==1:
        return 'positive'
    elif a_averaged==-1:
        return 'negative'
    
    
    
#input specifying type of perceptron(vanilla/averaged)
model=sys.argv[1]
#input specifying path to test data
test_file = sys.argv[2]

#reading parametes computed for vanilla perceptron in the perceplearn file from the training data
if 'vanillamodel.txt'in model:
    filehandle=[]
    with open('vanillamodel.txt', 'r') as file:
        for f in file:
            filehandle.append(f)
    vocab1=ast.literal_eval(filehandle[0])
    vocab2=ast.literal_eval(filehandle[1])
    w_senti=ast.literal_eval(filehandle[2])
    b_senti=ast.literal_eval(filehandle[3])
    w_tru=ast.literal_eval(filehandle[4])
    b_tru=ast.literal_eval(filehandle[5])
    vocab1=sorted(list(vocab1))
    vocab2=sorted(list(vocab2))
    file.close()
    #computing class and writing result to output files
    with open('percepoutput.txt', 'w') as output_file:    
        for dirpath,sub,files in os.walk(test_file):
            for filename in files:
                if filename.endswith('.txt') and 'README' not in filename:
                        file = open(os.path.join(dirpath, filename), 'r')
                        data = file.read()
                        predict_senti=TestSentiVanilla(data,vocab1,w_senti,b_senti)
                        predict_truth=TestTruthVanilla(data,vocab2,w_tru,b_tru)
                        output_file.write(predict_truth+' '+predict_senti+' '+dirpath+'/' + filename+'\n')


#reading parametes computed for averaged perceptron in the perceplearn file from the training data
if 'averagedmodel.txt'in model:
    filehandle=[]
    with open('averagedmodel.txt', 'r') as file:
        for f in file:
            filehandle.append(f)
    
    vocab1=ast.literal_eval(filehandle[0])
    vocab2=ast.literal_eval(filehandle[1])
    u_senti=ast.literal_eval(filehandle[2])
    B_senti=ast.literal_eval(filehandle[3])
    U_tru=ast.literal_eval(filehandle[4])
    b_tru=ast.literal_eval(filehandle[5])
    vocab1=sorted(list(vocab1))
    vocab2=sorted(list(vocab2))
    file.close()
    #computing class and writing result to output files
    with open('percepoutput.txt', 'w') as output_file:    
        for dirpath,sub,files in os.walk(test_file):
            for filename in files:
                if filename.endswith('.txt') and 'README' not in filename:
                        file = open(os.path.join(dirpath, filename), 'r')
                        data = file.read()
                        predict_senti=TestSentiAveraged(data,vocab1,u_senti,B_senti)
                        predict_truth=TestTruthAveraged(data,vocab2,U_tru,b_tru)
                        output_file.write(predict_truth+' '+predict_senti+' '+dirpath+'/' + filename+'\n')
                    
