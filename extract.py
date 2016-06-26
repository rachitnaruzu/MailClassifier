import os
import io
import re
import math
import numpy as np
from nltk.stem import porter

'''
def split_regex(s):
    lowercase = s.lower()
    return re.compile("[\]\[=!()+-:',.@? \t\n0-9|\/]").split(lowercase)
'''
    
def tokenize(s):
    s = s.lower()
    s = re.compile('(http|https)://[^\s]*').sub('httpaddr', s)
    s = re.compile('[^\s]+@[^\s]+').sub('emailaddr', s)
    s = re.compile('[0-9]+').sub('numbber', s)
    s = re.compile('[^a-z ]+').sub(' ', s)
    words = re.compile('[ ]+').split(s)

    stemmer = porter.PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    distinct_words = set(words)
    words = [str(word) for word in distinct_words]
    
    return words

def get_word_freq_tuples(msg):

    wrd_map = {}
    words = tokenize(msg)
    for word in words:
        if len(word) > 1:
            if word in wrd_map:
                wrd_map[word] += 1
            else:
                wrd_map[word] = 1
                    
    
    sorted_freq = sorted(wrd_map.items(),key= lambda t:(t[1],t[0]),reverse=True)
        
    return sorted_freq
        

def get_mail(path):
    msg = ""
    with io.open(path,encoding="Latin-1") as fp:
        for line in fp:
            msg = msg + line
    return msg
            

def get_total_spam_or_ham_words(path):
    files = os.listdir( path )
    training_text = ""
    no_of_taining_files = len(files)
    for i in range(no_of_taining_files):
        training_text = training_text + get_mail(path + files[i])
    return get_word_freq_tuples(training_text)


def get_features():
    basepath = "E:\\Work\\Mail_Classifier\\mails\\"
    
    ham_word_tuples = get_total_spam_or_ham_words(basepath + "ham\\")
    spam_word_tuples = get_total_spam_or_ham_words(basepath + "spam\\")
        
    a =  np.array(ham_word_tuples[:500])
    b = np.array(spam_word_tuples[:500])
    
    features = np.concatenate([a,b]) 
    
    feature_keys = [feature[0] for feature in features]
    
    distinct_features = set(feature_keys)
    
    distinct_features = sorted(distinct_features)
    
    features = [str(word) for word in distinct_words]
    
    return distinct_features


#print (get_features())    
