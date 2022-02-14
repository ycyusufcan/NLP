import os
import re
import pickle
from nltk.stem.porter import PorterStemmer

modelG=pickle.load(open("spam_classifier.pickle","rb"))
vectorizer=pickle.load(open("spam_classifier_vectorizer.pickle","rb"))

ps = PorterStemmer()

from nltk.corpus import stopwords
stop=set(stopwords.words("english"))

def read_ham(ham_dir):
    ham_list=[]
    for mail in os.listdir(ham_dir):
        with open("{}\{}".format(ham_dir,mail),"r", encoding='ansi') as file:
            scan_mail=file.read()
            ham_list.append(scan_mail)
    #print(scan_mail)
    #print(scanned_list[5])
    return ham_list

def read_spam(spam_dir): # sadece bir directoryden alır for içerisinde 
    spam_list=[]
    for mail in os.listdir(spam_dir):
        with open("{}\{}".format(spam_dir,mail),"r", encoding='ansi') as file:
            scan_mail=file.read()
            spam_list.append(scan_mail)
    #print(scan_mail)
    #print(scanned_list[5])
    return spam_list

def delete(s):
    pattern="[^A-Za-z]"
    return re.sub(pattern, " ", s)

def remove_StopWords(s):
    s=delete(s)
    word_list=s.split()
    clean_word_list=[ps.stem(word).lower() for word in word_list if word not in stop]
    return " ".join(clean_word_list)

def create_labels(ham_list,spam_list):
    labels=len(ham_list)*[0]+len(spam_list)*[1]
    return labels

def production(mail):
    mail=remove_StopWords(mail)
    mail_vectorized=vectorizer.transform([mail]).toarray()
    pred_dict={0:"ham",1:"spam"}
    return pred_dict[modelG.predict(mail_vectorized)[0]]