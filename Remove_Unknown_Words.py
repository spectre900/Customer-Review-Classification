import pickle as pkl
import numpy as np

file=open('word_to_num.pkl','rb')
hm=pkl.load(file)

def clean(string):
    string=list(string)
    for i in range(len(string)):
        if ord(string[i])<97 or ord(string[i])>122:
            string[i]=0
    for i in range(string.count(0)):
        string.remove(0)
    return ''.join(string)


df=np.load('reviews.npy')

for i in range(2500):
    sent=df[i]
    words=sent.split()
    mod_word=[]
    for w in words:
        w=w.lower()
        w=clean(w)
        if w not in hm.keys():
            mod_word.append('ukn')
        else:
            mod_word.append(w)
    mod_sent=' '.join(mod_word)
    df[i]=mod_sent

np.save('reviews_cleaned.npy',df)
