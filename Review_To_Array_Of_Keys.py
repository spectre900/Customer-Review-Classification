import numpy as np
import pickle as pkl

m=2500
rev=np.load('reviews_cleaned.npy')
rate=np.load('ratings.npy')

l=[]

max_len=100
for i in range(m):
    sent=rev[i]
    words=sent.split()
    l.append(words)

word_to_num=pkl.load(open('word_to_num.pkl','rb'))

for i in range(m):
    for j in range(len(l[i])):
        l[i][j]=int(word_to_num[l[i][j]])

for i in range(m):
    for j in range(len(l[i]),max_len):
        l[i].append(0)

rev=np.array(l)

train_rev=rev[:2000]
train_rate=rate[:2000]
test_rev=rev[2000:2500]
test_rate=rate[2000:2500]

np.save('train_review.npy',train_rev)
np.save('train_rating.npy',train_rate)
np.save('test_review.npy',test_rev)
np.save('test_rating.npy',test_rate)
