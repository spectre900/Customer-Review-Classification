import numpy as np
import pandas as pd

file1=open('amazon.txt','r')
file2=open('yelp.txt','r')
file3=open('imdb.txt','r')

rev=[]
rating=[]

for lines in file1:
    lines=lines.split('\t')
    rev.append(lines[0])
    rating.append(int(lines[1]))

for lines in file2:
    lines=lines.split('\t')
    rev.append(lines[0])
    rating.append(int(lines[1]))

for lines in file3:
    lines=lines.split('\t')
    rev.append(lines[0])
    rating.append(int(lines[1]))

l=list(zip(rev,rating))
np.random.shuffle(l)
a,b=zip(*l)

rev=np.array(a)
rating=np.array(b)

np.save('reviews.npy',rev)
np.save('ratings.npy',rating)
