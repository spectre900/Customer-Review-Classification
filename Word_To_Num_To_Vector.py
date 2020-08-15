import numpy as np
import pickle as pkl

file=open('glove.6B.200d.txt','r')

hm={}
l=[]
for i in range(200):
    l.append(0.0)
    
hm[0]=np.array(l)

count=1
for lines in file:
    values=lines.split()
    word=values[0]
    vector=list(map(float,values[1:]))
    hm[count]=np.array(vector)
    count+=1
    if(count%50000==0):
        print(count,' words done out of 400000')

f= open('num_to_vec.pkl','wb')
pkl.dump(hm,f)
f.close()
file.close()


file=open('glove200.txt','r')
hm={}
hm['ukn']=0

count=1
for lines in file:
    values=lines.split()
    word=values[0]
    hm[word]=count
    count+=1
    if(count%50000==0):
        print(count,' words done out of 400000')

f= open('word_to_num.pkl','wb')
pkl.dump(hm,f)
f.close()

file.close()

