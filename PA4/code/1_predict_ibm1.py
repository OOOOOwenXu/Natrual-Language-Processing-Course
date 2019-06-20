import sys
import numpy as np
from collections import Counter

# load data
spanish_file = "dev.es"  #source, the foreign language file
english_file = "dev.en" ##target, the english file
translation_probabilities = "ibm1_tprob.txt"
output_file = "1_dev.out"

pairs=[]
fr_vocab = set()
with open(spanish_file,'r') as f:
    for s in f.readlines():
        s = s.strip().lower().split()
        pairs.append([s])
        for w in s:
            fr_vocab.add(w)
fr = {w:i for i,w in enumerate(fr_vocab)}
fr_count = len(fr_vocab)

eng_vocab = set()
count=0
with open(english_file,'r') as f:
    for s in f.readlines():
        s = s.strip().lower().split()
        s.append('NULL') #the special English word NULL.
        pairs[count].append(s)
        count+=1
        for w in s:
            eng_vocab.add(w)
eng = {w:i for i,w in enumerate(eng_vocab)}
eng_reverse = {i:w for i,w in enumerate(eng_vocab)}
eng_count = len(eng_vocab)


# Load the translation probabilities
translation = np.zeros([eng_count,fr_count])
with open(translation_probabilities,'r') as ff:
    for t in ff.readlines():
        t = t.strip().lower().split()
        if t[0] in eng and t[1] in fr:
            translation[eng[t[0]],fr[t[1]]]=float(t[2])

#predicting
with open(output_file,'w') as ff:
    for i,pair in enumerate(pairs):
        f=pair[0]
        e=pair[1]
        for j,fw in enumerate(f):
            indfr = fr[fw]
            scores=[]
            for k,ew in enumerate(e):
                indeng = eng[ew]
                scores.append(translation[indeng,indfr])
            v = np.argmax(scores) #translation word index
            
            ff.write("%i %i %i\n" % (i+1,v+1,j+1))

# python eval_alignment.py dev.key dev.out