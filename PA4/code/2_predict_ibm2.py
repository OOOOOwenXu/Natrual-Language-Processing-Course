import sys
from collections import defaultdict
import numpy as np

# load data
spanish_file = "dev.es"  #source, the foreign language file
english_file = "dev.en" ##target, the english file
translation_probabilities = "ibm2_trans.txt"
distortion_file = "ibm2_distor.txt"
output_file = "2_dev.out"

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


# load translation probabilities
translation = {}
with open(translation_probabilities,'r') as ff:
    for t in ff.readlines():
        t = t.strip().lower().split()
        translation[(t[0], t[1])] = float(t[2]) #translation(e,f)

# load distortion probabilities
distortion = {}
with open(distortion_file,'r') as ff:
    for t in ff.readlines():
        t = t.strip().lower().split()
        # distortion[(k, i, l, m)]
        distortion[(int(t[0]), int(t[1]), int(t[2]), int(t[3]))] = float(t[4])

#predicting
with open(output_file,'w') as ff:
    for sent_i,pair in enumerate(pairs):
        f=pair[0]
        e=pair[1]

        m = len(pair[0])
        l = len(pair[1])

        for i, fw in enumerate(f):
            scores=[]
            for k,ew in enumerate(e):
                try: 
                    p = distortion[(k, i, l, m)] * translation[(ew, fw)] 
                except KeyError:
                    p = 0
                scores.append(p)
                
            v = np.argmax(scores) #translation word index

            ff.write("%i %i %i\n" % (sent_i+1,v+1,i+1))

# python eval_alignment.py dev.key dev.out