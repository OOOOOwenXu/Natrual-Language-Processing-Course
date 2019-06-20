from collections import defaultdict
import numpy as np
import math
import os

# load data
output_file1 = "3_1_dev.out"
output_file2="3_2_dev.out"

if not os.path.exists(output_file1):
    print("predicting")
    spanish_file = "dev.es"  #source, the foreign language file
    english_file = "dev.en" ##target, the english file
    translation_probabilities = "ibm2_trans.txt"
    distortion_file = "ibm2_distor.txt"
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
    with open(output_file1,'w') as ff:
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


    print("reverse predicting")
    # load data
    spanish_file = "dev.en"  #source, the foreign language file
    english_file = "dev.es" ##target, the english file
    translation_probabilities = "3_ibm2_trans.txt"
    distortion_file = "3_ibm2_distor.txt"
    

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
    with open(output_file2,'w') as ff:
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

                ff.write("%i %i %i\n" % (sent_i+1,i+1,v+1))

print("growing")
output_file = "3_dev.out"
all_e=defaultdict(list) #from english to spanish
all_f=defaultdict(list) #from spanish to english :: task 2

with open(output_file1,'r') as ff:
    for line in ff.readlines():
        line=line.strip().split()
        all_f[int(line[0])].append((int(line[1]),int(line[2])))
with open(output_file2,'r') as ff:
    for line in ff.readlines():
        line=line.strip().split()
        all_e[int(line[0])].append((int(line[1]),int(line[2])))

union=defaultdict(list) #
for k in all_f.keys():
    s=set(all_f[k]).union(set(all_e[k]))
    union[k]=list(s)

grow=defaultdict(list) #
for k in all_f.keys():
    s=set(all_f[k]).intersection(set(all_e[k]))
    grow[k]=list(s)

def getdistance(pair,values):
    dis=[]
    for v in values:
        d=(pair[0]-v[0])**2+(pair[1]-v[1])**2
        dis.append(d)
    return min(dis)

for k in grow.keys():
    if len(grow[k])==0:
        continue
    for v in union[k]:
        fdone=set((i[1] for i in grow[k]))
        edone=set([i[0] for i in grow[k]])
        if v[1] not in fdone or v[0] not in edone:
            if getdistance(v,grow[k])<10:
                grow[k].append(v)

    d1=[i[0] for i in all_e[k]]
    d2=[i[1] for i in all_f[k]]
    d=abs(max(d1)-max(d2))+3
    grow[k]=[v for v in grow[k] if abs(v[0]-v[1])<d]

with open(output_file,'w') as ff:
    for k in grow.keys():
        for v in grow[k]:
            ff.write("%s %s %s\n" % (k,v[0],v[1]))