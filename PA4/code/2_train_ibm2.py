from math import log
from collections import defaultdict

english_file = "corpus.en" #source, the english file
spanish_file = "corpus.es" #target, the foreign language file
translation_probabilities = "ibm1_tprob.txt"

output_trans = "ibm2_trans.txt"
output_distor = "ibm2_distor.txt"

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
eng_count = len(eng_vocab)

# Load the translation probabilities from IBM model 1
translation = {}
with open(translation_probabilities,'r') as ff:
    for t in ff.readlines():
        t = t.strip().lower().split()
        translation[(t[0], t[1])] = float(t[2]) #translation(e,f)


# EM steps  
distortion = {}
iterations = 0

while iterations < 5:    
    iterations += 1     

    ## E-STEP
    likelihood = 0
    count_t = defaultdict(float)
    total_t = defaultdict(float)     

    count_a = defaultdict(float)
    total_a = defaultdict(float)

    for pair in pairs:
        m = len(pair[0]) #fr, source
        l = len(pair[1]) #eng

        z = defaultdict(float) #normalize constant
        for i, f in enumerate(pair[0]):
            for k, e in enumerate(pair[1]): 
                if (k, i, l, m) not in distortion:
                    distortion[(k, i, l, m)] = 1.0/(l+1) #Initialize
                try: 
                    z[f] += distortion[(k, i, l, m)] * translation[(e, f)]
                except KeyError:
                    pass

        for i, f in enumerate(pair[0]):
            for k, e in enumerate(pair[1]):
                try: 
                    if z[f] != 0:
                        p = distortion[(k, i, l, m)] * translation[(e, f)] / z[f]
                    else:
                        p = 0
                except KeyError: 
                    p = 0

                count_t[(e, f)] += p 
                total_t[e] += p
                count_a[(k, i, l, m)] += p
                total_a[(i, l, m)] += p

            if z[f] != 0:
                likelihood += log(z[f])
        
    ## M-STEP
    for e in eng:
        for f in fr:
            if total_t[e] != 0:
                translation[(e, f)] = count_t[(e, f)] / total_t[e]

    for pair in pairs:
        m = len(pair[0])
        l = len(pair[1])
        for i in range(m):
            for k in range(l):
                if total_a[(i, l, m)] != 0:
                    distortion[(k, i, l, m)] = count_a[(k, i, l, m)] / total_a[(i, l, m)]

    print('log-likelihood',likelihood)


# Save
with open(output_trans,'w') as ff:
    for t in translation:
        if translation[t]>0:
            ff.write('%s %s %f\n' % (t[0], t[1], translation[t]))

with open(output_distor,'w') as ff:
    for d in distortion:
        ff.write('%d %d %d %d %f\n' % (d[0], d[1], d[2], d[3], distortion[d])) #k, i, l, m


