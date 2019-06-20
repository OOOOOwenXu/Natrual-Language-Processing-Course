from collections import Counter,defaultdict
from math import log
import numpy as np
import sys
import os

# load data
spanish_file = "corpus.es" #source, the foreign language file
english_file = "corpus.en" #target, the english file
# translation_probabilities = "ibm1_tprob.txt"
translation_probabilities=["ibm1_tprob1.txt","ibm1_tprob2.txt","ibm1_tprob3.txt","ibm1_tprob4.txt","ibm1_tprob5.txt"]

pairs=[]
fr_vocab = set()
with open(spanish_file,'r') as ff:
	for s in ff.readlines():
		s = s.strip().lower().split()
		pairs.append([s])
		for w in s:
			fr_vocab.add(w)
fr = {w:i for i,w in enumerate(fr_vocab)}
fr_count = len(fr_vocab)

eng_vocab = set()
count=0
with open(english_file,'r') as ff:
	for s in ff.readlines():
		s = s.strip().lower().split()
		s.append('NULL') #the special English word NULL.
		pairs[count].append(s)
		count+=1
		for w in s:
			eng_vocab.add(w)
eng = {w:i for i,w in enumerate(eng_vocab)}
eng_count = len(eng_vocab)

# initialization step,
def count_ne(ew,pairs):
	voc=[]
	for pair in pairs:
		f = pair[0]
		e = pair[1]
		if ew in e:
			voc.extend(f)
	return len(set(voc))

translation = np.ones([eng_count,fr_count])/fr_count
for i,w in enumerate(eng_vocab):
	if w!='NULL':
		translation[i]=np.ones(fr_count)/count_ne(w,pairs)


# EM steps  
iterations = 0
while iterations < 5:           
	likelihood = 0

	## E-STEP
	count_f_e = np.zeros([eng_count,fr_count])
	total_e = np.zeros(eng_count)   
	z_ki = {}

	for k,pair in enumerate(pairs):

		z = defaultdict(float) #normalize constant
		for i, fw in enumerate(pair[0]):
			indf = fr[fw]
			for j, ew in enumerate(pair[1]): 
				inde = eng[ew]
				z[fw] += translation[inde,indf]
		
		for i, fw in enumerate(pair[0]):
			indf = fr[fw]
			for j, ew in enumerate(pair[1]):
				inde = eng[ew]
				try: 
					if z[fw] != 0:
						p = translation[inde,indf] / z[fw]
					else:
						p = 0
				except KeyError: 
					p = 0

				count_f_e[inde,indf] += p 
				total_e[inde] += p
			   
			if z[fw] != 0:
				likelihood += log(z[fw])    
		
	## M-STEP
	translation = (count_f_e.T / total_e).T
	print('log-likelihood',likelihood)


	# Save
	with open(translation_probabilities[iterations],'w') as ff:
		for e in eng:
			for f in fr:
				pr = translation[eng[e], fr[f]]
				if pr > 0:
					ff.write('%s %s %f\n' % (e,f, pr))
	iterations += 1 

# load data
spanish_file = "dev.es"  #source, the foreign language file
english_file = "dev.en" ##target, the english file
output_file = "1_dev.out"

for iterations in range(5):
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
	with open(translation_probabilities[iterations],'r') as ff:
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

	os.system("python eval_alignment.py dev.key 1_dev.out")
