from collections import Counter,defaultdict
from math import log
import numpy as np
import sys
import os

# load data
spanish_file = "corpus.es" #source, the foreign language file
english_file = "corpus.en" #target, the english file
translation_probabilities = "ibm1_tprob.txt"

output_trans = ["ibm2_trans1.txt","ibm2_trans2.txt","ibm2_trans3.txt","ibm2_trans4.txt","ibm2_trans5.txt"]
output_distor = ["ibm2_distor1.txt","ibm2_distor2.txt","ibm2_distor3.txt","ibm2_distor4.txt","ibm2_distor5.txt"]

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
	with open(output_trans[iterations],'w') as ff:
		for t in translation:
			if translation[t]>0:
				ff.write('%s %s %f\n' % (t[0], t[1], translation[t]))

	with open(output_distor[iterations],'w') as ff:
		for d in distortion:
			ff.write('%d %d %d %d %f\n' % (d[0], d[1], d[2], d[3], distortion[d])) #k, i, l, m

	iterations += 1 


# load data
spanish_file = "dev.es"  #source, the foreign language file
english_file = "dev.en" ##target, the english file
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

for iterations in range(5):
	
	# load translation probabilities
	translation = {}
	with open(output_trans[iterations],'r') as ff:
		for t in ff.readlines():
			t = t.strip().lower().split()
			translation[(t[0], t[1])] = float(t[2]) #translation(e,f)

	# load distortion probabilities
	distortion = {}
	with open(output_distor[iterations],'r') as ff:
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


	os.system("python eval_alignment.py dev.key 2_dev.out")
