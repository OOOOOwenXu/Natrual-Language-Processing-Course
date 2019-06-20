#! /usr/bin/python
from collections import defaultdict
import pdb
import os

def getnewFile(nfname,oriuni_counts,oritrain):
	vocab=[]
	with open(oriuni_counts,'r') as f:
		for line in f.readlines():
			w=line.strip().split()
			if len(w)==0 or w[1]!='WORDTAG':
				continue
			if int(w[0])>=5:
				vocab.append(w[3])

	with open(oritrain,'r') as f:
		lines=[line.strip() for line in f.readlines()]

	with open(nfname,'w') as f:
		for line in lines:
			w=line.split()
			if len(w)==2 and w[0] not in vocab:
				f.write('_RARE_ '+w[1]+'\n')
			else:
				f.write(line+'\n')

def good_turing(counts):
    N = sum(counts)  
    prob = [0] * len(counts)
    
    if N == 0:
        return prob
        
    Nr = [0] * (max(counts) + 1)
    for r in counts:
        Nr[r] += 1
        
    smooth_boundary = min(len(Nr)-1, 5) 
    for r in range(smooth_boundary):
        if Nr[r] != 0 and Nr[r+1] != 0:
            Nr[r] = (r+1) * Nr[r+1] / Nr[r]
        else:
            Nr[r] = r
    for r in range(smooth_boundary, len(Nr)):
        Nr[r] = r
        
    for i in range(len(counts)):
        prob[i] = Nr[counts[i]]
    total = sum(prob)
    return [p/total for p in prob] 

def emission(countFile):
	with open(countFile,'r') as f:
		lines=[line.strip() for line in f.readlines()]
	yxdict = defaultdict(list)
	vocab=set()
	uni_counts = defaultdict(float)
	bi_counts=defaultdict(float)
	tri_counts=defaultdict(float)
	for line in lines:
		w = line.split()
		if len(w)==0:
			continue
		if w[1]=='WORDTAG':
			y=w[2]
			x=w[3]
			c=int(w[0])
			yxdict[y].append((x,c))
			vocab.add(x)
		elif w[1]=='1-GRAM':
			uni=w[2]
			uni_counts[uni]=float(w[0])
		elif w[1]=='2-GRAM':
			bi='_'.join(w[2:])
			bi_counts[bi]=float(w[0])
		elif w[1]=='3-GRAM':
			tri='_'.join(w[2:])
			tri_counts[tri]=float(w[0])

	for y in yxdict.keys():
		tmp={w:0 for w in vocab}
		for x,c in yxdict[y]:
			tmp[x]+=c
		yxdict[y]=tmp
	emiss={y:{w:0 for w in vocab} for y in yxdict.keys()}
	for y in emiss.keys():
		counts=[]
		words=[]
		for x,c in emiss[y].items():
			counts.append(yxdict[y][x])
			words.append(x)
		probs=good_turing(counts)
		for x,p in zip(words,probs):
			emiss[y][x]=p

	states=list(uni_counts.keys())
	trans={x:{k:0 for k in states+['STOP']} for x in bi_counts.keys()}
	for prev,value in trans.items():
		for tag,c in value.items():
			tri=prev+'_'+tag
			trans[prev][tag]=tri_counts[tri]/bi_counts[prev]
	# print(trans)
	return emiss,trans,states
				
def hmmTagger(testFile,testout,emiss,trans,states):
	def get_max_from_dict(delta):
		delta_list=sorted(delta.items(),key=lambda x:x[1], reverse=True)
		max_key, max_val = delta_list[0]
		return max_key, max_val

	def viterbi(sent,emiss,trans,states):
		delta_i= {k:{} for k in range(len(sent))}
		trace_i= {k:{} for k in range(len(sent))}
		

		n=2
		for state in states:
			if sent[0] in emiss[state]:
				delta_i[0][state] = trans['_'.join(['*']*n)][state] * emiss[state][sent[0]]
			else:
				delta_i[0][state] = trans['_'.join(['*']*n)][state] * emiss[state]['_RARE_']
			trace_i[0][state]= '*'
		
		for i in range(1, len(sent)-1):
			obser_val = sent[i]
			for state in states:
				delta_j={}
				for state_last in states:
					prev=[trace_i[i-1][state_last]]+[state_last]
					prev='_'.join(prev)
					delta_j[state_last]=delta_i[i-1][state_last]*trans[prev][state]
				key, val = get_max_from_dict(delta_j)
				trace_i[i][state]=key
				if obser_val not in emiss[state]:
					delta_i[i][state]=val * emiss[state]['_RARE_']
				else:
					delta_i[i][state]=val * emiss[state][obser_val]

		state='STOP'
		i=len(sent)-1
		delta_j={}
		for state_last in states:
			prev=[trace_i[i-1][state_last]]+[state_last]
			prev='_'.join(prev)
			delta_j[state_last]=delta_i[i-1][state_last]*trans[prev][state]
		key, val = get_max_from_dict(delta_j)
		trace_i[i][state]=key

		path=[key] #backward get path
		for i in range(1,len(sent)):
			b=trace_i[len(sent)-1-i][key]
			path.append(b)
			key=b
		path=path[::-1] #reverse

		return path

	sentences=[]
	with open(testFile,'r') as f:
		sent=[] #observe
		for line in f.readlines():
			line=line.strip()
			if line=='':
				sent.append('')
				sentences.append(sent)
				sent=[]
			else:
				sent.append(line)

	with open(testout,'w') as f:
		for sent in sentences:
			tags=viterbi(sent,emiss,trans,states)

			for i,w in enumerate(sent):
				f.write(w)
				if w!='':
					f.write(' '+tags[i+1])
				f.write('\n')

def getnewFile_n(nfname,oricounts,oritrain,n=1):
	vocab=[]
	nws=defaultdict(int)
	with open(oricounts,'r') as f:
		for line in f.readlines():
			w=line.strip().split()
			if len(w)==0 or w[1]!='WORDTAG':
				continue
			if int(w[0])>=5:
				vocab.append(w[3])
			else:
				nw=w[3][:n]
				nws[nw]+=int(w[0])

	with open(oritrain,'r') as f:
		lines=[line.strip() for line in f.readlines()]

	groups=[nw for nw,c in nws.items() if c>=5]
	print('number of groups: ',len(groups)+1)

	with open(nfname,'w') as f:
		for line in lines:
			w=line.split()
			if len(w)==2 and w[0] not in vocab:
				nw=w[0][:n]
				if nw in groups:
					f.write(nw+'_RARE_ '+w[1]+'\n')
				else:
					f.write('_RARE_ '+w[1]+'\n')
			else:
				f.write(line+'\n')

def improvedTagger(testFile,testout,emiss,trans,states,ngram=1):
	def get_max_from_dict(delta):
		delta_list=sorted(delta.items(),key=lambda x:x[1], reverse=True)
		max_key, max_val = delta_list[0]
		return max_key, max_val

	def viterbi(sent,emiss,trans,states):
		delta_i= {k:{} for k in range(len(sent))}
		trace_i= {k:{} for k in range(len(sent))}
		

		n=2 #trigram HMM, depend on last two tags
		for state in states:
			if sent[0] in emiss[state]:
				delta_i[0][state] = trans['_'.join(['*']*n)][state] * emiss[state][sent[0]]
			else:
				nw=sent[0][:ngram]+'_RARE_'
				if nw in emiss[state]:
					delta_i[0][state] = trans['_'.join(['*']*n)][state] * emiss[state][nw]
				else:
					delta_i[0][state] = trans['_'.join(['*']*n)][state] * emiss[state]['_RARE_']
			trace_i[0][state]= '*'
		
		for i in range(1, len(sent)-1):
			obser_val = sent[i]
			for state in states:
				delta_j={}
				for state_last in states:
					prev=[trace_i[i-1][state_last]]+[state_last]
					prev='_'.join(prev)
					delta_j[state_last]=delta_i[i-1][state_last]*trans[prev][state]
				key, val = get_max_from_dict(delta_j)
				trace_i[i][state]=key
				if obser_val not in emiss[state]:
					nw=obser_val[:ngram]+'_RARE_'
					if nw in emiss[state]:
						delta_i[i][state]=val * emiss[state][nw]
					else:
						delta_i[i][state]=val * emiss[state]['_RARE_']
				else:
					delta_i[i][state]=val * emiss[state][obser_val]

		state='STOP'
		i=len(sent)-1
		delta_j={}
		for state_last in states:
			prev=[trace_i[i-1][state_last]]+[state_last]
			prev='_'.join(prev)
			delta_j[state_last]=delta_i[i-1][state_last]*trans[prev][state]
		key, val = get_max_from_dict(delta_j)
		trace_i[i][state]=key

		path=[key] #backward get path
		for i in range(1,len(sent)):
			b=trace_i[len(sent)-1-i][key]
			path.append(b)
			key=b
		path=path[::-1] #reverse

		return path

	sentences=[]
	with open(testFile,'r') as f:
		sent=[] #observe
		for line in f.readlines():
			line=line.strip()
			if line=='':
				sent.append('')
				sentences.append(sent)
				sent=[]
			else:
				sent.append(line)

	with open(testout,'w') as f:
		for sent in sentences:
			tags=viterbi(sent,emiss,trans,states)

			for i,w in enumerate(sent):
				f.write(w)
				if w!='':
					f.write(' '+tags[i+1])
				f.write('\n')

if __name__ == "__main__":
	# # 1
	print('step 1')
	getnewFile('new_gene.train','gene.counts','gene.train')
	os.system("python count_freqs.py new_gene.train > new_gene.counts")

	# 2
	print('step 2')
	emiss,trans,states=emission('new_gene.counts')
	hmmTagger('gene.dev','gene_dev.p1.out',emiss,trans,states)
	os.system("python eval_gene_tagger.py gene.key gene_dev.p1.out")

	# # 3
	# print('step 3')
	# for i in range(1,9):
	# 	print(i)
	# 	getnewFile_n('new_gene.train','gene.counts','gene.train',n=i)
	# 	os.system("python count_freqs.py new_gene.train > new_gene.counts")
	# 	emiss,trans,states=emission('new_gene.counts')
	# 	improvedTagger('gene.dev','gene_dev.p1.out',emiss,trans,states,ngram=i)
	# 	os.system("python eval_gene_tagger.py gene.key gene_dev.p1.out")

