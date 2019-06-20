#! /usr/bin/python
from collections import defaultdict
import porter
p = porter.PorterStemmer()
import os

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
			c=float(w[0])
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


	emiss={y:{w:0 for w in vocab} for y in yxdict.keys()}
	for y in yxdict.keys():
		for x,c in yxdict[y]:
			newc = c/uni_counts[y]
			emiss[y][x]=newc

	states=list(uni_counts.keys())
	trans={x:{k:0 for k in states+['STOP']} for x in bi_counts.keys()}
	for tri,c in tri_counts.items():
		bi='_'.join(tri.split('_')[:-1])
		t=tri.split('_')[-1]
		trans[bi][t]=c/bi_counts[bi]

	return emiss,trans,states


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

def getnewFile_p(nfname,oricounts,oritrain):
    vocab=[]
    nws=defaultdict(int)
    tri_nws=defaultdict(list) #log
    with open(oricounts,'r') as f:
        for line in f.readlines():
            w=line.strip().split()
            if len(w)==0 or w[1]!='WORDTAG':
                continue
            if int(w[0])>=5:
                vocab.append(w[3])
            else:
                nw=p.stem(w[3].lower())
                nws[nw]+=int(w[0])
                tri_nws[nw].append(w[3])

    with open('2tmp1.txt','w') as f:
            for k,v in tri_nws.items():
                f.write(k+'\n'+'; '.join(v)) #log for discussion
                f.write('\n')

    with open(oritrain,'r') as f:
        lines=[line.strip() for line in f.readlines()]

    groups=[nw for nw,c in nws.items() if c>=5]
    print('number of groups: ',len(groups)+1)

    with open(nfname,'w') as f:
        for line in lines:
            w=line.split()
            if len(w)==2 and w[0] not in vocab:
                nw=p.stem(w[0].lower())
                if nw in groups:
                    f.write(nw+'_RARE_ '+w[1]+'\n')
                else:
                    f.write('_RARE_ '+w[1]+'\n')
            else:
                f.write(line+'\n')

def getnewFile_n(nfname,oricounts,oritrain,n=1):
	vocab=[]
	nws=defaultdict(int)
	tri_nws=defaultdict(list) #log
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
				if n==3:
					tri_nws[nw].append(w[3]) #log
	if n==3:
		with open('3tmp2.txt','w') as f:
			for k,v in tri_nws.items():
				f.write(k+'\n'+'; '.join(v)) #log for discussion
				f.write('\n')

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
				if ngram==-1:
					nw = p.stem(sent[0].lower())+'_RARE_'
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
					if ngram==-1:
						nw = p.stem(obser_val.lower())+'_RARE_'
					else:
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
	if len(sent)>0:
		sent.append('')
		sentences.append(sent)

	with open(testout,'w') as f:
		for sent in sentences:
			tags=viterbi(sent,emiss,trans,states)

			for i,w in enumerate(sent):
				f.write(w)
				if w!='':
					f.write(' '+tags[i+1])
				f.write('\n')


if __name__ == "__main__":
	# 1
	print('step 1')
	getnewFile('new_gene.train','gene.counts','gene.train')
	os.system("python count_freqs.py new_gene.train > new_gene.counts")

	# 2
	print('step 2')
	emiss,trans,states=emission('new_gene.counts')
	# print(emiss)
	print(emiss['O']['free'],emiss['I-GENE']['free'])
	# exit()
	hmmTagger('gene.dev','gene_dev.p1.out',emiss,trans,states)
	os.system("python eval_gene_tagger.py gene.key gene_dev.p1.out")

	#3
	print('step 3')
	getnewFile_p('new_gene.train','gene.counts','gene.train')
	os.system("python count_freqs.py new_gene.train > new_gene.counts")
	emiss,trans,states=emission('new_gene.counts')
	improvedTagger('gene.dev','gene_dev.p1.out',emiss,trans,states,ngram=-1)
	os.system("python eval_gene_tagger.py gene.key gene_dev.p1.out")

	# # getunlabel_train('gene.train','unlabel_gene.train')
	improvedTagger('unlabel_gene.train','gene_train.p1.out',emiss,trans,states,ngram=-1)
	os.system("python eval_gene_tagger.py gene.train gene_train.p1.out")
	
	for i in range(1,8):
		print(i)
		getnewFile_n('new_gene.train','gene.counts','gene.train',n=i)
		os.system("python count_freqs.py new_gene.train > new_gene.counts")
		emiss,trans,states=emission('new_gene.counts')
		improvedTagger('gene.dev','gene_dev.p1.out',emiss,trans,states,ngram=i)
		os.system("python eval_gene_tagger.py gene.key gene_dev.p1.out")

	getnewFile_n('new_gene.train','gene.counts','gene.train',n=2)
	os.system("python count_freqs.py new_gene.train > new_gene.counts")
	emiss,trans,states=emission('new_gene.counts')
	improvedTagger('unlabel_gene.train','gene_train.p1.out',emiss,trans,states,ngram=2)
	os.system("python eval_gene_tagger.py gene.train gene_train.p1.out")

# python 3HMM.py