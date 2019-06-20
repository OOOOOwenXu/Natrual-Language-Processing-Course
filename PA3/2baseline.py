#! /usr/bin/python
from collections import defaultdict
import os
import porter
p = porter.PorterStemmer()


def emission(fname):
    with open(fname,'r') as f:
        lines=[line.strip() for line in f.readlines()]
    counts = defaultdict(float)
    xydict = defaultdict(list)
    for line in lines:
        w = line.split()
        if len(w)==0 or w[1]!='WORDTAG':
            continue
        y=w[2]
        x=w[3]
        c=float(w[0])
        counts[y]+=c
        xydict[x].append((y,c))
    # print(len(xydict))

    emiss=defaultdict(list)
    for key in xydict.keys():
        for y,c in xydict[key]:
            newc = c/counts[y]
            emiss[key].append((y,newc))

    # print(len(emiss))
    return emiss

def getnewFile(nfname,oricounts,oritrain):
    vocab=[]
    with open(oricounts,'r') as f:
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

def simpleTagger(testFile,emiss,testout):
    with open(testFile,'r') as f:
        lines=[line.strip() for line in f.readlines()]

    propTag=defaultdict(str)
    for w in emiss.keys():
        ylist=emiss[w]
        ylist = sorted(ylist,key=lambda x:x[1], reverse=True)
        propTag[w]=ylist[0][0]

    with open(testout,'w') as f:
        for w in lines:
            if len(w)==0:
                f.write(w+'\n')
            else:
                if w in propTag:
                    f.write(w+' '+propTag[w]+'\n')
                else:
                    f.write(w+' '+propTag['_RARE_']+'\n')


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

def improvedTagger1(testFile,emiss,testout):
    with open(testFile,'r') as f:
        lines=[line.strip() for line in f.readlines()]

    propTag=defaultdict(str)
    for w in emiss.keys():
        ylist=emiss[w]
        ylist = sorted(ylist,key=lambda x:x[1], reverse=True)
        propTag[w]=ylist[0][0]

    with open(testout,'w') as f:
        for w in lines:
            if len(w)==0:
                f.write(w+'\n')
            else:
                if w in propTag:
                    f.write(w+' '+propTag[w]+'\n')
                else:
                    nw=p.stem(w.lower())+'_RARE_'
                    if nw in propTag:
                        f.write(w+' '+propTag[nw]+'\n')
                    else:
                        f.write(w+' '+propTag['_RARE_']+'\n')

def getnewFile_n(nfname,oricounts,oritrain,n=1):
    vocab=[]
    nws=defaultdict(int)
    tri_nws=defaultdict(list)
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
        with open('2tmp2.txt','w') as f:
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

def improvedTagger2(testFile,emiss,testout,n=1):
    with open(testFile,'r') as f:
        lines=[line.strip() for line in f.readlines()]

    propTag=defaultdict(str)
    for w in emiss.keys():
        ylist=emiss[w]
        ylist = sorted(ylist,key=lambda x:x[1], reverse=True)
        propTag[w]=ylist[0][0]

    with open(testout,'w') as f:
        for w in lines:
            if len(w)==0:
                f.write(w+'\n')
            else:
                if w in propTag:
                    f.write(w+' '+propTag[w]+'\n')
                else:
                    nw=w[:n]+'_RARE_'
                    if nw in propTag:
                        f.write(w+' '+propTag[nw]+'\n')
                    else:
                        f.write(w+' '+propTag['_RARE_']+'\n')

def getunlabel_train(ftrain,unftrain):
    with open(ftrain,'r') as f:
        lines=[line.strip() for line in f.readlines()]
    with open(unftrain,'w') as f:
        for line in lines:
            if line!='':
                f.write(line.split()[0]+'\n')
            else:
                f.write('\n')

if __name__ == "__main__":
    # 1
    print('step 1')
    os.system("python count_freqs.py gene.train > gene.counts")

    ########2
    print('step 2')
    getnewFile('new_gene.train','gene.counts','gene.train')

    # 3
    print('step 3')
    os.system("python count_freqs.py new_gene.train > new_gene.counts")

    #######4
    print('step 4')
    emiss = emission('new_gene.counts')
    simpleTagger('gene.dev',emiss,'gene_dev.p1.out')

    # 5
    print('step 5')
    os.system("python eval_gene_tagger.py gene.key gene_dev.p1.out")

    # 6. improve tagger: stemming
    print('step 6')
    
    getnewFile_p('new_gene.train','gene.counts','gene.train')
    os.system("python count_freqs.py new_gene.train > new_gene.counts")
    emiss = emission('new_gene.counts')
    
    improvedTagger1('gene.dev',emiss,'gene_dev.p1.out')
    os.system("python eval_gene_tagger.py gene.key gene_dev.p1.out")

    getunlabel_train('gene.train','unlabel_gene.train')
    improvedTagger1('unlabel_gene.train',emiss,'gene_train.p1.out')
    os.system("python eval_gene_tagger.py gene.train gene_train.p1.out")

    # 7. improve tagger: shape
    print('step 7')
    for i in range(1,8):
        print(i)
        getnewFile_n('new_gene.train','gene.counts','gene.train',n=i)
        os.system("python count_freqs.py new_gene.train > new_gene.counts")
        emiss = emission('new_gene.counts')
        improvedTagger2('gene.dev',emiss,'gene_dev.p1.out',n=i)
        os.system("python eval_gene_tagger.py gene.key gene_dev.p1.out")

    getnewFile_n('new_gene.train','gene.counts','gene.train',n=3)
    os.system("python count_freqs.py new_gene.train > new_gene.counts")
    emiss = emission('new_gene.counts')   
    improvedTagger2('unlabel_gene.train',emiss,'gene_train.p1.out',n=3)
    os.system("python eval_gene_tagger.py gene.train gene_train.p1.out")
        


# python 2baseline.py