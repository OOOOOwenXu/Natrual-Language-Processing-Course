***[CSE 256 SP 19: Programming Assignment 3 : Sequence Tagging ]***

This code is developed based on the provided python scripts. Thus following the same structure. In brief, we add five scripts into the orginal fold, including 2baseline.py, 3HMM.py, 4extension.py, 4extension1.py, porter.py.

The scripts should be run:
***part 1***
> python 2baseline.py
steps:
1. using (count_freqs.py), input gene.train, output gene.counts
2. input gene.train, output new_gene.train: replace _RARE_
3. using (count_freqs.py), input new_gene.train, output new_gene.counts
4. input new_gene.counts, gene.dev, output gene_dev.p1.out
5. using (eval_gene_tagger.py), input gene_dev.p1.out, obtain result
6. improve tagger with strategy 1
7. improve tagger with strategy 2	
- both the improved tagger are evaluate follows similar steps as above:
	input gene.train, output a few new_gene.train
	using (count_freqs.py), input new_gene.train, output new_gene.counts
	input new_gene.counts, gene.dev, output gene_dev.p1.out
	using (eval_gene_tagger.py), input gene_dev.p1.out, obtain result
	get the unlabeled train file, evaluate on the train data

***part 2***
> python 3HMM.py
steps:
1. 
input gene.train, output new_gene.train: replace _RARE_
using (count_freqs.py), input new_gene.train, output new_gene.counts
2. HMM tagger
input new_gene.counts, gene.dev, output gene_dev.p1.out
using (eval_gene_tagger.py), input gene_dev.p1.out, obtain result
3. improve HMM tagger: the same two strategy

***part 3***
> python 4extension.py
smoothing the parameters; using Good-Turing method
steps: contain two steps, same with 3HMM.py

> python 4extension1.py
using 4-gram HMM
steps: contain two steps, same with 3HMM.py

****[extral file]****
porter.py is a public script we find on the internet for stemming