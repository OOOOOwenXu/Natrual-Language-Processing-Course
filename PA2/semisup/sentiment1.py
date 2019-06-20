#!/bin/python
import nltk
from nltk.tokenize import WordPunctTokenizer

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

import tarfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
import copy
import pdb


def filterPOS(data):
	tmp_data=[]
	goodPOS=['DT','MD','NN','NNS','NNP','NNPS','VBP','VBD','VB','VBG',\
	'VBN','VBZ','JJ','JJR','JJS','RB','RBR','RBS']
	for i,t in enumerate(data):
		s=[]
		words=WordPunctTokenizer().tokenize(t.lower())
		for w,p in nltk.pos_tag(words):
			if p in goodPOS:
				s.append(w)
		tmp_data.append(' '.join(s))
	return tmp_data

def read_files(tarfname):
	"""Read the training and development data from the sentiment tar file.
	The returned object contains various fields that store sentiment data, such as:

	train_data,dev_data: array of documents (array of words)
	train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
	train_labels,dev_labels: the true string label for each document (same length as data)

	The data is also preprocessed for use with scikit-learn, as:

	count_vec: CountVectorizer used to process the data (for reapplication on new data)
	trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
	le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
	target_labels: List of labels (same order as used in le)
	trainy,devy: array of int labels, one for each document
	"""
	tar = tarfile.open(tarfname, "r:gz")
	trainname = "train.tsv"
	devname = "dev.tsv"
	for member in tar.getmembers():
		if 'train.tsv' in member.name:
			trainname = member.name
		elif 'dev.tsv' in member.name:
			devname = member.name
			
			
	class Data: pass
	sentiment = Data()
	# print("-- train data")
	sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
	# print(len(sentiment.train_data))

	# print("-- dev data")
	sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
	# print(len(sentiment.dev_data))
	
	# print("-- transforming data and labels")
	sentiment.count_vect = CountVectorizer()
	sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
	sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
	
	sentiment.le = preprocessing.LabelEncoder()
	sentiment.le.fit(sentiment.train_labels)
	sentiment.target_labels = sentiment.le.classes_
	sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
	sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
	tar.close()
	return sentiment

def read_files_combined(tarfname):
	tar = tarfile.open(tarfname, "r:gz")
	trainname = "train.tsv"
	devname = "dev.tsv"
	for member in tar.getmembers():
		if 'train.tsv' in member.name:
			trainname = member.name
		elif 'dev.tsv' in member.name:
			devname = member.name
			
			
	class Data: pass
	sentiment = Data()
	sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
	sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)

	sentiment.train_data=filterPOS(sentiment.train_data)
	sentiment.dev_data = filterPOS(sentiment.dev_data)

	sentiment.count_vect = CountVectorizer(max_features=6500)
	sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
	sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
	sentiment.features=sentiment.count_vect.get_feature_names()
	# print("-------3 Boolean weighting")
	# sentiment.trainX=sentiment.trainX.toarray()
	sentiment.binarizer = preprocessing.Binarizer().fit(sentiment.trainX)
	sentiment.trainX=sentiment.binarizer.transform(sentiment.trainX)
	# sentiment.devX=sentiment.devX.toarray()
	sentiment.devX=sentiment.binarizer.transform(sentiment.devX)
	# -----end 3 Boolean weighting------------

	sentiment.le = preprocessing.LabelEncoder()
	sentiment.le.fit(sentiment.train_labels)
	sentiment.target_labels = sentiment.le.classes_
	sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
	sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
	tar.close()
	return sentiment

def read_unlabeled_combined(tarfname, sentiment):
	tar = tarfile.open(tarfname, "r:gz")
	class Data: pass
	unlabeled = Data()
	unlabeled.data = []
	
	unlabeledname = "unlabeled.tsv"
	for member in tar.getmembers():
		if 'unlabeled.tsv' in member.name:
			unlabeledname = member.name
			
	print(unlabeledname)
	tf = tar.extractfile(unlabeledname)
	for line in tf:
		line = line.decode("utf-8")
		text = line.strip()
		unlabeled.data.append(text)
	unlabeled.data=filterPOS(unlabeled.data)
		
			
	unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
	# unlabeled.X=unlabeled.X.toarray()
	unlabeled.X=sentiment.binarizer.transform(unlabeled.X)
	print(unlabeled.X.shape)
	tar.close()
	return unlabeled

def read_unlabeled(tarfname, sentiment):
	"""Reads the unlabeled data.

	The returned object contains three fields that represent the unlabeled data.

	data: documents, represented as sequence of words
	fnames: list of filenames, one for each document
	X: bag of word vector for each document, using the sentiment.vectorizer
	"""
	tar = tarfile.open(tarfname, "r:gz")
	class Data: pass
	unlabeled = Data()
	unlabeled.data = []
	
	unlabeledname = "unlabeled.tsv"
	for member in tar.getmembers():
		if 'unlabeled.tsv' in member.name:
			unlabeledname = member.name
			
	print(unlabeledname)
	tf = tar.extractfile(unlabeledname)
	for line in tf:
		line = line.decode("utf-8")
		text = line.strip()
		unlabeled.data.append(text)
		
			
	unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
	print(unlabeled.X.shape)
	tar.close()
	return unlabeled

def read_tsv(tar, fname):
	member = tar.getmember(fname)
	# print(member.name)
	tf = tar.extractfile(member)
	data = []
	labels = []
	for line in tf:
		line = line.decode("utf-8")
		(label,text) = line.strip().split("\t")
		labels.append(label)
		data.append(text)
	return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
	"""Writes the predictions in Kaggle format.

	Given the unlabeled object, classifier, outputfilename, and the sentiment object,
	this function write sthe predictions of the classifier on the unlabeled data and
	writes it to the outputfilename. The sentiment object is required to ensure
	consistent label names.
	"""
	yp = cls.predict(unlabeled.X)
	labels = sentiment.le.inverse_transform(yp)
	f = open(outfname, 'w')
	f.write("ID,LABEL\n")
	for i in range(len(unlabeled.data)):
		f.write(str(i+1))
		f.write(",")
		f.write(labels[i])
		f.write("\n")
	f.close()


def write_gold_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the truth.

	You will not be able to run this code, since the tsvfile is not
	accessible to you (it is the test labels).
	"""
	f = open(outfname, 'w')
	f.write("ID,LABEL\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(label,review) = line.strip().split("\t")
			i += 1
			f.write(str(i))
			f.write(",")
			f.write(label)
			f.write("\n")
	f.close()

def write_basic_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the naive baseline.

	This baseline predicts POSITIVE for all the instances.
	"""
	f = open(outfname, 'w')
	f.write("ID,LABEL\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(label,review) = line.strip().split("\t")
			i += 1
			f.write(str(i))
			f.write(",")
			f.write("POSITIVE")
			f.write("\n")
	f.close()

def predict_train(sentiment):
	sentiment.count_vect = CountVectorizer(max_features=6500)
	sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
	sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
	sentiment.features=sentiment.count_vect.get_feature_names()

	# print("-------3 Boolean weighting")
	# sentiment.trainX=sentiment.trainX.toarray()
	sentiment.binarizer = preprocessing.Binarizer().fit(sentiment.trainX)
	sentiment.trainX=sentiment.binarizer.transform(sentiment.trainX)
	# sentiment.devX=sentiment.devX.toarray()
	sentiment.devX=sentiment.binarizer.transform(sentiment.devX)
	# -----end 3 Boolean weighting------------

	cls = LogisticRegression(C=0.2, class_weight='balanced', random_state=0, solver='lbfgs', max_iter=10000)
	cls.fit(sentiment.trainX, sentiment.trainy)
	yp=cls.predict(sentiment.devX)
	acc = metrics.accuracy_score(sentiment.devy, yp)
	
	return sentiment,cls,acc

def predict_unlabeled(sentiment,unlabeled,cls):
	unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
	unlabeled.X=sentiment.binarizer.transform(unlabeled.X)
	unlabledypp = cls.predict_proba(unlabeled.X)
	unlabledyp = cls.predict(unlabeled.X)
	return unlabledypp,unlabledyp,unlabeled

def write_kaggle_file(data, cls, outfname, sentiment):
	X=sentiment.count_vect.transform(data)
	X=sentiment.binarizer.transform(X)
	yp = cls.predict(X)
	labels = sentiment.le.inverse_transform(yp)
	f = open(outfname, 'w')
	f.write("ID,LABEL\n")
	for i in range(len(data)):
		f.write(str(i+1))
		f.write(",")
		f.write(labels[i])
		f.write("\n")
	f.close()

def write_feautures(sentiment,fname):
	with open(fname,'w',encoding='utf-8') as f:
		f.write('\t'.join(sentiment.features))

def write_analysis(sentiment,cls,fname):
	with open(fname,'w',encoding='utf-8') as f:
		f.write(str(cls.n_iter_)+'\n')
		weights=cls.coef_
		for x in list(np.argsort(weights[0])[-5:]):
			f.write(str(weights[0][x])+'\t'+sentiment.features[x]+'\n')
		f.write('\n\n')
		for x in list(np.argsort(weights[0])[:5]):
			f.write(str(weights[0][x])+'\t'+sentiment.features[x]+'\n')
		f.write('\n\n')
		weights1=abs(cls.coef_)
		for x in list(np.argsort(weights1[0])[:5]):
			f.write(str(weights[0][x])+'\t'+sentiment.features[x]+'\n')


if __name__ == "__main__":
	# print("Reading data")
	tarfname = "../data/sentiment.tar.gz"
	init_acc=0
	sentiment = read_files_combined(tarfname)
	cls = LogisticRegression(C=0.2, class_weight='balanced',random_state=0, solver='lbfgs', max_iter=10000)
	cls.fit(sentiment.trainX, sentiment.trainy)
	yp=cls.predict(sentiment.devX)
	acc = metrics.accuracy_score(sentiment.devy, yp)
	print(acc)
	# write_feautures(sentiment,"features-0.txt")
	write_analysis(sentiment,cls,'analysis-0.txt')

	unlabeled = read_unlabeled_combined(tarfname, sentiment)
	oriunlabeleddata = copy.deepcopy(unlabeled.data)
	unlabledypp = cls.predict_proba(unlabeled.X)
	unlabledyp = cls.predict(unlabeled.X)
	#strategy 1: expanding most confident predictions 
	# and stop when all unlabeled data are added and acc on dev set > 0.77
	minprop=1.0
	maxacc=acc
	accs=[acc]
	threshold=0.985
	trainsize=[sentiment.trainX.shape[0]]
	addlen=int(unlabeled.X.shape[0]*0.0005)
	# pdb.set_trace()
	while unlabeled.X.shape[0]>0 and minprop>threshold:
		init_acc=acc
		print('init train size',sentiment.trainX.shape[0])
		yp0=unlabledypp[:,0]
		highest0=list(np.argsort(yp0)[-addlen:])
		minprop=yp0[highest0[0]]
		for h in highest0:
			sentiment.train_data.append(unlabeled.data[h])
			sentiment.trainy=np.append(sentiment.trainy,unlabledyp[h])
		
		yp1=unlabledypp[:,1]
		highest=list(np.argsort(yp1)[-addlen:])
		minprop=min(yp1[highest[0]],minprop)
		for h in highest:
			sentiment.train_data.append(unlabeled.data[h])
			sentiment.trainy=np.append(sentiment.trainy,unlabledyp[h])
		
		highest.extend(highest0)
		unlabeled.data=[k for i,k in enumerate(unlabeled.data) if i not in highest]
		
		sentiment,cls,acc=predict_train(sentiment)
		print('expand train size',sentiment.trainX.shape[0]-trainsize[-1])
		trainsize.append(sentiment.trainX.shape[0])
		accs.append(acc)
		print(init_acc,acc)
		unlabledypp,unlabledyp,unlabeled=predict_unlabeled(sentiment,unlabeled,cls)
		if acc>maxacc:
			maxacc=acc
			print("Writing predictions to a file")
			print(len(oriunlabeleddata))
			print(maxacc,sentiment.trainX.shape[0],trainsize[0])
			write_kaggle_file(oriunlabeleddata, cls, "sentiment-pred-1.csv", sentiment)
			# pdb.set_trace()
			# write_feautures(sentiment,"features-1.txt")
			write_analysis(sentiment,cls,'analysis-1.txt')
			

		print('minprop',minprop)

	plt.figure()
	plt.plot(trainsize,accs)
	plt.xlabel('Size of Training Data')
	plt.ylabel('Accuracy')
	plt.grid()
	plt.savefig('1strategy.jpg')

	print("Writing predictions to a file")
	print(len(oriunlabeleddata))
	print(maxacc,sentiment.trainX.shape[0],trainsize[0],sentiment.trainX.shape[0]-trainsize[0]) #0.790
	write_kaggle_file(oriunlabeleddata, cls, "sentiment-pred-2.csv", sentiment)
	# write_feautures(sentiment,"features-2.txt")
	write_analysis(sentiment,cls,'analysis-2.txt')

	