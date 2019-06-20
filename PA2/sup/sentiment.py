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

def read_files_1filterfeatures(tarfname):
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

	# print("-----1 filter features") 
	sentiment.train_data=filterPOS(sentiment.train_data)
	sentiment.dev_data=filterPOS(sentiment.dev_data)
	# ----------end 1 filter-------
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

def read_files_2tfidf(tarfname):
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

	sentiment.count_vect = CountVectorizer()
	sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
	sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
	
	# print("-----2 TFIDF")
	transformer = TfidfTransformer()
	sentiment.trainX = transformer.fit_transform(sentiment.trainX)
	sentiment.devX = transformer.transform(sentiment.devX)
	#------end 2 tfidf----------

	sentiment.le = preprocessing.LabelEncoder()
	sentiment.le.fit(sentiment.train_labels)
	sentiment.target_labels = sentiment.le.classes_
	sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
	sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
	tar.close()
	return sentiment

def read_files_3boolean(tarfname):
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

	sentiment.count_vect = CountVectorizer()
	sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
	sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)

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

def read_files_4df(tarfname,mindf=1):
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
	sentiment.count_vect = CountVectorizer(min_df=mindf)
	sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
	sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)

	sentiment.le = preprocessing.LabelEncoder()
	sentiment.le.fit(sentiment.train_labels)
	sentiment.target_labels = sentiment.le.classes_
	sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
	sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
	tar.close()
	return sentiment

def read_files_5maxfeature(tarfname,mf):
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
	sentiment.count_vect = CountVectorizer(max_features=mf)
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


if __name__ == "__main__":
	# print("Reading data")
	tarfname = "../data/sentiment.tar.gz"
	# print("\nTraining classifier")
	sentiment = read_files(tarfname)
	cls = LogisticRegression(C=1,random_state=0, solver='lbfgs', max_iter=10000)
	cls.fit(sentiment.trainX, sentiment.trainy)
	yp=cls.predict(sentiment.devX)
	acc = metrics.accuracy_score(sentiment.devy, yp)
	print('benchmark',acc)

	sentiment = read_files_1filterfeatures(tarfname)
	cls = LogisticRegression(C=1,random_state=0, solver='lbfgs', max_iter=10000)
	cls.fit(sentiment.trainX, sentiment.trainy)
	yp=cls.predict(sentiment.devX)
	acc = metrics.accuracy_score(sentiment.devy, yp)
	print('filter features',acc)

	sentiment = read_files_2tfidf(tarfname)
	cls = LogisticRegression(C=1,random_state=0, solver='lbfgs', max_iter=10000)
	cls.fit(sentiment.trainX, sentiment.trainy)
	yp=cls.predict(sentiment.devX)
	acc = metrics.accuracy_score(sentiment.devy, yp)
	print('TFIDF weighting',acc)

	sentiment = read_files_3boolean(tarfname)
	cls = LogisticRegression(C=1,random_state=0, solver='lbfgs', max_iter=10000)
	cls.fit(sentiment.trainX, sentiment.trainy)
	yp=cls.predict(sentiment.devX)
	acc = metrics.accuracy_score(sentiment.devy, yp)
	print('boolean weighting',acc)

	accs=[]
	for mindf in np.arange(0.0,0.5,0.05):
		sentiment = read_files_4df(tarfname,mindf)
		cls = LogisticRegression(C=1,random_state=0, solver='lbfgs', max_iter=10000)
		cls.fit(sentiment.trainX, sentiment.trainy)
		yp=cls.predict(sentiment.devX)
		acc = metrics.accuracy_score(sentiment.devy, yp)
		print('df filtering',mindf,acc)
		accs.append(acc)
	plt.figure(figsize=(12,5))
	plt.subplot(121)
	plt.plot(list(np.arange(0.0,0.5,0.05)),accs)
	plt.xlabel('min_df')
	plt.ylabel('Accuracy')
	plt.grid()
	# plt.title('Using DF to filter features')
	

	accs=[]
	for mf in np.arange(500,10000,500):
		sentiment = read_files_5maxfeature(tarfname,mf)
		cls = LogisticRegression(C=1,random_state=0, solver='lbfgs', max_iter=10000)
		cls.fit(sentiment.trainX, sentiment.trainy)
		yp=cls.predict(sentiment.devX)
		acc = metrics.accuracy_score(sentiment.devy, yp)
		print('TF filtering',mf,acc)
		accs.append(acc)
	plt.subplot(122)
	plt.plot(list(np.arange(500,10000,500)),accs)
	plt.xlabel('max_feature')
	plt.ylabel('Accuracy')
	plt.grid()
	# plt.title('Using TF to filter features')
	plt.savefig('12df-mf.jpg') #6500

	###############7###############

	sentiment = read_files_combined(tarfname)
	accs=[]
	for c in np.arange(.1, 3, .1):
		cls = LogisticRegression(C=c,random_state=0, solver='lbfgs', max_iter=10000)
		cls.fit(sentiment.trainX, sentiment.trainy)
		yp=cls.predict(sentiment.devX)
		acc = metrics.accuracy_score(sentiment.devy, yp)
		print('regularization strength',c,acc)
		accs.append(acc)
	plt.figure()
	plt.plot(list(np.arange(.1, 3, .1)),accs)
	plt.xlabel('C')
	plt.ylabel('Accuracy')
	plt.grid()
	# plt.title('Changing regularization strength')
	plt.savefig('3regularization.jpg')

	bestp=np.argmax(accs)
	bestc=list(np.arange(.1, 3, .1))[bestp]
	# bestc=0.1
	print(bestc)
	cls=LogisticRegression(C=bestc,random_state=0, solver='lbfgs', max_iter=10000)
	cls.fit(sentiment.trainX, sentiment.trainy)
	trainyp=cls.predict(sentiment.trainX)
	devyp=cls.predict(sentiment.devX)
	print(classification_report(sentiment.trainy, trainyp))
	print(classification_report(sentiment.devy, devyp))

	print("\nReading unlabeled data")
	unlabeled = read_unlabeled_combined(tarfname, sentiment)
	print("Writing predictions to a file")
	write_pred_kaggle_file(unlabeled, cls, "sentiment-pred.csv", sentiment)
	#write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

	# # # # You can't run this since you do not have the true labels
	# # # # print "Writing gold file"
	# # # # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
