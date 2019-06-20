using the following steps, we can obtain the results in our paper.
#the scripts should be put in the same directory of all input files.
#or change the path in the scripts

python 1_train_ibm1.py
python 1_predict_ibm1.py
python eval_alignment.py dev.key 1_dev.out
python 1_discussion.py

python 2_train_ibm2.py
python 2_predict_ibm2.py
python eval_alignment.py dev.key 2_dev.out
python 2_discussion.py

python 3_train.py
python 3_predict.py
python eval_alignment.py dev.key 3_dev.out
# python eval_alignment.py dev.key 3_1_dev.out
# python eval_alignment.py dev.key 3_2_dev.out
