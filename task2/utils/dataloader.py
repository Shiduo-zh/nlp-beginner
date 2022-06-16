import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)
from itertools import islice
from sklearn.model_selection import train_test_split
from utils.tokenizer import *
import os 
import sys
project_path = os.path.join(
        os.path.dirname(__file__),
        ".."
    )
sys.path.append(project_path)

def load_text():
    datapath='data/train.tsv'
    data=dict(
        PhraseId=[],
        SentenceId=[],
        Phrase=[],
        Class=[],
        PhraseNum=[],
    )
    with open(datapath,'rb') as f:
        for num,line in enumerate(f):
            if(num==0):
                continue
            linedata=line.decode(encoding = "utf-8")
            phraseid,sentenceid,phrase,classid=linedata.split('\t')
            data['PhraseId'].append(phraseid)
            data['SentenceId'].append(sentenceid)
            data['Phrase'].append(phrase)
            data['Class'].append(int(classid))
            data['PhraseNum'].append(len(phrase.split(' ')))
        for i in range(len(data['Class'])):
            onehot=np.zeros(5)
            onehot[data['Class'][i]]=1
            data['Class'][i]=onehot
                
        # for i in range(len(data['Phrase'])):
        #     data['Phrase'][i]=data['Phrase'][i].split(' ')
    return data

def split_train_dev_test(data):
    """
    Split the dataset to train dataset,validation dataset and test dataset.
    Train:validation:test=0.8:0.1:0.1

    params:
    data:original data including inputids and labels(classes)

    return:
    divided dataset included in a data dictionary
    """
    input_ids=data['Input_ids']
    labels=data['Class']
    train_inputs, dev_test_inputs, train_labels, dev_test_labels = train_test_split(
    input_ids, labels, test_size=0.2, random_state=42)
    
    dev_inputs,test_inputs,dev_labels,test_labels=train_test_split(
    dev_test_inputs,dev_test_labels,test_size=0.5,random_state=3
    )

    data_input=dict(
        train_x=train_inputs,
        dev_x=dev_inputs,
        test_x=test_inputs,
        train_class=train_labels,
        dev_class=dev_labels,
        test_class=test_labels
    )
    return data_input

if __name__=='__main__':
    data=load_text()
    tokenized_texts, word2idx, max_len=tokenize(data['Phrase'])
    input_ids=encode(tokenized_texts, word2idx, max_len)
    data['Input_ids']=input_ids
    data_input=split_train_dev_test(data)   
  

