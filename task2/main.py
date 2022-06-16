import torch
import torch.nn as nn
import argparse
import os
from config import ConfigCNN,ConfigRNN
from utils.dataloader import load_text
from utils.dataloader import tokenize
from run import train,test
from utils.dataloader import *
from utils.tokenizer import *
import pickle
import inspect

parentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
embedding_file=os.path.join(parentdir,'data/embeddings.pkl')

def main(args):
    #init params and configs
    if(args.model_type=='cnn'):
        conf=ConfigCNN()
    elif(args.model_type=='rnn'):
        conf=ConfigRNN()
    default_conf=conf.get_default_param()
    default_conf['exp_id']=args.exp_id
    #load initial dataset
    print('loading texts...')
    data=load_text()
    #tokenize data
    print('initialzing tokens...')
    tokenized_texts, word2idx, max_len=tokenize(data['Phrase'])
    if(args.model_type=='rnn'):
        default_conf['model_param']['max_len']=max_len
    if(args.pre_embedding==True):
        with open(embedding_file,'rb') as f:
            embedding=torch.tensor(pickle.load(f))
        default_conf['model_param']['pretrained_embedding']=embedding
    input_ids=encode(tokenized_texts, word2idx, max_len)
    data['Input_ids']=input_ids
    default_conf['model_param']['num_embeddings']=len(word2idx)
    #init dataset for network
    data_input=split_train_dev_test(data)
    print('start training...')
    bestmodel=train(data_input,**default_conf)
    print('training completed...') 
    save_path=os.path.join(args.save_dir,args.model_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(os.path.join(args.save_dir,args.model_type))
    torch.save(bestmodel,save_path)

def get_args():
    parser=argparse.ArgumentParser(description='nlp for classify sentences')
    parser.add_argument(
        '--save-dir',
        default='task2/saved_models',
        help='the tpye of training environment'
    )
    parser.add_argument(
        '--model-name',
        default='bestmodel.pt',
        help='model name saved'
    )
    parser.add_argument(
        '--seed',
        default='0',
        help='random seed'
    )
    parser.add_argument(
        '--model-type',
        default='rnn',
        help='choose cnn or rnn for training'
    )
    parser.add_argument(
        "--pre_embedding",
        default=False,
        help='whether use pretrained embeddings or not'
    )
    parser.add_argument(
        "--exp-id",
        default='cnn-word-emd',
        help='description of the exp logs'
    )
    args=parser.parse_args()
    return args

if __name__=='__main__':
    args=get_args()
    main(args)