from copy import deepcopy
import os
import inspect
from torch.autograd import Variable
from model.cnn import CNN
from model.rnn import RNN
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

parentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
logdir=os.path.join(parentdir,'logs')

def train(data,**kwargs):
    logpath=os.path.join(logdir,kwargs['exp_id'])
    if(not os.path.exists(logpath)):
        os.makedirs(logpath)
    
    writer=SummaryWriter(logpath)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    if(kwargs['model_type']=='cnn'):
        model=CNN(**kwargs['model_param'])
    elif(kwargs['model_type']=='rnn'):
        model=RNN(**kwargs['model_param'])
    
    model.to(device)

    param=filter(lambda p: p.requires_grad, model.parameters())
    optimizer=torch.optim.Adam(param,kwargs['lr'])
    criterion = nn.CrossEntropyLoss()

    batch_size=kwargs['batch_size']
    pre_dev_acc = 0
    max_dev_acc = 0
    max_test_acc = 0
    #train in many epochs for best model 
    print('epoch | train loss | val_acc | best_acc1 | test acc | best_acc2')
    print('----------------------------------------')
    step=0
    for epoch in range(kwargs['epoch']):
        N_batch=len(data['train_x'])//batch_size
        for i in range(0,N_batch*batch_size,batch_size):
            batch_x=[data['train_x'][j] for j in range(i,i+batch_size)]
            batch_y=[data['train_class'][j] for j in range(i,i+batch_size)]

            batch_x=Variable(torch.tensor(np.array(batch_x)).to(device))
            batch_y=Variable(torch.tensor(np.array(batch_y).astype(np.float32)).to(device).squeeze(0))

            optimizer.zero_grad()
            model.train()
            pred=model(batch_x)
            loss=criterion(pred.to(device),batch_y)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
        
        dev_acc=test(data,model,device,mode='dev',**kwargs)
        test_acc=test(data,model,device,mode='test',**kwargs)
        if (dev_acc<pre_dev_acc):
            step+=1
        else:
            step=0

        if(kwargs['early_stop'] is True and step>5):
            break
            
        else:
            pre_dev_acc=dev_acc

        if dev_acc>max_dev_acc:
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            bestmodel=deepcopy(model)
        
        writer.add_scalar('train_loss',loss,epoch)
        writer.add_scalar('val_acc',dev_acc,epoch)
        writer.add_scalar('max_dev_acc',max_dev_acc,epoch)
        writer.add_scalar('test_acc',test_acc,epoch)
        writer.add_scalar('max_test_acc',max_test_acc,epoch)
        
        
        #loginfo='epoch:{},current max dev accuracy is {},current max test accuracy is {},\
            #dev accucary this time is {},test accurary this time is {}'.format(epoch,max_dev_acc,max_test_acc,dev_acc,test_acc)

        print('{:^5} | {:^10} | {:^7} | {:^9} | {:^8} | {:^9} |'.format(epoch+1,loss,dev_acc,max_dev_acc,test_acc,max_test_acc))

    return bestmodel

def test(data,model,device,mode,**kwargs):
    model.eval()
    if mode=='dev':
        x,y=data['dev_x'],data['dev_class']
    elif mode=='test':
        x,y=data['test_x'],data['test_class']
    
    x=Variable(torch.tensor(x).to(device))
    pred_y=model(x)
    pred_y_class=[torch.argmax(cls) for cls in pred_y] 
    ground_y_class=[torch.argmax(torch.tensor(cls)) for cls in y] 
    acc=sum([1 if p==g else 0 for p,g in zip(pred_y_class,ground_y_class)])/len(pred_y_class)
    
    return acc

if __name__=='__main__':
    testtorch=torch.rand((50,5))
    index=[torch.argmax(t) for t in testtorch]
    a=1
