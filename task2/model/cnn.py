import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self,
                pretrained_embedding=None,
                static=False,
                embedding_dim=100,
                num_embeddings=None,
                filter_sizes = [2, 3, 4],
                out_channels = [2, 2, 2],
                num_classes=5,
                dropout=0.5
                ):
        """
        params:"""
        super(CNN,self).__init__()
        #embedding layer
        if pretrained_embedding is not None:
            self.num_embeddings, self.embedding_dim = pretrained_embedding.shape
            # self.embedding = nn.Embedding(num_embeddings=num_embeddings,
            #                           embedding_dim=self.embedding_dim,
            #                           padding_idx=0,
            #                           max_norm=5.0,
            #                           _weight=pretrained_embedding)
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=static,
                                                          )
        else:
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=self.embedding_dim,
                                      padding_idx=0,
                                      max_norm=5.0)
        #conv1dNet
        self.convNet=nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=out_channels[i],
                kernel_size=filter_sizes[i])
             for i in range(len(filter_sizes))
             ])
        
        #full connection layer and dropout layer
        self.fc_layer=nn.Linear(np.sum(out_channels),num_classes)
        self.dropout=nn.Dropout(p=dropout)
    
    def forward(self,x):
        #with shape(batch_size,max_sentence_len,embedding_dim)
        x_embedding=self.embedding(x).type(torch.float32)
        x_embedding = x_embedding.permute(0, 2, 1)
        #conv input shape(batch_size,embedding_dim,max_sentence_len)
        x_conv_list = [F.relu(conv1d(x_embedding)) for conv1d in self.convNet]
        # Max pooling. 
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
        #output shape: (batch_size, n_classes)
        onehot = self.fc_layer(self.dropout(x_fc))
        # class_result=[]
        # for result in onehot:
        #     classid=torch.argmax(result)
        #     class_result.append(classid)
        # return torch.unsqueeze(torch.Tensor(class_result).float(),0)
        return onehot