import torch 
import torch.nn as nn 
import torch.nn.functional as f

class RNN(nn.Module):
    def __init__(self,
                pretrained_embedding=None,
                static=False,
                embedding_dim=100,
                num_embeddings=None,
                max_len=53,
                hidden_size=300,
                num_classes=5,
                dropout=0.2
                ):
        super(RNN,self).__init__()
        #embedding layer
        if pretrained_embedding is not None:
            self.num_embeddings, self.embedding_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=static)
        else:
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=self.embedding_dim,
                                      padding_idx=0,
                                      max_norm=5.0)
        self.hidden_size=hidden_size
        self.LSTM=nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=hidden_size)
        self.dropout=nn.Dropout(dropout)
        self.fc_layer=nn.Linear(self.hidden_size*max_len,num_classes)

    def forward(self,x):
        batch_size=x.shape[0]
        #with shape(batch_size,max_sentence_len,embedding_dim)
        x_embedding=self.embedding(x).type(torch.float32)
        x_embedding = x_embedding.permute(1, 0, 2)
        #rnn input shape(max_sentence_len,,batch_size,embedding_dim)
        #hidden_state=torch.rand((1,batch_size,self.hidden_size))
        x_rnn,(_,_)=self.LSTM(x_embedding)
        x_rnn=x_rnn.permute(1,0,2).reshape(batch_size,-1)
        onehot=self.fc_layer(self.dropout(x_rnn))
        return onehot
