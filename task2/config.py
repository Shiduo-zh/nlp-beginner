class ConfigCNN():
    def __init__(self):
        self.lr=1e-4
        self.model_type='cnn'
        self.model_param=dict(
            pretrained_embedding=None,
            static=False,
            embedding_dim=300,
            num_embeddings=None,
            filter_sizes = [3, 4, 5],
            out_channels = [100, 100, 100],
            num_classes=5,
            dropout=0.5
        )
        self.batch_size=50
        self.epoch=50
        self.early_stop=True
    
    def get_default_param(self):
        param=dict(
            lr=self.lr,
            model_type=self.model_type,
            model_param=self.model_param,
            batch_size=self.batch_size,
            epoch=self.epoch,
            early_stop=self.early_stop
        )
        return param

class ConfigRNN():
    def __init__(self):
        self.lr=1e-4
        self.model_type='rnn'
        self.model_param=dict(
            pretrained_embedding=None,
            static=False,
            embedding_dim=300,
            num_embeddings=None,
            hidden_size=300,
            num_classes=5,
            dropout=0.2
        )
        self.batch_size=50
        self.epoch=50
        self.early_stop=True
    
    def get_default_param(self):
        param=dict(
            lr=self.lr,
            model_type=self.model_type,
            model_param=self.model_param,
            batch_size=self.batch_size,
            epoch=self.epoch,
            early_stop=self.early_stop
        )
        return param