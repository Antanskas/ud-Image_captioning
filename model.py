import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        """
        Embedding layer - returns embeddings shape of (*, H) where * is shape of inputs
        and H is size of the desired embeddings - (10,11,256) in our case
        """
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        # lstm layer

        self.lstm = nn.LSTM(
                            input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True
                            )
        # linear layer
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        # weights initialization
        self.init_weights()

    def init_weights(self):
        # linear layer weight initialization
        torch.nn.init.xavier_uniform(self.fc.weight)
        self.fc.bias.data.fill_(0)
        # embedding layer weight initialization
        torch.nn.init.xavier_uniform(self.embed.weight)

    
    def forward(self, features, captions):
        """
        * features - outputs from convolutional blocks.
          Contains shape of (batch_size, embed_dim) and its in our case (10,256)

        * captions - annotations converted into tokend.
          Contains shape of (batch_size, len_of_annot) and its in our case (10,11)
          because all annotation length are the same in the same batch. It was provided
          by data_loader.

        """
        captions = captions[:,:-1] # removing <END> token
        features = features.unsqueeze(1) # shape of (10,1,256)
        embed = self.embed(captions) # shape of (10,11,256)

        inputs = torch.cat(tensors=(embed, features), dim=1) # concat on dimension 1

        lstm_out, (h_n, c_n) = self.lstm(inputs)

        output = self.fc(lstm_out)

        return output



    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        predicts = []
        count = 0
        word = None

        while count < max_len and word != 1:
            
            # get outputs
            lstm_out, states = self.lstm(inputs, states)
            fc_out = self.fc(lstm_out)

            # get probs and words indices
            p, word = fc_out.max(2)
            predicts.append(word.item())

            # predicted word in t-1 feeded as input to t
            inputs = self.embed(word)

            count+=1

        return predicts