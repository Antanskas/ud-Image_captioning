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
        Embedding layer
        
        num_embeddings (int) – size of the dictionary of embeddings
        embedding_dim (int) – the size of each embedding vector
        
        Input: (*), LongTensor of arbitrary shape containing the indices to extract
        Output: (*, H), where * is the input shape and H=embedding_dim (10,11,512)        
        
        """
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        
        
        """
        LSTM layer
        
        input_size – The number of expected features in the input x
        hidden_size – The number of features in the hidden state h
        num_layers – Number of recurrent layers
        batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)       
    
        Inputs: input, (h_0, c_0)
        Outputs: output, (h_n, c_n)
        
        """
        self.lstm = nn.LSTM(
                            input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True
                            )
        """
        Linear layer
        
        in_features – size of each input sample
        out_features – size of each output sample
        
        Input: (N, *, H_in) where * means any number of additional dimensions and H_in = in_features
        Output: (N, *, H_out) where all but the last dimension are the same shape as the input and H_out = out_features        
        
        """
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        
        
        """
        Weight initialization
        """
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
          Contains shape of (batch_size, embed_dim) what's in our case (10,512)

        * captions - annotations converted into tokend.
          Contains shape of (batch_size, len_of_annot) what's in our case (10,11)
          because all annotation length are the same in the same batch. It was provided
          by data_loader.

        """
        
        """
        Removing <END> token because during training it has never been passed.
        '<START> The man standing on the skateoard <END>'
        All starts with passing image features which has size of embeddings (512) as input to LSTM and hidden previous state as zero.
        Then it will train to predict <START> from that (at the beginning it can predict anything). Then first true
        token <START> will be passed and the hidden state will be the output from previous hidden unit. Then
        RNN need predict next word (The), then second word of sequence will be passed to second unit as input 
        and hidden state from previous unit. This goes to the end of sequence where given last token 'skateboard' 
        last unit t takes it as input together with h(t-1) and need to predict token <END>. Whatever it predicts, 
        the loss will be calculated treating that the last prediction should be <END> cuz it goes after 'skateboard'.
        """
        captions = captions[:,:-1] 
        embeddings = self.embed(captions) # shape of (10,11,512)      
        
        features = features.unsqueeze(1) # shape of (10,1,512)
        
        
        """
        Concat on dimension 1. 
        It is SUPER important keep order of features and embeddings when concatenating 
        (first entry in whole sequence should be vector of image of length 512, 
        then first token <START> embedded as vector of length 512, second token embedded of lenght 512 and so on).
        because image features serves as first input in LSTM system - first lstm cell need to 
        receive image features, get output and use those outputs for next lstm cell as hidden state. 
        The second lstm cell receives firsts lstm outputs generated from image features as hidden state 
        and input as first word from sequence (<START>) in embeddings (as vector of embed_size length).
        """
        inputs = torch.cat(tensors=(features, embeddings), dim=1) 
        
        """
        Passing inputs to LSTM system. 
        If no hidden states passed, then by default it is zeros. h_0 = 0, c_0 = 0
        """
        lstm_out, (h_n, c_n) = self.lstm(inputs)
        output = self.fc(lstm_out)

        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        preds = []
        count = 0
        word_item = None
        
        while count < max_len and word_item != 1 :
            
            #Predict output
            output_lstm, states = self.lstm(inputs, states)
            output = self.fc(output_lstm)
            
            #Get max value
            prob, word = output.max(2)
            
            #append word
            word_item = word.item()
            preds.append(word_item)
            
            #next input is current prediction
            inputs = self.embed(word)
            
            count+=1
        
        return preds