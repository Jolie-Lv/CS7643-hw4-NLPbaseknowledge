import random

import torch
import torch.nn as nn
import torch.optim as optim

class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """
    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout = 0.2, model_type = "RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #############################################################################
        self.emb = nn.Embedding(output_size,emb_size)
        if model_type == "RNN":
            self.rnn = nn.RNN(emb_size,decoder_hidden_size,batch_first=True)
        else:
            self.rnn = nn.LSTM(emb_size, decoder_hidden_size,batch_first=True)
        self.linear = nn.Linear(decoder_hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim = 2)
        self.dropout = nn.Dropout(dropout)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, 1); HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """
        

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply linear layer and softmax activation to output tensor before   #
        #       returning it.                                                       #
        #############################################################################
     
        # output, hidden = None, None

        embs = self.dropout(self.emb(input))
        # print(hidden.shape)
        if self.model_type == 'RNN':
            output, hidden = self.rnn(embs, hidden)
            # hidden = hidden[:, 0].unsqueeze(0)

        else:

            output, (hidden1,cell) = self.rnn(embs, hidden)
            # hidden1 = hidden1[:,-1].unsqueeze(0)
            # cell = cell[:,-1].unsqueeze(0)
            hidden = (hidden1,cell)

        output = self.logsoftmax(self.linear(output))
        output = output[:,0,:]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
