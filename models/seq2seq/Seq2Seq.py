import random

import torch
import torch.nn as nn
import torch.optim as optim

# import custom models



class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the Seq2Seq model. You should use .to(device) to make sure  #
        #    that the models are on the same device (CPU/GPU). This should take no  #
        #    more than 2 lines of code.                                             #
        #############################################################################
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, source, out_seq_len = None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]
        if out_seq_len is None:
            seq_len = source.shape[1]

        
        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass of the Seq2Seq model. Please refer to the    #
        #   following steps:                                                        #
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hidden state of the decoder                            #
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence.                             #
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs.                                                   #
        #       4) Update the input and hidden weights being fed into the decoder   #
        #          at each time step. The decoder output at the previous time step  # 
        #          will have to be manipulated before being fed in as the decoder   #
        #          input at the next time step.                                     #
        #############################################################################
        if out_seq_len is not None:
            seq_len = out_seq_len

        outputs = torch.zeros(batch_size, seq_len, self.decoder.output_size).to(self.device)
        source = source.to(self.device)
        enc_outs, decoder_hidden = self.encoder(source)
        dec_inputs = source[:,0].unsqueeze(1)


        for i in range(1,seq_len):

            dec_out, decoder_hidden = self.decoder(dec_inputs, decoder_hidden)
            outputs[:,i,:] = dec_out

            # dec_inputs = dec_out.argmax(1).view(1,1)
            dec_inputs = torch.argmax(dec_out, dim=1, keepdim=True)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs


# from Encoder import Encoder
# from Decoder import Decoder
# import numpy as np
# RANDOM_SEED = 0
# def set_seed_nb():
#     torch.manual_seed(RANDOM_SEED)
#     np.random.seed(RANDOM_SEED + 1)
# set_seed_nb()
# embedding_size = 32
# hidden_size = 32
# input_size = 8
# output_size = 8
# batch, seq = 1, 2
#
# encoder = Encoder(input_size, embedding_size, hidden_size, hidden_size)
# decoder = Decoder(embedding_size, hidden_size, hidden_size, output_size)
#
# seq2seq = Seq2Seq(encoder, decoder, 'cpu')
# x_array = np.random.rand(batch, seq) * 10
# x = torch.LongTensor(x_array)
# out = seq2seq.forward(x)
# print(out)
# expected_out = torch.FloatTensor([[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
#            0.0000],
#          [-2.4136, -2.2861, -1.7145, -2.5612, -1.9864, -2.0557,
#           -1.7461,
#           -2.1898]]])
# print('Close to out: ', expected_out.allclose(out, atol=1e-4))