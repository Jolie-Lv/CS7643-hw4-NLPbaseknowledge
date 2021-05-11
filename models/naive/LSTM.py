import numpy as np
import torch
import torch.nn as nn

class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization


    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes as you wish here.                      #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   You also need to include correct activation functions                      #
        #   Initialize the gates in the order above!                                   #
        #   Initialize parameters in the order they appear in the equation!            #                                                              #
        ################################################################################


        #i_t: input gate
        self.w_ii = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.b_ii = nn.Parameter(torch.zeros(hidden_size))
        self.w_hi = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.b_hi = nn.Parameter(torch.zeros(hidden_size))



        # f_t: the forget gate
        self.w_if = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.b_if = nn.Parameter(torch.zeros(hidden_size))
        self.w_hf = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.b_hf = nn.Parameter(torch.zeros(hidden_size))

        # g_t: the cell gate
        self.w_ig = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.b_ig = nn.Parameter(torch.zeros(hidden_size))
        self.w_hg = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.b_hg = nn.Parameter(torch.zeros(hidden_size))

        # o_t: the output gate
        self.w_io = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.b_io = nn.Parameter(torch.zeros(hidden_size))
        self.w_ho = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.b_ho = nn.Parameter(torch.zeros(hidden_size))

        self.sigmoid_f = nn.Sigmoid()
        self.sigmoid_i = nn.Sigmoid()
        self.tanh_i = nn.Tanh()
        self.sigmoid_o = nn.Sigmoid()
        self.tanh_o = nn.Tanh()
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        
        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################

        h_t, c_t = None, None
        if init_states is None:
            h_t, c_t = nn.Parameter(torch.zeros(x.size(0), self.hidden_size)),\
                   nn.Parameter(torch.zeros(x.size(0), self.hidden_size))
        else:
            h_t, c_t = init_states


        for t in range(x.size(1)):
            x_t = x[:,t,:]
            i_t = self.sigmoid_i(torch.matmul(x_t, self.w_ii)+torch.matmul(h_t,self.w_hi)+self.b_ii+self.b_hi)
            f_t = self.sigmoid_f(torch.matmul(x_t, self.w_if)+torch.matmul(h_t,self.w_hf)+self.b_if+self.b_hf)
            g_t = self.tanh_i(torch.matmul(x_t,self.w_ig)+torch.matmul(h_t,self.w_hg)+self.b_ig+self.b_hg)
            o_t = self.sigmoid_o(torch.matmul(x_t,self.w_io)+torch.matmul(h_t,self.w_ho)+self.b_io+self.b_ho)
            c_t = f_t*c_t+i_t*g_t
            h_t = o_t*self.tanh_o(c_t)


        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)

