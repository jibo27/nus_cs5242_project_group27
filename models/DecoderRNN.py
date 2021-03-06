import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .Attention import Attention


class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    """

    def __init__(self,
                 obj_vocab_size,
                 rel_vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 n_layers=1,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1):
        super(DecoderRNN, self).__init__()

        self.bidirectional_encoder = bool(bidirectional)

        self.obj_vocab_size = obj_vocab_size
        self.rel_vocab_size = rel_vocab_size
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = 1
        self.eos_id = 0
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.attention = Attention(self.dim_hidden)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(
            self.dim_hidden + dim_word, # PAD
            self.dim_hidden,
            n_layers,
            batch_first=True,
            dropout=rnn_dropout_p)


        self.out_obj = nn.Linear(self.dim_hidden, self.obj_vocab_size)
        self.out_rel = nn.Linear(self.dim_hidden, self.rel_vocab_size)

        self._init_weights()

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                targets=None,
                mode='train',
                opt={}):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences

        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)
        padding_word = Variable(encoder_outputs.data.new(batch_size, self.dim_word)).zero_()

        seq_logprobs = []
        seq_preds = []
        preds5_list = []
        self.rnn.flatten_parameters()
        if mode == 'train':
            # use targets as rnn inputs
            for i in range(self.max_length):
                context = self.attention(decoder_hidden[0].squeeze(0), encoder_outputs)


                decoder_input = torch.cat([padding_word, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)

                if i != 1:
                    logits = self.out_obj(decoder_output.squeeze(1))
                else:
                    logits = self.out_rel(decoder_output.squeeze(1))


                logprobs = F.log_softmax(logits, dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))

            return seq_logprobs, seq_preds

        else:
            if beam_size > 1:
                return self.sample_beam(encoder_outputs, decoder_hidden, opt)

            for t in range(self.max_length):
                context = self.attention(
                    decoder_hidden[0].squeeze(0), encoder_outputs)

                decoder_input = torch.cat([padding_word, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)


                if t != 1:
                    logits = self.out_obj(decoder_output.squeeze(1))
                else:
                    logits = self.out_rel(decoder_output.squeeze(1))

                logprobs = F.log_softmax(logits, dim=1)

                seq_logprobs.append(logprobs.unsqueeze(1))
                _, preds5 = logprobs.topk(5, 1, True, True)

                preds5_list.append(preds5)

            return seq_logprobs, preds5_list


    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out_obj.weight)
        nn.init.xavier_normal_(self.out_rel.weight)

    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(
                [self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
