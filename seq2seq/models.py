import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import logging
import sys
import torch.optim as optim

import lib_pdf
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import math
import copy

from myutil import *

logger = logging.getLogger()

class RNNLatentDecoder(nn.Module): 
    def __init__(self, embed_size, hidden_size, latent_size, vocab, vocab_inv, seq_len, layer_num = 1, dropout_rate = 0):
        super(RNNLatentDecoder, self).__init__()
        
        vocab_size = len(vocab)
        self.decode_embedding = nn.Embedding(len(vocab), embed_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.seq_len = seq_len
        self.vocab = vocab
        self.vocab_inv = vocab_inv
        self.layer_num = layer_num
        self.dropout_rate = dropout_rate
        logger.info('building lstm latent decoder, dropout: %f layer_num: %d', dropout_rate, layer_num)

        self.decode_rnn = nn.LSTM(embed_size + latent_size, hidden_size, layer_num, batch_first = True, dropout = dropout_rate)
        self.dropout_h1 = nn.Dropout(p=dropout_rate)
        self.dropout_i = nn.Dropout(p=dropout_rate)
        
        self.outLinear = nn.Linear(hidden_size, vocab_size, bias=True)
        #self.softmax = nn.LogSoftmax()

    def init_hidden_lstm(self, bsz):
        zeros1 = Variable(torch.zeros(self.layer_num, bsz, self.hidden_size)).cuda()
        zeros2 = Variable(torch.zeros(self.layer_num, bsz, self.hidden_size)).cuda()
        return zeros1, zeros2

    def forward(self, latent, indices, lengths): #change from decode to forward for dataparallel
        len_now = indices.size(1) #here everything is SEQ_LEN
        batch_size = indices.size(0) #len(lengths)
        # batch x hidden
        #print latent.unsqueeze(1).size()
        #moved to main #all_latent = latent.unsqueeze(1).repeat(1, len_now, 1)
        
        state = self.init_hidden_lstm(batch_size)

        embeddings = self.decode_embedding(indices)
        embeddings = self.dropout_i(embeddings)
        augmented_embeddings = torch.cat([embeddings, latent], 2)

        output, state = self.decode_rnn(augmented_embeddings, state)
        
        output = self.dropout_h1(output) 
        # reshape to batch_size*maxlen x nhidden before linear over vocab
        logit = self.outLinear(output)
        #decoded = decoded.view(batch_size, len_now, len(self.vocab))

        return logit
    
    def generate(self, latent, maxlen, sample=False):
        batch_size = latent.size(0)

        state = self.init_hidden_lstm(batch_size)
        #maxlen = latent.size(1)
        start_symbols = Variable(torch.LongTensor(batch_size, 1)).cuda()
        start_symbols.data.fill_(self.vocab_inv['<s>'])

        embedding = self.decode_embedding(start_symbols)
        embedding = self.dropout_i(embedding)
        inputs = torch.cat([embedding, latent[:,0,:].unsqueeze(1)], 2)

        all_indices = []
        for i in range(maxlen):
            output, state = self.decode_rnn(inputs, state)
            output = self.dropout_h1(output)
            overvocab = self.outLinear(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
                indices = indices.unsqueeze(1)
            else:
                probs = F.softmax(torch.clamp(overvocab, -40, 40)) #could prevent inf after softmax, which will be bad for multinomail 
                indices = torch.multinomial(probs, 1)
            all_indices.append(indices)

            embedding = self.decode_embedding(indices)
            if i < (maxlen - 1):
                inputs = torch.cat([embedding, latent[:,i + 1,:].unsqueeze(1)], 2)
        
        max_indices = torch.cat(all_indices, 1)

        return max_indices.data

    def generate_samplemin(self, latent, maxlen, thres): #very reduntant code, sorry...
        batch_size = latent.size(0)

        state = self.init_hidden_lstm(batch_size)
        #maxlen = latent.size(1)
        start_symbols = Variable(torch.LongTensor(batch_size, 1)).cuda()
        start_symbols.data.fill_(self.vocab_inv['<s>'])

        embedding = self.decode_embedding(start_symbols)
        embedding = self.dropout_i(embedding)
        inputs = torch.cat([embedding, latent[:,0,:].unsqueeze(1)], 2)

        all_indices = []
        for i in range(maxlen):
            output, state = self.decode_rnn(inputs, state)
            output = self.dropout_h1(output)
            overvocab = self.outLinear(output.squeeze(1))

            probs = F.softmax(overvocab)
            for j in range(batch_size):
                ss = 0
                min_num = torch.sum(torch.log(probs[j]) > thres)
                if min_num > 0:
                    #print probs[j]
                    probs[j] = probs[j] * ((torch.log(probs[j]) > thres).type(torch.FloatTensor)).cuda()
                    #print torch.sum(probs[j])
                    probs[j] = probs[j] / torch.sum(probs[j]) #re-normalize
                else:
                    print 'min_num is zero, skipping...'
            indices = torch.multinomial(probs, 1)
            all_indices.append(indices)

            embedding = self.decode_embedding(indices)
            if i < (maxlen - 1):
                inputs = torch.cat([embedding, latent[:,i + 1,:].unsqueeze(1)], 2)
        
        max_indices = torch.cat(all_indices, 1)

        return max_indices.data

class RNNAttDecoder(nn.Module): 
    def __init__(self, embed_size, hidden_size, latent_size, vocab, vocab_inv, seq_len, layer_num = 1, dropout_rate = 0):
        super(RNNAttDecoder, self).__init__()
        
        vocab_size = len(vocab)
        self.decode_embedding = nn.Embedding(len(vocab), embed_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.seq_len = seq_len
        self.vocab = vocab
        self.vocab_inv = vocab_inv
        self.layer_num = layer_num
        self.dropout_rate = dropout_rate
        logger.info('building LSTM att decoder, dropout: %f layer_num: %d', dropout_rate, layer_num)

        self.decode_rnn = nn.LSTM(embed_size + hidden_size, hidden_size, layer_num, batch_first = True, dropout = dropout_rate)
        self.dropout_h1 = nn.Dropout(p=dropout_rate)
        self.dropout_i = nn.Dropout(p=dropout_rate)
        
        self.attn_bi = nn.Linear(self.hidden_size * 2 + self.embed_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size + self.embed_size, self.hidden_size)        

        self.outLinear = nn.Linear(hidden_size, vocab_size, bias=True)
        #self.softmax = nn.LogSoftmax()

    def init_hidden_lstm(self, bsz):
        zeros1 = Variable(torch.zeros(self.layer_num, bsz, self.hidden_size)).cuda()
        zeros2 = Variable(torch.zeros(self.layer_num, bsz, self.hidden_size)).cuda()
        return zeros1, zeros2

    def init_hidden_gru(self, bsz):
        zeros1 = Variable(torch.zeros(self.layer_num, bsz, self.hidden_size)).cuda()
        return zeros1

    def forward(self, encoder_outputs, indices, h_state = None): #changed from decode to forward for data parallel
        len_now = indices.size(1)
        batch_size = indices.size(0)
        # batch x hidden
        #print latent.unsqueeze(1).size()
        #moved to main #all_latent = latent.unsqueeze(1).repeat(1, len_now, 1)
        
        #print encoder_outputs.size() #[20, 30, 300]
        """
        # old method debug
        state = self.init_hidden_lstm(batch_size)

        embeddings = self.decode_embedding(indices)
        embeddings = self.dropout_i(embeddings)
        augmented_embeddings = torch.cat([embeddings, latent.repeat(1, len_now, 1)], 2)

        output, state = self.decode_rnn(augmented_embeddings, state)
        
        output = self.dropout_h1(output) 
        # reshape to batch_size*maxlen x nhidden before linear over vocab
        logit = self.outLinear(output)
        #decoded = decoded.view(batch_size, len_now, len(self.vocab))

        return logit
        """
        if h_state == None:
            state = self.init_hidden_lstm(batch_size)
        else:
            state = h_state
        weights_lis = [] 
        logit_lis = [] 
        for i in range(len_now):
            embeddings = self.decode_embedding(indices[:, i].unsqueeze(1))
            embeddings = self.dropout_i(embeddings)
            #print embeddings.size(), state.size()
            #print (state[0].permute(1, 0, 2))[:,0,:].unsqueeze(1).size()
            augmented_embeddings = torch.cat([embeddings, state[0].permute(1, 0, 2)[:,0,:].unsqueeze(1), state[1].permute(1, 0, 2)[:,0,:].unsqueeze(1)], 2) #only extract the first layer in the state
            bi1 = F.relu(self.attn_bi(augmented_embeddings.squeeze(1))).unsqueeze(2) #[20, 300, 1]
            attn_weights = F.softmax(torch.bmm(encoder_outputs, bi1), dim = 1) #[20, 30, 1]
            weights_lis.append(attn_weights.permute(0, 2, 1))
            attn_applied = torch.bmm(encoder_outputs.permute(0, 2, 1), attn_weights).squeeze(2) #[20, 300]
            aug = torch.cat([attn_applied.unsqueeze(1), embeddings], dim = 2)
            output, state = self.decode_rnn(aug, state)

            output = self.dropout_h1(output) 
            # reshape to batch_size*maxlen x nhidden before linear over vocab
            logit = self.outLinear(output)
            #print logit.size() #[20, 1, 10004]
            logit_lis.append(logit)
            #sys.exit(1)
            #decoded = decoded.view(batch_size, len_now, len(self.vocab))
        
        logit_all = torch.cat(logit_lis, dim = 1)
        weights_all = torch.cat(weights_lis, dim = 1)
        return logit_all, weights_all, state
    
    def generate(self, encoder_outputs, maxlen, sample=False):
        batch_size = encoder_outputs.size(0)
        #print 'batch_size', batch_size
        state = self.init_hidden_lstm(batch_size)
        #maxlen = latent.size(1)
        start_symbols = Variable(torch.LongTensor(batch_size, 1)).cuda()
        start_symbols.data.fill_(self.vocab_inv['<s>'])

        embedding = self.decode_embedding(start_symbols)
        embedding = self.dropout_i(embedding)

        all_indices = []
        weights_lis = []
        for i in range(maxlen):
            augmented_embeddings = torch.cat([embedding, state[0].permute(1, 0, 2)[:,0,:].unsqueeze(1), state[1].permute(1, 0, 2)[:,0,:].unsqueeze(1)], 2) #only extract the first layer in the state
            #augmented_embeddings = torch.cat([embedding, state[0].permute(1, 0, 2), state[1].permute(1, 0, 2)], 2)
            #print augmented_embeddings.size()
            bi1 = F.relu(self.attn_bi(augmented_embeddings.squeeze(1))).unsqueeze(2) #[20, 300, 1]
            #print encoder_outputs.size(), bi1.size()
            attn_weights = F.softmax(torch.bmm(encoder_outputs, bi1), dim = 1) #[20, 30, 1]
            weights_lis.append(attn_weights.permute(0, 2, 1))
            attn_applied = torch.bmm(encoder_outputs.permute(0, 2, 1), attn_weights).squeeze(2) #[20, 300]
            aug = torch.cat([attn_applied.unsqueeze(1), embedding], dim = 2)
            output, state = self.decode_rnn(aug, state)

            output = self.dropout_h1(output) 
            overvocab = self.outLinear(output).squeeze(1)
 
            if not sample:
                vals, indices = torch.max(overvocab, 1)
                indices = indices.unsqueeze(1)
            else:
                if torch.max(overvocab).item() > 40:
                    logger.info('WARNING: torch.max(overvocab).item() > 40 (%f) during sampling, will be clamped', torch.max(overvocab).item())
                probs = F.softmax(torch.clamp(overvocab, -40, 40)) #could prevent inf after softmax, which will be bad for multinomail 
                indices = torch.multinomial(probs, 1)
            all_indices.append(indices)

            embedding = self.decode_embedding(indices)
        
        max_indices = torch.cat(all_indices, 1)
        attn_weights = torch.cat(weights_lis, 1)
        return max_indices.data, attn_weights

    def generate_samplemin(self, encoder_outputs, maxlen, thres):
        batch_size = encoder_outputs.size(0)
        #print 'batch_size', batch_size
        state = self.init_hidden_lstm(batch_size)
        #maxlen = latent.size(1)
        start_symbols = Variable(torch.LongTensor(batch_size, 1)).cuda()
        start_symbols.data.fill_(self.vocab_inv['<s>'])

        embedding = self.decode_embedding(start_symbols)
        embedding = self.dropout_i(embedding)

        all_indices = []
        weights_lis = []
        for i in range(maxlen):
            #print state[0].permute(1, 0, 2)
            augmented_embeddings = torch.cat([embedding, state[0].permute(1, 0, 2)[:,0,:].unsqueeze(1), state[1].permute(1, 0, 2)[:,0,:].unsqueeze(1)], 2) #only extract the first layer in the state    
            #augmented_embeddings = torch.cat([embedding, state[0].permute(1, 0, 2), state[1].permute(1, 0, 2)], 2)
            #print augmented_embeddings.size()
            bi1 = F.relu(self.attn_bi(augmented_embeddings.squeeze(1))).unsqueeze(2) #[20, 300, 1]
            #print encoder_outputs.size(), bi1.size()
            attn_weights = F.softmax(torch.bmm(encoder_outputs, bi1), dim = 1) #[20, 30, 1]
            weights_lis.append(attn_weights.permute(0, 2, 1))
            attn_applied = torch.bmm(encoder_outputs.permute(0, 2, 1), attn_weights).squeeze(2) #[20, 300]
            aug = torch.cat([attn_applied.unsqueeze(1), embedding], dim = 2)
            output, state = self.decode_rnn(aug, state)

            output = self.dropout_h1(output) 
            overvocab = self.outLinear(output).squeeze(1)
  
            probs = F.softmax(overvocab)
            for j in range(batch_size):
                ss = 0
                min_num = torch.sum(torch.log(probs[j]) > thres)
                if min_num > 0:
                    #print probs[j]
                    probs[j] = probs[j] * ((torch.log(probs[j]) > thres).type(torch.FloatTensor)).cuda()
                    #print torch.sum(probs[j])
                    probs[j] = probs[j] / torch.sum(probs[j]) #re-normalize
                else:
                    print 'min_num is zero, skipping...'
            indices = torch.multinomial(probs, 1)
            all_indices.append(indices)

            embedding = self.decode_embedding(indices)
        
        max_indices = torch.cat(all_indices, 1)
        attn_weights = torch.cat(weights_lis, 1)
        return max_indices.data, attn_weights

class HighwayMLP(nn.Module):

    def __init__(self,
                 input_size,
                 gate_bias=-2,
                 activation_function=nn.functional.relu,
                 gate_activation=nn.functional.sigmoid): #????? is softmax better?

        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)

class LSTM_onehot_D(nn.Module): 
    def __init__(self, input_size, hidden_size, vocab_inv, dropout_rate = 0, layer_num = 1, final_layer_num = 2):
        super(LSTM_onehot_D, self).__init__()
        logger.info('initing lstm discriminator hidden_size: %d dropout_rate: %f layer_num: %d', hidden_size, dropout_rate, layer_num)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = len(vocab_inv)
        self.vocab_inv = vocab_inv
        self.layer_num = layer_num
        self.final_layer_num = final_layer_num
        self.embedding_1 = nn.Linear(self.vocab_size, self.input_size) #input is oneho
        self.lstm_1 = nn.LSTM(input_size, hidden_size, self.layer_num, dropout = dropout_rate, batch_first = True)
        self.embedding_2 = nn.Linear(self.vocab_size, self.input_size) #input is oneho
        self.lstm_2 = nn.LSTM(input_size, hidden_size, self.layer_num, dropout = dropout_rate, batch_first = True)
        
        self.final_m_list = nn.ModuleList()
        for i in range(self.final_layer_num):
            self.final_m_list.append(nn.Linear(self.hidden_size * 2, self.hidden_size * 2))
            self.final_m_list.append(nn.ReLU())
            #self.final_m_list.append(nn.Dropout(p=dropout_rate))
        self.final_m_list.append(nn.Linear(self.hidden_size * 2, 1))
        self.final_m_list.append(nn.Sigmoid())

        self.final_highway_1 = HighwayMLP(self.hidden_size * 2)
        self.final_highway_2 = HighwayMLP(self.hidden_size * 2)
        self.final_linear = nn.Linear(self.hidden_size * 2, 1)
        self.final_sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax()

    def initHidden(self, batch_size = 1):
        return Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size)).cuda().contiguous(),  Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size)).cuda().contiguous()

    def forward(self, input_idx_1, input_idx_2):
        assert(input_idx_1.size(0) == input_idx_2.size(0))
        bz = input_idx_1.size(0)
        emb_1 = self.embedding_1(input_idx_1)
        output_1, hidden_1 = self.lstm_1(emb_1, self.initHidden(batch_size = bz))
        output_1 = output_1[:, -1, :]
        
        emb_2 = self.embedding_2(input_idx_2)
        output_2, hidden_2 = self.lstm_2(emb_2, self.initHidden(batch_size = bz))
        output_2 = output_2[:, -1, :]
        
        output = torch.cat([output_1, output_2], dim = 1) 
        #for layer in self.final_m_list:
        #    output = layer(output)
        output = self.final_sigmoid(self.final_linear(self.final_highway_2(self.final_highway_1(output))))

        return output

class CNN_KIM_TWOINPUT_D(nn.Module):
    def __init__(self, vocab_inv, args):
        super(CNN_KIM_TWOINPUT_D,self).__init__()
        logger.info('initing cnn, dropout: %s final_sigmoid: %s', str(args['dropout']), str(args['final_sigmoid']))
        self.vocab_inv = vocab_inv
        self.args = args
         
        V = len(self.vocab_inv) #args.embed_num
        D = 300 #args.embed_dim
        #C = 2 #args.class_num
        Ci = 1
        Co = 300 #args.kernel_num
        Ks = [3,4,5,6] #args.kernel_sizes

        self.embed = nn.Linear(V, D)
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1_1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.convs1_2 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args['dropout'])
        self.hidden_size = len(Ks) * Co * 2
        
        self.layer_num = 3
        self.final_layers = []
        for i in range(self.layer_num):
            layer = HighwayMLP(self.hidden_size)
            self.final_layers.append(layer)
            self.add_module("highway"+str(i+1), layer)

        self.final_linear = nn.Linear(self.hidden_size, 1)
        self.have_final_sigmoid = args['final_sigmoid']
        if args['final_sigmoid'] == True:
            self.final_sigmoid = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x1, x2):
        x1 = self.embed(x1) # (N,W,D)
        x1 = x1.unsqueeze(1) # (N,Ci,W,D)
        x1 = [F.relu(conv(x1)).squeeze(3) for conv in self.convs1_1] #[(N,Co,W), ...]*len(Ks)
        x1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x1] #[(N,Co), ...]*len(Ks)
        x1 = torch.cat(x1, 1)

        x2 = self.embed(x2) # (N,W,D)
        x2 = x2.unsqueeze(1) # (N,Ci,W,D)
        x2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs1_2] #[(N,Co,W), ...]*len(Ks)
        x2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x2] #[(N,Co), ...]*len(Ks)
        x2 = torch.cat(x2, 1)
	
        x = torch.cat([x1, x2], 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x) # (N,len(Ks)*Co)
        
        for ll in self.final_layers:
            x = ll(x)
        
        x = self.final_linear(x) # (N,C)

        if self.have_final_sigmoid == True:
            x = self.final_sigmoid(x)

        return x

class LSTMLM_onehot(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size, vocab_inv, dropout_rate = 0, layer_num = 1):
        super(LSTMLM_onehot, self).__init__()
        logger.info('initing rnn dropout_rate: %f layer_num: %d', dropout_rate, layer_num)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = output_size
        self.vocab_inv = vocab_inv
        self.layer_num = layer_num
        self.embedding = nn.Linear(self.vocab_size, self.input_size) #input is onehot
        self.dropout_i = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(input_size, hidden_size, self.layer_num, dropout = dropout_rate, batch_first = True)
        self.o2o = nn.Linear(hidden_size, output_size, bias=True)
        self.dropout_h1 = nn.Dropout(p=dropout_rate)
        self.dropout_h2 = nn.Dropout(p=dropout_rate)
        self.dropout_h3 = nn.Dropout(p=dropout_rate)
        #self.logsoftmax = nn.LogSoftmax()

    def forward(self, input_idx, hidden):
        emb = self.embedding(input_idx)
        emb = self.dropout_i(emb)
        #emb = pack_padded_sequence(emb, list(lengths), batch_first=True)
        output, hidden = self.lstm(emb, hidden)
        #output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.dropout_h1(output).contiguous()
        decoded = self.o2o(output.view(output.size(0) * output.size(1), output.size(2)))
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded, hidden

    def initHidden(self, batch_size = 1):
        return Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size)).cuda().contiguous(),  Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size)).cuda().contiguous()

    def sampleBatch(self, length, bs, full_length = False):
        rnn = self
        input_x = torch.LongTensor(bs, 1).cuda()
        input_x[:] = self.vocab_inv['<s>']
        input_x = Variable(input_x)
        h = rnn.initHidden(batch_size = bs)
        res = [[] for x in range(bs)]
        ends = [False] * bs
        logprobs = [[] for x in range(bs)]
        for k in range(length):
            output, h = rnn(idx2onehot(input_x, len(self.vocab_inv)), h)
            output = output.squeeze()
            output = torch.nn.functional.softmax(output, dim = 1)
            out_dis = output.cpu().data.numpy()
            #print sum(out_dis)
            out_dis = out_dis / 1.0001 #to make it stable
            
            for i in range(bs):
                next_word = np.argmax(np.random.multinomial(1, out_dis[i])) #sampling, not top1
                if full_length == True:
                    while next_word in (self.vocab_inv['</s>'], self.vocab_inv['<pad>'], self.vocab_inv['<s>']):
                        next_word = np.argmax(np.random.multinomial(1, out_dis[i])) #sampling, not top1
                logprobs[i].append(math.log(out_dis[i][next_word]))
                input_x[i] = next_word
                if ends[i] == False:
                    res[i].append(next_word)
                if next_word == self.vocab_inv['</s>']:
                    ends[i] = True

            if np.sum(ends) == bs:
                break
        
        #print 'logprobs[0]', logprobs[0]
         
        for i in range(bs): #clean it for seq2seq input
            if res[i][-1] == self.vocab_inv['</s>']: res[i] = res[i][:-1]    
            if len(res[i]) == length: 
                res[i][-1] = self.vocab_inv['<eou>']
            else:
                res[i].append(self.vocab_inv['<eou>'])
            if len(res[i]) > length: res[i] = res[i][-length:]
            if len(res[i]) < length: res[i] = [self.vocab_inv['<pad>']] * (length - len(res[i])) + res[i]
            
        return res
    
    def calMeanLogp(self, ss, onehotv, mode = 'input_eou', train_flag = True): 
        assert(mode == 'input_eou')
        
        rnn = self
        if train_flag == True:
            rnn.train()
        else:
            rnn.eval()
        vocab_inv = self.vocab_inv
        
        ss = copy.deepcopy(ss)
         
        bz = len(ss)
        length = len(ss[0])

        for i in range(bz):
            assert(ss[i][0] != vocab_inv['<pad>'] and ss[i][-1] == vocab_inv['<eou>'])
            ss[i] = [vocab_inv['<s>']] + ss[i][:-1] + [vocab_inv['</s>']]
        
        input_ohv = torch.cat([idx2onehot(torch.LongTensor([vocab_inv['<s>']] * bz).view(-1, 1).cuda(), len(vocab_inv)), onehotv[:, :-1, :]], dim = 1)
        target_ohv = torch.cat([onehotv[:, :-1, :], idx2onehot(torch.LongTensor([vocab_inv['</s>']] * bz).view(-1, 1).cuda(), len(vocab_inv))], dim = 1)

        w_logit_rnn, _ = rnn(input_ohv, rnn.initHidden(bz))
        logpdf_all = torch.nn.functional.log_softmax(w_logit_rnn, dim = 2)
        logpdf = logpdf_all * target_ohv
        
        w_loss = torch.sum(logpdf, dim = 2)
        
        return (torch.sum(w_loss, dim = 1) / length)

def encoder_decoder_forward(src_mb, tgt_mb, tgt_len, m_dict, config, aux_return = None):
    MT, HIDDEN_SIZE, vocab, LAYER_NUM = config['MT'], config['HIDDEN_SIZE'], config['vocab'], config['LAYER_NUM']
    m_encode_w_rnn, m_decode_w_rnn, m_embed = m_dict['m_encode_w_rnn'], m_dict['m_decode_w_rnn'], m_dict['m_embed']
    bz = src_mb.size(0)
    src_inputv = Variable(src_mb).cuda() 
    tgt_inputv = Variable(tgt_mb[:, :-1]).cuda() 
    tgt_targetv = Variable(tgt_mb[:, 1:]).cuda()
    tgt_mask = Variable(torch.FloatTensor([([1] * l + [0] * (tgt_inputv.size(1) - l)) for l in tgt_len]).cuda(), requires_grad = False)

    output, _ = m_encode_w_rnn(m_embed(idx2onehot(src_inputv, len(vocab))).permute(1, 0, 2), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM)) #for parallel!
    e_output = output.permute(1, 0, 2) #for parallel!
    
    if MT == 'latent':
        latent = e_output[:, -1, :].squeeze(1)
        latent = latent.unsqueeze(1).repeat(1, tgt_inputv.size(1), 1)
        w_logit_rnn = m_decode_w_rnn(latent, tgt_inputv, tgt_len) #change from decode to forward for data parallel
    if MT == 'attention':
        w_logit_rnn, attn_weights, _ = m_decode_w_rnn(e_output, tgt_inputv)
    
    if aux_return != None:
        aux_return['w_logit_rnn'] = w_logit_rnn

    flat_output = w_logit_rnn.view(-1, len(vocab))
    flat_target = tgt_targetv.contiguous().view(-1)
    flat_logpdf = lib_pdf.logsoftmax_idxselect(flat_output, flat_target)
    batch_logpdf = flat_logpdf.view(bz, -1) * tgt_mask
     
    return batch_logpdf

def beam_search(src_mb, max_len, beam_size, m_dict, config):
    MT, HIDDEN_SIZE, vocab, vocab_inv, LAYER_NUM = config['MT'], config['HIDDEN_SIZE'], config['vocab'], config['vocab_inv'], config['LAYER_NUM']
    m_encode_w_rnn, m_decode_w_rnn, m_embed = m_dict['m_encode_w_rnn'], m_dict['m_decode_w_rnn'], m_dict['m_embed']
    for m in m_dict: m_dict[m].eval()
    bz = src_mb.size(0)
    assert(bz == 1 and MT == 'attention')
    src_inputv = Variable(src_mb).cuda() 
    output, _ = m_encode_w_rnn(m_embed(idx2onehot(src_inputv, len(vocab))).permute(1, 0, 2), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM)) #for parallel!
    e_output = output.permute(1, 0, 2) #now batch first, for parallel!
   
    ini = {
        'll': 0,
        'll_w': [],
        'w_lis': [],
        'state': init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM),
        'wid_now': vocab_inv['<s>'], #this is included in the ll score now
        'finished': False,
    }
    beam = [ini] 
    for l in range(max_len):
        state_cat = (torch.cat([m['state'][0] for m in beam], dim = 1), torch.cat([m['state'][1] for m in beam], dim = 1))
        idx_cat = torch.LongTensor([m['wid_now'] for m in beam]).cuda().view(len(beam), 1)
        if MT == 'attention':
            n_logits, _, n_state = m_decode_w_rnn.forward(e_output.expand(len(beam), e_output.size(1), e_output.size(2)), 
                idx_cat, h_state = state_cat)
        beam_next = []
        for i in range(len(beam)):
            m = beam[i]
            if m['finished'] == True:
                beam_next.append(m); continue
            distro = F.log_softmax(n_logits[i, 0, :], dim = 0).view(-1)
            sval, sidx = torch.sort(distro, descending = True)
            for j in range(beam_size):
                new_m = {
                    'll': m['ll'] + sval[j].item(),
                    'll_w': m['ll_w'] + [sval[j]],
                    'w_lis': m['w_lis'] + [vocab[sidx[j].item()]],
                    'state': (n_state[0][:, i, :].unsqueeze(1), n_state[1][:, i, :].unsqueeze(1)),
                    'wid_now': sidx[j].item(),
                    'finished': sidx[j].item() == vocab_inv['</s>'],
                }
                beam_next.append(new_m)
        beam_next = sorted(beam_next, key = lambda m: m['ll'], reverse = True)
        beam_next = beam_next[:beam_size]
        beam = beam_next
    return beam

def get_samples(batches, ty, m_dict, config):
    MT, HIDDEN_SIZE, vocab, vocab_inv, LAYER_NUM, TGT_SEQ_LEN, BEAM_SIZE = config['MT'], config['HIDDEN_SIZE'], config['vocab'], config['vocab_inv'], config['LAYER_NUM'], config['TGT_SEQ_LEN'], config['BEAM_SIZE']
    m_encode_w_rnn, m_decode_w_rnn, m_embed = m_dict['m_encode_w_rnn'], m_dict['m_decode_w_rnn'], m_dict['m_embed']
    for m in m_dict: m_dict[m].eval()
    #logger.info('ty: %s', ty)
    assert(ty == 'max' or ty == 'sample' or ty == 'beam') #for the sake of current use
    ref_lis, sample_lis, src_lis, raw_samples_lis, b_co = [], [], [], [], 0
    extend_num = 1
    for src_mb, tgt_mb, tgt_len, src_w, tgt_w in batches: 
        b_co = b_co + 1
        bz = src_mb.size(0)
        for i in range(bz):
            ref_lis.append(tgt_w[i][1:])
            src_lis.append(src_w[i][0:])
        
        if ty == 'beam':
            for i in range(bz):
                beam = beam_search(src_mb[i].unsqueeze(0), TGT_SEQ_LEN, BEAM_SIZE, m_dict, config)
                sample_lis.append(beam[0]['w_lis'])
                #print beam[0]['w_lis']
                raw_samples_lis.append([vocab_inv[w] for w in beam[0]['w_lis']])                  
        else:
            input_src = Variable(src_mb).cuda()
            input_src = torch.cat([input_src] * extend_num, dim = 0)
            output, _ = m_encode_w_rnn(m_embed(idx2onehot(input_src, len(vocab))).permute(1,0,2), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM))
            output = output.permute(1, 0, 2)
            maxlen = config['TGT_SEQ_LEN']
            
            if MT == 'latent': latent = output[:, -1, :].unsqueeze(1).repeat(1, tgt_mb.size(1), 1)
            if MT == 'attention': latent = output
            
            if ty == 'sample_min':
                if MT == 'attention': samples, _ = m_decode_w_rnn.generate_samplemin(latent, maxlen, NORMAL_WORD_AVG_LOSS)
                if MT == 'latent': samples = m_decode_w_rnn.generate_samplemin(latent, maxlen, NORMAL_WORD_AVG_LOSS)
            else:
                samples, _ = m_decode_w_rnn.generate(latent, maxlen, sample = (False if ty == 'max' else True))
            raw_samples_lis.append(samples)
            samples = samples.cpu().numpy().tolist()
            for i in range(bz):
                sen = [vocab[k] for k in samples[i]]
                if '</s>' in sen:
                    end_idx = sen.index('</s>')
                    sen = sen[:(end_idx + 1)]
                sample_lis.append(sen)
        
    res = {
        'src_lis': src_lis,
        'ref_lis': ref_lis,
        'sample_lis': sample_lis,
        'raw_sample_lis': raw_samples_lis,
    }

    return res
