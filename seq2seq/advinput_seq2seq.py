import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle
import logging
import os, sys 
import random
import models
from myutil import *
import lib_pdf
import copy

logger = logging.getLogger()

def adv_model_forward(input_onehot_v, input_idx_lis, target_mb, m_dict, adv_config, decay_loss = True, do_train = True): 
    bz = input_onehot_v.size(0)
    globals().update(adv_config)
    globals().update(m_dict)
    m_list = [m_embed, m_encode_w_rnn, ADV_I_LM, m_decode_w_rnn, m_embed_dp, m_encode_w_rnn_dp, m_decode_w_rnn_dp]
    if do_train == True: 
        for m in m_list: 
            if m != None: m.train()
    if do_train == False: 
        for m in m_list: 
            if m != None: m.eval()
    
    tgt_idx, tgt_w, tgt_len = target_mb
    tgt_inputv = tgt_idx[:, :-1].cuda()
    tgt_targetv = tgt_idx[:, 1:].cuda()
 
    output, _ = m_encode_w_rnn_dp(m_embed_dp(input_onehot_v).permute(1,0,2), init_lstm_hidden(bz, HIDDEN_SIZE))
    output = output.permute(1,0,2)
    latent = output[:, -1, :].unsqueeze(1).repeat(1, tgt_targetv.size(1), 1)
    
    if MT == 'latent': w_logit_rnn = m_decode_w_rnn_dp(latent, tgt_inputv, tgt_len)
    if MT == 'attention': w_logit_rnn, attn_weights, _ = m_decode_w_rnn_dp(output, tgt_inputv)

    flat_output = w_logit_rnn.view(-1, len(vocab))
    flat_target = tgt_targetv.contiguous().view(-1)
    flat_logpdf = lib_pdf.logsoftmax_idxselect(flat_output, flat_target)
    batch_logpdf = flat_logpdf.view(bz, -1)
    #print 'tgt_targetv[8,9,10]', tgt_targetv[8], tgt_targetv[9], tgt_targetv[10]
    #print tgt_w[8], tgt_w[9]

    tgt_mask = Variable(torch.FloatTensor([([1] * l + [0] * (tgt_inputv.size(1) - l)) for l in tgt_len]).cuda(), requires_grad = False)
    w_logit_pred = torch.max(w_logit_rnn, dim = 2)[1]
    #if decay_loss == True:
    #print 'decay_weight:', GIBBSENUM_DECAYLOSS_WEIGHT 
    o_mins = []
    for i in range(bz):
        minn = 1000
        for j in range(tgt_len[i]):
            if ADV_CARE_MODE == 'max' and decay_loss == True and w_logit_pred[i][j] == tgt_targetv[i][j]:
                tgt_mask[i][j] = min(tgt_mask[i][j], 0.01) #debug! #why not just zero?
            if ADV_CARE_MODE == 'sample_min' and decay_loss == True and batch_logpdf[i][j].item() >= NORMAL_WORD_AVG_LOSS:
                tgt_mask[i][j] = min(tgt_mask[i][j], 0.01)
            if batch_logpdf[i][j].item() < minn:
                minn = batch_logpdf[i][j].item()
            #if tgt_targetv[i][j] != 0:
                #logger.info('setting to 0.001: |%s|', vocab[tgt_targetv[i][j]])
        o_mins.append(minn)
    weight_batch_logpdf = batch_logpdf * tgt_mask
    sen_loss_rnn = torch.sum(weight_batch_logpdf, dim = 1)
    w_loss_rnn = sen_loss_rnn / torch.FloatTensor(tgt_len).cuda() #
    
    if ADV_I_LM_FLAG == True:
        i_w_loss_lm = ADV_I_LM.calMeanLogp(input_idx_lis, input_onehot_v, mode = 'input_eou', train_flag = do_train)
    else:
        i_w_loss_lm = None
    o_scal = Variable(torch.ones(bz).cuda(), requires_grad = False)
    i_scal = Variable(torch.ones(bz).cuda(), requires_grad = False)
    #print i_w_loss_lm.size()
    for k in range(bz):
        if ADV_I_LM_FLAG == True and decay_loss == True:
            if i_w_loss_lm[k].item() > GE_I_WORD_AVG_LOSS: i_scal[k] = 0.01
        if ADV_CARE_MODE == 'sample_avg' and w_loss_rnn[k].item() > NORMAL_WORD_AVG_LOSS: o_scal[k] = 0.01 
        if ADV_CARE_MODE == 'sample_min' and o_mins[k] > NORMAL_WORD_AVG_LOSS: o_scal[k] = 0.01

    if ADV_I_LM_FLAG == False:
        loss_combine = w_loss_rnn
    else:
        loss_combine = w_loss_rnn * o_scal + i_w_loss_lm * i_scal * GIBBSENUM_I_LM_LAMBDA

    return w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf

def adv_gibbs_enum_mb(target_mb, m_dict, adv_config):
    tgt_mb, b_w, b_len = target_mb
    bz = tgt_mb.size(0)
    #vocab, vocab_inv, SRC_SEQ_LEN, TGT_SEQ_LEN = adv_config['vocab'], adv_config['vocab_inv'], adv_config['SRC_SEQ_LEN'], adv_config['TGT_SEQ_LEN']
    #MT, HIDDEN_SIZE, ADV_CARE_MODE = adv_config['MT'], adv_config['HIDDEN_SIZE'], adv_config['ADV_CARE_MODE']
    globals().update(adv_config)
    m_encode_w_rnn, m_decode_w_rnn, m_embed = m_dict['m_encode_w_rnn'], m_dict['m_decode_w_rnn'], m_dict['m_embed']
     
    best_idx_glo = [([-1 for k in range(SRC_SEQ_LEN - 1)] + [vocab_inv['<eou>']]) for kk in range(bz)]
    best_loss_glo = [-1000 for k in range(bz)]
    
    for random_iter in range(GIBBSENUM_RANDOM_TIME):
        if ADV_I_LM_FLAG == False:
            best_idx = []
            for kk in range(bz):
                ini_seq = [vocab_inv['<eou>']]
                for k in range(SRC_SEQ_LEN - 1):
                    ini_seq = [random.randint(5, len(vocab) - 1)] + ini_seq
                best_idx.append(ini_seq)
            #best_idx = [([vocab_inv['e'] for k in range(SRC_SEQ_LEN - 1)] + [vocab_inv['<eou>']]) for kk in range(bz)]
        else:
            best_idx = ADV_I_LM.sampleBatch(SRC_SEQ_LEN, bz, full_length = True)
        onehot_v = torch.zeros(bz, SRC_SEQ_LEN, len(vocab)).cuda().detach()
        for i in range(bz): 
            for pos in range(SRC_SEQ_LEN):
                onehot_v[i, pos, best_idx[i][pos]] = 1
 
        best_loss = [-1000 for k in range(bz)]
        w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = adv_model_forward(onehot_v, best_idx, target_mb, m_dict, adv_config, decay_loss = True, do_train = False)
        #print w_loss_rnn
        w_loss_lis = [[w_loss_rnn.mean().item()], [np.mean(o_mins)]]; # print w_loss_lis #for figure drawing, not actually used, 0 for avg, 1 for min
        #print 'w_loss_lis[0] now:', w_loss_lis[0]
        w_loss_now = [[-1000] * bz, [-1000] * bz]
        iter_co = 0
        while True:
            last_mean_best = np.mean(best_loss)
            for pos in range(SRC_SEQ_LEN - 1):
                cur_idx = copy.deepcopy(best_idx)            
                if GIBBSENUM_E_NUM != -1:
                    onehot_opt = Variable(onehot_v, requires_grad = True) 
                    #print onehot_opt.grad[9, pos, :10] #gradeint wrong!!
                    w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = adv_model_forward(onehot_opt, best_idx, target_mb, m_dict, adv_config, decay_loss = True, do_train = True)
                    (-loss_combine).mean().backward()
                    sval, sidx = torch.sort(onehot_opt.grad[:, pos, :], dim = 1, descending = False) #descending=False! Remember that gradient is minused to the parameter!
                    sidx = sidx.cpu().numpy().tolist()
                    tn = GIBBSENUM_E_NUM
                else:
                    sidx = [range(len(vocab)) for kkk in range(bz)]
                    tn = len(vocab) - 4 #minus the <s>, </s>, and <pad>, <eou>
                idx_try = [-1 for kkk in range(bz)]
                for try_num in range(tn):
                    onehot_v[:, pos, :] = 0
                    for i in range(bz):
                        idx_try[i] += 1
                        while vocab[sidx[i][idx_try[i]]] in ('<s>', "</s>", "<pad>", "<eou>"): idx_try[i] += 1
                        onehot_v[i, pos, sidx[i][idx_try[i]]] = 1        
                        cur_idx[i][pos] = sidx[i][idx_try[i]]
                        """
                        for w_idx in range(len(vocab_inv)):
                            if not vocab[w_idx] in ('<s>', "</s>", "<pad>", "<eou>"):
                                onehot_v[:, pos, :] = 0
                                onehot_v[:, pos, w_idx] = 1
                            else:
                                continue
                        """
                    w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = adv_model_forward(onehot_v, cur_idx, target_mb, m_dict, adv_config, decay_loss = True, do_train = False)
                    for i in range(bz):
                        if loss_combine[i].item() > best_loss[i]:
                            w_loss_now[0][i] = w_loss_rnn[i].item(); w_loss_now[1][i] = o_mins[i];
                            best_loss[i] = loss_combine[i].item()
                            best_idx[i][pos] = sidx[i][idx_try[i]] #w_widx
                onehot_v[:, pos, :] = 0
                for i in range(bz): onehot_v[i, pos, best_idx[i][pos]] = 1
                #print best_loss[0], best_loss[1], best_loss[2], best_loss[3]
                w_loss_lis[0].append(np.mean(w_loss_now[0])); w_loss_lis[1].append(np.mean(w_loss_now[1])); #print 'w_loss_lis[0][-1]', w_loss_lis[0][-1]
            iter_co += 1;
            if np.mean(best_loss) > last_mean_best and iter_co <= GIBBSENUM_MAX_ITER_CO:
                last_mean_best = np.mean(best_loss)
            else: break
        #print w_loss_lis[0]; #sys.exit(1)
        for i in range(bz):
            if best_loss[i] > best_loss_glo[i]: 
                best_loss_glo[i] = best_loss[i]; best_idx_glo[i] = best_idx[i]
    
    onehot_v_glo = torch.zeros(bz, SRC_SEQ_LEN, len(vocab)).cuda().detach()
    for i in range(bz): 
        for pos in range(SRC_SEQ_LEN):
            onehot_v_glo[i, pos, best_idx_glo[i][pos]] = 1  
    w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = adv_model_forward(onehot_v_glo, best_idx_glo, target_mb, m_dict, adv_config, decay_loss = False, do_train = False) #here we want the true loss so decay_loss=False
    
    stat_mb = []
    for i in range(bz): 
        best_loss_glo[i] = loss_combine[i].item() #best_loss_glo could be weighted
        stat_mb.append({
            'gibbs_enum_best_loss': best_loss_glo[i],
            'gibbs_enum_best_sen_loss': sen_loss_rnn[i].item(),
            'gibbs_enum_best_w_loss': w_loss_rnn[i].item(),
            'gibbs_enum_best_i_w_loss': i_w_loss_lm[i].item() if ADV_I_LM_FLAG == True else None,
            'gibbs_enum_best_o_min_loss': o_mins[i],
            'gibbs_enum_best_o_logpdf': batch_logpdf[i].detach().cpu().numpy().tolist()})
    
    return best_idx_glo, stat_mb

def get_adv_seq2seq_mb(target_mb, ty, m_dict, adv_config):
    assert(ty == 'random' or ty == 'gibbs_enum')
    tgt_idx, tgt_w, tgt_len = target_mb
    bz = tgt_idx.size(0)
    globals().update(adv_config)
    
    if ty == 'random':
        #TODO: add I_LM sampler
        assert(ADV_I_LM_FLAG == False)
        mb_src = []
        src_lis = []
        for kk in range(bz):
            mb_src.append([vocab_inv['<eou>']])
            src_lis.append(['<eou>'])
            for k in range(SRC_SEQ_LEN - 1):
                w_id = random.randint(5, len(vocab) - 1)
                mb_src[kk] = [w_id] + mb_src[kk]
                src_lis[kk] = [vocab[w_id]] + src_lis[kk]
        mb_src = torch.LongTensor(mb_src) 
        return mb_src, tgt_idx, tgt_len, src_lis, tgt_w

    if ty == 'gibbs_enum':
        adv_idx, stat_mb = adv_gibbs_enum_mb(target_mb, m_dict, adv_config)
        mb_src = torch.LongTensor(adv_idx)
        src_lis = []
        for i in range(bz):
            src_lis.append([vocab[kk] for kk in adv_idx[i]])
        return mb_src, tgt_idx, tgt_len, src_lis, tgt_w



