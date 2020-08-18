import torch 
import numpy as np
from torch.autograd import Variable
import os, sys
import logging
import math
from myutil import *
import nltk
import models
import lib_pdf
import torch.nn.functional as F
#import advidx_draw
import hashlib
import random
import copy

logger = logging.getLogger()

def build_lis_tgt_w(DATA_SET, vocab_inv, tgt_fn_lis):
    logger.info('building lis_tgt_w for data_set %s...', DATA_SET)
    lis_tgt_w = {}
    for fn in tgt_fn_lis:
        lis_r = open('../adv_lists/' + DATA_SET + '/' + fn, 'r').readlines()
        lis_tgt_w[fn] = []
        for l in lis_r:
            l = l.strip()
            if (len(l.split()) <= 2): continue
            discard = False
            for w in l.strip().split():
                if not w in vocab_inv:
                    discard = True
                    break
            if discard == True:
                logger.info('[%s] in %s discarded because of oov', l, fn)
                continue
            lis_tgt_w[fn].append({'str':l.strip(), 'co':0, 'adv_input_str':None, 'substr_full':None, 'substr_co':0, 'substr_adv_input_str':None, 'fn':fn, 'tgt_len': (len(l.strip().split()) - 1)})
        #lis_tgt_w[fn] = [{'str':l.strip(), 'co':0, 'adv_input_str':None, 'substr_full':None, 'substr_co':0, 'substr_adv_input_str':None, 'fn':fn, 'tgt_len': (len(l.strip().split()) - 1)} for l in lis_r if (len(l.split()) > 2)]
        logger.info('fn: %s len: %d', fn, len(lis_tgt_w[fn]))
    return lis_tgt_w

def adv_greedyflip(MODEL_TYPE, m_dict, HIDDEN_SIZE, DATA_SET, tgt_w_data_fn, run_fn_lis, bz, vocab, vocab_inv, SRC_SEQ_LEN, TGT_SEQ_LEN, save_dir, NORMAL_WORD_AVG_LOSS, GIBBSENUM_DECAYLOSS_WEIGHT, GIBBSENUM_E_NUM, GIBBSENUM_RANDOM_TIME, GREEDYFLIP_NUM, ENUM_START_LENGTH = 1, ENUM_END_LENGTH = 100, GE_I_DO = False, GE_I_WORD_AVG_LOSS = None, GE_I_LAMBDA = None, GE_I_LM = None, GE_I_SAMPLE_LOSSDECAY = False, GE_MAX_ITER_CO = 50, log_fn = '', LOG_SUC = ''):  
    m_embed, m_encode_w_rnn, m_decode_w_rnn = m_dict['m_embed'], m_dict['m_encode_w_rnn'], m_dict['m_decode_w_rnn']
    m_embed_dp, m_encode_w_rnn_dp, m_decode_w_rnn_dp = m_dict['m_embed_dp'], m_dict['m_encode_w_rnn_dp'], m_dict['m_decode_w_rnn_dp']
    
    if tgt_w_data_fn != None:
        logger.info('loading data lis_tgt_w from %s', tgt_w_data_fn)
        lis_tgt_w = torch.load(tgt_w_data_fn)
    else:
        logger.info('tgt_w_data_fn is None, building an empty one, but note that no enumall_information will be aviable')
        lis_tgt_w = build_lis_tgt_w(DATA_SET, vocab_inv, run_fn_lis)  

    """
    lis_tgt_w = {}
    for fn in target_fn_lis:
        lis_r = open('../adv_lists/' + DATA_SET + '/' + fn, 'r').readlines()
        lis_tgt_w[fn] = [{'str':l.strip(), 'co':0, 'adv_input_str':None, 'fn':fn} for l in lis_r if (len(l.split()) > 2)]
        logger.info('fn: %s len: %d', fn, len(lis_tgt_w[fn])) 
    """
    
    logger.info('GREEDYFLIP_NUM: %d', GREEDYFLIP_NUM)

    if GE_I_DO == True:
        logger.info('GE_I_DO is true!! adding input lm score, also initializing from lm samples')
        logger.info('GE_I_LAMBDA: %f GE_I_SAMPLE_LOSSDECAY: %s', GE_I_LAMBDA, str(GE_I_SAMPLE_LOSSDECAY))  
    else:
        logger.info('GE_I_DO is false, setting ge_i_lambda to zero')
        GE_I_LAMBDA = 0
    logger.info('RANDOM_TIME: %d E_NUM: %d MAX_ITER_CO: %d', GIBBSENUM_RANDOM_TIME, GIBBSENUM_E_NUM, GE_MAX_ITER_CO)
    
    def greedyflip_attack_mb(mb, care_mode = ''):
        assert(care_mode in ('max', 'sample_min', 'sample_avg'))
        bz = len(mb)
        lis_tgt_w = [m['str'].split() for m in mb]
        tgt_len = [len(l) - 1 for l in lis_tgt_w]
        max_len = max(tgt_len)
        lis_tgt_w = [l + ['<pad>'] * (max_len + 1 - len(l)) for l in lis_tgt_w]
        lis_tgt_idx = [[(vocab_inv[w] if w in vocab_inv else vocab_inv['<unk>'] ) for w in l] for l in lis_tgt_w]
        tgt_mb = Variable(torch.LongTensor(lis_tgt_idx).cuda())
        tgt_inputv = tgt_mb[:, :-1]
        tgt_targetv = tgt_mb[:, 1:]
       
        def model_forward(onehot_v, idx_lis, bz, decay_loss = True, do_train = True): 
            m_list = [m_embed, m_encode_w_rnn, GE_I_LM, m_decode_w_rnn, m_embed_dp, m_encode_w_rnn_dp, m_decode_w_rnn_dp]
            if do_train == True: 
                for m in m_list: 
                    if m != None: m.train()
            if do_train == False: 
                for m in m_list: 
                    if m != None: m.eval()

            output, _ = m_encode_w_rnn_dp(m_embed_dp(onehot_v).permute(1,0,2), init_lstm_hidden(bz, HIDDEN_SIZE))
            output = output.permute(1,0,2)
            latent = output[:, -1, :].unsqueeze(1).repeat(1, tgt_targetv.size(1), 1)
            
            if MODEL_TYPE == 'latent': w_logit_rnn = m_decode_w_rnn_dp(latent, tgt_inputv, tgt_len)
            if MODEL_TYPE == 'attention': w_logit_rnn, attn_weights = m_decode_w_rnn_dp(output, tgt_inputv, tgt_len)

            flat_output = w_logit_rnn.view(-1, len(vocab))
            flat_target = tgt_targetv.contiguous().view(-1)
            flat_logpdf = lib_pdf.logsoftmax_idxselect(flat_output, flat_target)
            batch_logpdf = flat_logpdf.view(bz, -1)

            tgt_mask = Variable(torch.FloatTensor([([1] * l + [0] * (tgt_inputv.size(1) - l)) for l in tgt_len]).cuda(), requires_grad = False)
            w_logit_pred = torch.max(w_logit_rnn, dim = 2)[1]
            #if decay_loss == True:
            #print 'decay_weight:', GIBBSENUM_DECAYLOSS_WEIGHT 
            o_mins = []
            for i in range(bz):
                minn = 1000
                for j in range(tgt_len[i]):
                    if care_mode == 'max' and decay_loss == True and w_logit_pred[i][j] == tgt_targetv[i][j]:
                        tgt_mask[i][j] = min(tgt_mask[i][j], GIBBSENUM_DECAYLOSS_WEIGHT) #debug! #why not just zero?
                    if care_mode == 'sample_min' and decay_loss == True and batch_logpdf[i][j].item() >= NORMAL_WORD_AVG_LOSS:
                        tgt_mask[i][j] = min(tgt_mask[i][j], 0.01)
                    if batch_logpdf[i][j].item() < minn:
                        minn = batch_logpdf[i][j].item()
                    #if tgt_targetv[i][j] != 0:
                        #logger.info('setting to 0.001: |%s|', vocab[tgt_targetv[i][j]])
                o_mins.append(minn)
            weight_batch_logpdf = batch_logpdf * tgt_mask
            sen_loss_rnn = torch.sum(weight_batch_logpdf, dim = 1)
            w_loss_rnn = sen_loss_rnn / torch.FloatTensor(tgt_len).cuda() #
            
            if GE_I_DO == True:
                i_w_loss_lm = GE_I_LM.calMeanLogp(idx_lis, onehot_v, mode = 'input_eou', train_flag = True)
            else:
                i_w_loss_lm = None
            o_scal = Variable(torch.ones(bz).cuda(), requires_grad = False)
            i_scal = Variable(torch.ones(bz).cuda(), requires_grad = False)
            #print i_w_loss_lm.size()
            for k in range(bz):
                if GE_I_DO == True and GE_I_SAMPLE_LOSSDECAY == True:
                    if i_w_loss_lm[k].item() > GE_I_WORD_AVG_LOSS: i_scal[k] = 0.01
                if care_mode == 'sample_avg' and w_loss_rnn[k].item() > NORMAL_WORD_AVG_LOSS: o_scal[k] = 0.01 #debug!!
                if care_mode == 'sample_min' and o_mins[k] > NORMAL_WORD_AVG_LOSS: o_scal[k] = 0.01
        
            if GE_I_DO == False:
                loss_combine = w_loss_rnn
            else:
                loss_combine = w_loss_rnn * o_scal + i_w_loss_lm * i_scal * GE_I_LAMBDA

            return w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf
           
        best_idx_glo = [([-1 for k in range(SRC_SEQ_LEN - 1)] + [vocab_inv['<eou>']]) for kk in range(bz)]
        best_loss_glo = [-1000 for k in range(bz)]
        
        for random_iter in range(GIBBSENUM_RANDOM_TIME):
            if GE_I_DO == False:
                cur_idx = []
                for kk in range(bz):
                    ini_seq = [vocab_inv['<eou>']]
                    for k in range(SRC_SEQ_LEN - 1):
                        ini_seq = [random.randint(5, len(vocab) - 1)] + ini_seq
                    cur_idx.append(ini_seq)
                #best_idx = [([vocab_inv['e'] for k in range(SRC_SEQ_LEN - 1)] + [vocab_inv['<eou>']]) for kk in range(bz)]
            else:
                cur_idx = GE_I_LM.sampleBatch(SRC_SEQ_LEN, bz, full_length = True)
            onehot_v = torch.zeros(bz, SRC_SEQ_LEN, len(vocab)).cuda().detach()
            for i in range(bz): 
                for pos in range(SRC_SEQ_LEN):
                    onehot_v[i, pos, cur_idx[i][pos]] = 1
            #The two important state variables are cur_idx and onehot_v
            
            best_loss = [-1000 for k in range(bz)]
            best_idx = copy.deepcopy(cur_idx)            
            w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(onehot_v, cur_idx, bz, decay_loss = True, do_train = False)
            print w_loss_rnn
            w_loss_rnn_lis = [torch.mean(w_loss_rnn).detach().item()] #for figure drawing
            print 'first:', w_loss_rnn_lis
            w_loss_now = [[-1000] * bz, [-1000] * bz]
            iter_co = 0
            while True:
                last_mean_best = np.mean(best_loss)
                #for pos in range(SRC_SEQ_LEN - 1): #no more pos!
                onehot_opt = Variable(onehot_v, requires_grad = True) 
                w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(onehot_opt, cur_idx, bz, decay_loss = True, do_train = True)
                (loss_combine).mean().backward() #the loss here is actually not loss, but the log-likelihood(objective)
                oh_grad = onehot_opt.grad
                
                at_least_one_change = False 
                for i in range(bz): #this is the important greedyglip step!
                    for j in range(SRC_SEQ_LEN - 1):
                        oh_grad[i, j] = oh_grad[i, j] - oh_grad[i, j, cur_idx[i][j]] #this is the first-order approximation of an index change
                    sval, sidx = torch.sort(oh_grad[i, :].view(-1), descending = True) #descending=True! Remember that gradient is in the ascending direction.
                    changed = [False for kk in range(SRC_SEQ_LEN)]
                    try_id = 0
                    #TODO: change the order
                    for k in range(GREEDYFLIP_NUM):
                        while try_id < len(vocab) * SRC_SEQ_LEN:
                            try_pos, try_wid = sidx[try_id] / len(vocab), sidx[try_id] % len(vocab)
                            try_id += 1
                            if (changed[try_pos] == True) or (try_pos == (SRC_SEQ_LEN - 1)) or (vocab[try_wid] in ('<s>', "</s>", "<pad>", "<eou>")): continue
                            if sval[try_id] <= 0: break #we are finding "ascent" direction
                            changed[try_pos] = True
                            at_least_one_change = True
                            assert(onehot_v[i, try_pos, cur_idx[i][try_pos]] == 1)
                            onehot_v[i, try_pos, cur_idx[i][try_pos]] = 0
                            cur_idx[i][try_pos] = try_wid
                            onehot_v[i, try_pos, try_wid] = 1
                            #logger.info('changing... sidx[try_id]: %d sval: %f', sidx[try_id], sval[try_id])
                            break 
                if at_least_one_change == False: break
                w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(onehot_v, cur_idx, bz, decay_loss = True, do_train = False)
                print torch.mean(loss_combine)
                for i in range(bz):
                    if loss_combine[i].item() > best_loss[i]:
                        best_loss[i] = loss_combine[i].item()
                        best_idx[i] = copy.deepcopy(cur_idx[i])
                w_loss_rnn_lis.append(torch.mean(w_loss_rnn).detach().item())
                #logger.info('iter_co: %d cur_best_loss: %f last_mean_best: %f', iter_co, np.mean(best_loss), last_mean_best)
                last_mean_best = np.mean(best_loss) 
                iter_co += 1; 
                if iter_co > GE_MAX_ITER_CO * 5: break
            print 'w_loss_rnn_lis:', str(w_loss_rnn_lis); sys.exit(1)
            for i in range(bz): #in the end the random for loop
                if best_loss[i] > best_loss_glo[i]: 
                    best_loss_glo[i] = best_loss[i]; best_idx_glo[i] = best_idx[i]
             
        onehot_v_glo = torch.zeros(bz, SRC_SEQ_LEN, len(vocab)).cuda().detach()
        for i in range(bz): 
            for pos in range(SRC_SEQ_LEN):
                onehot_v_glo[i, pos, best_idx_glo[i][pos]] = 1  
        w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(onehot_v_glo, best_idx_glo, bz, decay_loss = False, do_train = False) #here we want the true loss so decay_loss=False
         
        for i in range(bz): 
            best_loss_glo[i] = loss_combine[i].item() #best_loss_glo could be weighted
            mb[i]['gf_best_loss'] = best_loss_glo[i]
            mb[i]['gf_best_sen_loss'] = sen_loss_rnn[i].item()
            mb[i]['gf_best_w_loss'] = w_loss_rnn[i].item()
            mb[i]['gf_best_i_w_loss'] = i_w_loss_lm[i].item() if GE_I_DO == True else None
            mb[i]['gf_best_o_min_loss'] = o_mins[i]
            mb[i]['gf_best_o_logpdf'] = batch_logpdf[i].detach().cpu().numpy().tolist()

        max_success_co = 0
        enum_success_co = 0
        for i in range(bz):
            max_success = True
            pred_str = '<s>'
            for j in range(tgt_len[i]):
                pred_str += ' ' + vocab[w_logit_pred[i][j]]
                if w_logit_pred[i][j] != tgt_targetv[i][j] or vocab[tgt_targetv[i][j]] == '<unk>': #just to be consistent with enum
                    max_success = False;
            if max_success == True: max_success_co += 1
            if mb[i]['co'] > 0: enum_success_co += 1
            mb[i]['gf_max_success'] = max_success
            mb[i]['enum_max_success'] = enum_success_co
            mb[i]['gf_inputstr'] = ' '.join([vocab[best_idx_glo[i][kkk]] for kkk in range(SRC_SEQ_LEN)])
            print mb[i]['str'], 'best_loss:', mb[i]['gf_best_loss'], 'w_loss:', mb[i]['gf_best_w_loss'], 'o_min:', mb[i]['gf_best_o_min_loss'], 'ge_inputstr:', mb[i]['gf_inputstr'], 'i_w_loss:', mb[i]['gf_best_i_w_loss'], '||', 'max_success:', max_success, '||', pred_str, '|| enum_hit_co:', mb[i]['co'], 'enum_input_str:', mb[i]['adv_input_str']
         
        aux_d = {'iter_co': iter_co, 'max_success_co': max_success_co, 'enum_success_co': enum_success_co}
        if enum_success_co > 0:
            assert(enum_success_co >= max_success_co)
        return best_loss_glo, best_idx_glo, onehot_v_glo, aux_d

    sum_d = {}
    cache = []
    for fn in run_fn_lis: #['normal_words.txt']: #in target_fn_lis:
        logger.info('===starting opt on file %s===', fn)
        care_mode = ''
        if fn.startswith('normal_'):
            GIBBSENUM_DECAYLOSS_WEIGHT = 0
            care_mode = 'max'
        if fn.startswith('mal_') or fn.startswith('random_') or fn.startswith('testing_sam'):
            GIBBSENUM_DECAYLOSS_WEIGHT = 1 #weighted decay disabled beacuse we optimize for best_loss, instead of maxdecoding_hit
            care_mode = 'sample_min'
        logger.info('GF_DECAYLOSS_WEIGHT: %f care_mode: %s', GIBBSENUM_DECAYLOSS_WEIGHT, care_mode)
        if fn == 'normal_words.txt':
            lis_tgt_w[fn] = lis_tgt_w[fn][:100] #debug!!!
        logger.info('all_num_enum_hit: %d', np.sum([(m['co'] > 0) for m in lis_tgt_w[fn]]))
        mb_co = 0
        for m in lis_tgt_w[fn]:
            m['gf_max_success'] = False
            m['gf_best_loss'] = -100
            m['gf_best_sen_loss'] = -100
            m['gf_best_w_loss'] = -100
            if not 'tgt_len' in m: m['tgt_len'] =  (len(m['str'].strip().split()) - 1)
            #if m['co'] > 0: #hit by enum #for fast debug!
            cache.append(m)
            if len(cache) == bz:
                best_loss, best_idx, onehot_v, aux_d = greedyflip_attack_mb(cache, care_mode = care_mode)
                logger.info('mb_co: %d, aux_d: %s', mb_co, str(aux_d))
                logger.info('save_dir is %s', save_dir)
                mb_co += 1
                cache = []        
                #break #debug!!
        if len(cache) > 0:
            best_loss, best_idx, onehot_v, aux_d = greedyflip_attack_mb(cache, care_mode = care_mode)
            logger.info('mb_co: %d, aux_d: %s', mb_co, str(aux_d))
            logger.info('save_dir is %s', save_dir)
            mb_co += 1
            cache = []

        sum_d[fn] = {'max_hit_co': np.sum([m['gf_max_success'] for m in lis_tgt_w[fn]]),
            'avg_best_loss': np.mean([m['gf_best_loss'] for m in lis_tgt_w[fn]]),
            'avg_sen_best_loss': np.mean([m['gf_best_sen_loss'] for m in lis_tgt_w[fn]]),
            'avg_word_best_loss': np.sum([m['gf_best_sen_loss'] for m in lis_tgt_w[fn]]) / np.sum([(m['tgt_len']) for m in lis_tgt_w[fn]]),
            }
        enum_hit_co = np.sum([(m['co'] > 0) for m in lis_tgt_w[fn]])
        sum_d[fn]['enum_hit_co'] = enum_hit_co
        if enum_hit_co == 0:
            sum_d[fn]['max_hit_rate'] = 0
        else:
            sum_d[fn]['max_hit_rate'] = sum_d[fn]['max_hit_co'] * 1.0 / enum_hit_co
        
        sum_d[fn]['o_avg_sample1hit'] = np.sum([(m['gf_best_w_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
        sum_d[fn]['o_avg_sample2hit'] = np.sum([(m['gf_best_w_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
        sum_d[fn]['o_min_sample1hit'] = np.sum([(m['gf_best_o_min_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
        sum_d[fn]['o_min_sample2hit'] = np.sum([(m['gf_best_o_min_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
        if GE_I_DO == True:
            sum_d[fn]['io_avg_sample1hit'] = np.sum([(m['gf_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(1) and m['gf_best_w_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
            sum_d[fn]['io_avg_sample2hit'] = np.sum([(m['gf_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(2) and m['gf_best_w_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
            sum_d[fn]['io_min_sample1hit'] = np.sum([(m['gf_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(1) and m['gf_best_o_min_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
            sum_d[fn]['io_min_sample2hit'] = np.sum([(m['gf_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(2) and m['gf_best_o_min_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
            sum_d[fn]['i_avg_sample1hit'] = np.sum([(m['gf_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
            sum_d[fn]['i_avg_sample2hit'] = np.sum([(m['gf_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
     
        save_fn = save_dir + '/adv_greedyflip_' + LOG_SUC + '_' + fn[:-4] + '_' + 'Flipnum{}RanT{}EnumT{}MaxDecayW{}LambdaGEI{}GEIAvgSampleDecay{}Care{}Iterco{}'.format(str(GREEDYFLIP_NUM), str(GIBBSENUM_RANDOM_TIME), str(GIBBSENUM_E_NUM), str(GIBBSENUM_DECAYLOSS_WEIGHT), str(GE_I_LAMBDA), str(GE_I_SAMPLE_LOSSDECAY), care_mode, GE_MAX_ITER_CO) + '.data'
        logger.info('sum_d[%s]: %s', fn, str(sum_d[fn]))
        logger.info('saving lis_tgt_w[%s] to %s', fn, save_fn)
        torch.save(lis_tgt_w[fn], save_fn)

    logger.info('sum_d: %s', str(sum_d))
    
def adv_globalenum(MODEL_TYPE, m_dict, HIDDEN_SIZE, DATA_SET, tgt_w_data_fn, run_fn_lis, bz, vocab, vocab_inv, SRC_SEQ_LEN, TGT_SEQ_LEN, save_dir, NORMAL_WORD_AVG_LOSS, GIBBSENUM_DECAYLOSS_WEIGHT, GIBBSENUM_E_NUM, GIBBSENUM_RANDOM_TIME, ENUM_START_LENGTH = 1, ENUM_END_LENGTH = 100, GE_I_DO = False, GE_I_WORD_AVG_LOSS = None, GE_I_LAMBDA = None, GE_I_LM = None, GE_I_SAMPLE_LOSSDECAY = False, GE_MAX_ITER_CO = 50, log_fn = '', LOG_SUC = ''):  
    m_embed, m_encode_w_rnn, m_decode_w_rnn = m_dict['m_embed'], m_dict['m_encode_w_rnn'], m_dict['m_decode_w_rnn']
    m_embed_dp, m_encode_w_rnn_dp, m_decode_w_rnn_dp = m_dict['m_embed_dp'], m_dict['m_encode_w_rnn_dp'], m_dict['m_decode_w_rnn_dp']
    
    if tgt_w_data_fn != None:
        logger.info('loading data lis_tgt_w from %s', tgt_w_data_fn)
        lis_tgt_w = torch.load(tgt_w_data_fn)
    else:
        logger.info('tgt_w_data_fn is None, building an empty one, but note that no enumall_information will be aviable')
        lis_tgt_w = build_lis_tgt_w(DATA_SET, vocab_inv, run_fn_lis)  

    """
    lis_tgt_w = {}
    for fn in target_fn_lis:
        lis_r = open('../adv_lists/' + DATA_SET + '/' + fn, 'r').readlines()
        lis_tgt_w[fn] = [{'str':l.strip(), 'co':0, 'adv_input_str':None, 'fn':fn} for l in lis_r if (len(l.split()) > 2)]
        logger.info('fn: %s len: %d', fn, len(lis_tgt_w[fn])) 
    """
    
    logger.info('THIS IS GLOAL_ENUM')

    if GE_I_DO == True:
        logger.info('GE_I_DO is true!! adding input lm score, also initializing from lm samples')
        logger.info('GE_I_LAMBDA: %f GE_I_SAMPLE_LOSSDECAY: %s', GE_I_LAMBDA, str(GE_I_SAMPLE_LOSSDECAY))  
    else:
        logger.info('GE_I_DO is false, setting ge_i_lambda to zero')
        GE_I_LAMBDA = 0
    logger.info('RANDOM_TIME: %d E_NUM: %d MAX_ITER_CO: %d', GIBBSENUM_RANDOM_TIME, GIBBSENUM_E_NUM, GE_MAX_ITER_CO)
    
    def globalenum_attack_mb(mb, care_mode = ''):
        assert(care_mode in ('max', 'sample_min', 'sample_avg'))
        bz = len(mb)
        lis_tgt_w = [m['str'].split() for m in mb]
        tgt_len = [len(l) - 1 for l in lis_tgt_w]
        max_len = max(tgt_len)
        lis_tgt_w = [l + ['<pad>'] * (max_len + 1 - len(l)) for l in lis_tgt_w]
        lis_tgt_idx = [[(vocab_inv[w] if w in vocab_inv else vocab_inv['<unk>'] ) for w in l] for l in lis_tgt_w]
        tgt_mb = Variable(torch.LongTensor(lis_tgt_idx).cuda())
        tgt_inputv = tgt_mb[:, :-1]
        tgt_targetv = tgt_mb[:, 1:]
       
        def model_forward(onehot_v, idx_lis, bz, decay_loss = True, do_train = True): 
            m_list = [m_embed, m_encode_w_rnn, GE_I_LM, m_decode_w_rnn, m_embed_dp, m_encode_w_rnn_dp, m_decode_w_rnn_dp]
            if do_train == True: 
                for m in m_list: 
                    if m != None: m.train()
            if do_train == False: 
                for m in m_list: 
                    if m != None: m.eval()

            output, _ = m_encode_w_rnn_dp(m_embed_dp(onehot_v).permute(1,0,2), init_lstm_hidden(bz, HIDDEN_SIZE))
            output = output.permute(1,0,2)
            latent = output[:, -1, :].unsqueeze(1).repeat(1, tgt_targetv.size(1), 1)
            
            if MODEL_TYPE == 'latent': w_logit_rnn = m_decode_w_rnn_dp(latent, tgt_inputv, tgt_len)
            if MODEL_TYPE == 'attention': w_logit_rnn, attn_weights = m_decode_w_rnn_dp(output, tgt_inputv, tgt_len)

            flat_output = w_logit_rnn.view(-1, len(vocab))
            flat_target = tgt_targetv.contiguous().view(-1)
            flat_logpdf = lib_pdf.logsoftmax_idxselect(flat_output, flat_target)
            batch_logpdf = flat_logpdf.view(bz, -1)

            tgt_mask = Variable(torch.FloatTensor([([1] * l + [0] * (tgt_inputv.size(1) - l)) for l in tgt_len]).cuda(), requires_grad = False)
            w_logit_pred = torch.max(w_logit_rnn, dim = 2)[1]
            #if decay_loss == True:
            #print 'decay_weight:', GIBBSENUM_DECAYLOSS_WEIGHT 
            o_mins = []
            for i in range(bz):
                minn = 1000
                for j in range(tgt_len[i]):
                    if care_mode == 'max' and decay_loss == True and w_logit_pred[i][j] == tgt_targetv[i][j]:
                        tgt_mask[i][j] = min(tgt_mask[i][j], GIBBSENUM_DECAYLOSS_WEIGHT) #debug! #why not just zero?
                    if care_mode == 'sample_min' and decay_loss == True and batch_logpdf[i][j].item() >= NORMAL_WORD_AVG_LOSS:
                        tgt_mask[i][j] = min(tgt_mask[i][j], 0.01)
                    if batch_logpdf[i][j].item() < minn:
                        minn = batch_logpdf[i][j].item()
                    #if tgt_targetv[i][j] != 0:
                        #logger.info('setting to 0.001: |%s|', vocab[tgt_targetv[i][j]])
                o_mins.append(minn)
            weight_batch_logpdf = batch_logpdf * tgt_mask
            sen_loss_rnn = torch.sum(weight_batch_logpdf, dim = 1)
            w_loss_rnn = sen_loss_rnn / torch.FloatTensor(tgt_len).cuda() #
            
            if GE_I_DO == True:
                i_w_loss_lm = GE_I_LM.calMeanLogp(idx_lis, onehot_v, mode = 'input_eou', train_flag = True)
            else:
                i_w_loss_lm = None
            o_scal = Variable(torch.ones(bz).cuda(), requires_grad = False)
            i_scal = Variable(torch.ones(bz).cuda(), requires_grad = False)
            #print i_w_loss_lm.size()
            for k in range(bz):
                if GE_I_DO == True and GE_I_SAMPLE_LOSSDECAY == True:
                    if i_w_loss_lm[k].item() > GE_I_WORD_AVG_LOSS: i_scal[k] = 0.01
                if care_mode == 'sample_avg' and w_loss_rnn[k].item() > NORMAL_WORD_AVG_LOSS: o_scal[k] = 0.01 #debug!!
                if care_mode == 'sample_min' and o_mins[k] > NORMAL_WORD_AVG_LOSS: o_scal[k] = 0.01
        
            if GE_I_DO == False:
                loss_combine = w_loss_rnn
            else:
                loss_combine = w_loss_rnn * o_scal + i_w_loss_lm * i_scal * GE_I_LAMBDA

            return w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf
           
        def get_onehotv(idx_arr):
            onehot_v = torch.zeros(bz, SRC_SEQ_LEN, len(vocab)).cuda().detach()
            for i in range(bz): 
                for pos in range(SRC_SEQ_LEN):
                    onehot_v[i, pos, idx_arr[i][pos]] = 1
            return onehot_v
        
        best_idx_glo = [([-1 for k in range(SRC_SEQ_LEN - 1)] + [vocab_inv['<eou>']]) for kk in range(bz)]
        best_loss_glo = [-1000 for k in range(bz)]
        
        for random_iter in range(GIBBSENUM_RANDOM_TIME):
            if GE_I_DO == False:
                best_idx = []
                for kk in range(bz):
                    ini_seq = [vocab_inv['<eou>']]
                    for k in range(SRC_SEQ_LEN - 1):
                        ini_seq = [random.randint(5, len(vocab) - 1)] + ini_seq
                    best_idx.append(ini_seq)
                #best_idx = [([vocab_inv['e'] for k in range(SRC_SEQ_LEN - 1)] + [vocab_inv['<eou>']]) for kk in range(bz)]
            else:
                best_idx = GE_I_LM.sampleBatch(SRC_SEQ_LEN, bz, full_length = True)
            onehot_v = get_onehotv(best_idx) 
            
            best_loss = [-1000 for k in range(bz)]        
            #The important booking keeping if best_loss and best_idx
            w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(onehot_v, best_idx, bz, decay_loss = True, do_train = False)
            for i in range(bz): best_loss[i] = loss_combine[i].item()
            #print w_loss_rnn
            w_loss_rnn_lis = [torch.mean(w_loss_rnn).detach().item()] #for figure drawing
            print 'first:', w_loss_rnn_lis
            w_loss_now = [[-1000] * bz, [-1000] * bz]
            iter_co = 0
            while True:
                last_mean_best = np.mean(best_loss)
                onehot_v = get_onehotv(best_idx)
                onehot_opt = Variable(onehot_v, requires_grad = True) 
                w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(onehot_opt, best_idx, bz, decay_loss = True, do_train = True)
                (loss_combine).mean().backward() #the loss here is actually not loss, but the log-likelihood(objective)
                oh_grad = onehot_opt.grad
                
                try_id = [0 for kk in range(bz)] 
                tmp_idx = copy.deepcopy(best_idx) #best_idx is being updated during enumeration
                for k in range(GIBBSENUM_E_NUM):
                    cur_idx = copy.deepcopy(tmp_idx)
                    onehot_v = get_onehotv(cur_idx)
                    for i in range(bz): #this is the important greedyglip step!
                        for j in range(SRC_SEQ_LEN - 1):
                            oh_grad[i, j] = oh_grad[i, j] - oh_grad[i, j, cur_idx[i][j]] #this is the first-order approximation of an index change
                        sval, sidx = torch.sort(oh_grad[i, :].view(-1), descending = True) #descending=True! Remember that gradient is in the ascending direction.
                        while try_id[i] < len(vocab) * SRC_SEQ_LEN:
                            try_pos, try_wid = sidx[try_id[i]] / len(vocab), sidx[try_id[i]] % len(vocab)
                            try_id[i] += 1
                            if (try_pos == (SRC_SEQ_LEN - 1)) or (vocab[try_wid] in ('<s>', "</s>", "<pad>", "<eou>")): continue
                            #if sval[try_id[i]] <= 0: break #we are finding "ascent" direction
                            assert(onehot_v[i, try_pos, cur_idx[i][try_pos]] == 1)
                            onehot_v[i, try_pos, cur_idx[i][try_pos]] = 0
                            cur_idx[i][try_pos] = try_wid
                            onehot_v[i, try_pos, try_wid] = 1
                            #logger.info('changing... sidx[try_id]: %d sval: %f', sidx[try_id], sval[try_id])
                            break 
                    w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(onehot_v, cur_idx, bz, decay_loss = True, do_train = False)
                    for i in range(bz):
                        if loss_combine[i].item() > best_loss[i]:
                            best_loss[i] = loss_combine[i].item()
                            best_idx[i] = copy.deepcopy(cur_idx[i])
                #===just for loss curve drawing===
                w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(get_onehotv(best_idx), best_idx, bz, decay_loss = True, do_train = False)
                w_loss_rnn_lis.append(torch.mean(w_loss_rnn).detach().item())
                #print 'last w_loss_rnn:', w_loss_rnn_lis[-1]
                #logger.info('iter_co: %d cur_best_loss: %f last_mean_best: %f', iter_co, np.mean(best_loss), last_mean_best)
                #===just for end===
                if np.mean(best_loss) <= last_mean_best: break
                last_mean_best = np.mean(best_loss) 
                iter_co += 1; 
                if iter_co > GE_MAX_ITER_CO * 10: break
            print 'w_loss_rnn_lis:', str(w_loss_rnn_lis); #sys.exit(1)
            for i in range(bz): #in the end the random for loop
                if best_loss[i] > best_loss_glo[i]: 
                    best_loss_glo[i] = best_loss[i]; best_idx_glo[i] = best_idx[i]
             
        onehot_v_glo = get_onehotv(best_idx_glo)
        w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(onehot_v_glo, best_idx_glo, bz, decay_loss = False, do_train = False) #here we want the true loss so decay_loss=False
         
        for i in range(bz): 
            best_loss_glo[i] = loss_combine[i].item() #best_loss_glo could be weighted
            mb[i]['gf_best_loss'] = best_loss_glo[i]
            mb[i]['gf_best_sen_loss'] = sen_loss_rnn[i].item()
            mb[i]['gf_best_w_loss'] = w_loss_rnn[i].item()
            mb[i]['gf_best_i_w_loss'] = i_w_loss_lm[i].item() if GE_I_DO == True else None
            mb[i]['gf_best_o_min_loss'] = o_mins[i]
            mb[i]['gf_best_o_logpdf'] = batch_logpdf[i].detach().cpu().numpy().tolist()

        max_success_co = 0
        enum_success_co = 0
        for i in range(bz):
            max_success = True
            pred_str = '<s>'
            for j in range(tgt_len[i]):
                pred_str += ' ' + vocab[w_logit_pred[i][j]]
                if w_logit_pred[i][j] != tgt_targetv[i][j] or vocab[tgt_targetv[i][j]] == '<unk>': #just to be consistent with enum
                    max_success = False;
            if max_success == True: max_success_co += 1
            if mb[i]['co'] > 0: enum_success_co += 1
            mb[i]['gf_max_success'] = max_success
            mb[i]['enum_max_success'] = enum_success_co
            mb[i]['gf_inputstr'] = ' '.join([vocab[best_idx_glo[i][kkk]] for kkk in range(SRC_SEQ_LEN)])
            print mb[i]['str'], 'best_loss:', mb[i]['gf_best_loss'], 'w_loss:', mb[i]['gf_best_w_loss'], 'o_min:', mb[i]['gf_best_o_min_loss'], 'ge_inputstr:', mb[i]['gf_inputstr'], 'i_w_loss:', mb[i]['gf_best_i_w_loss'], '||', 'max_success:', max_success, '||', pred_str, '|| enum_hit_co:', mb[i]['co'], 'enum_input_str:', mb[i]['adv_input_str']
         
        aux_d = {'iter_co': iter_co, 'max_success_co': max_success_co, 'enum_success_co': enum_success_co}
        if enum_success_co > 0:
            assert(enum_success_co >= max_success_co)
        return best_loss_glo, best_idx_glo, onehot_v_glo, aux_d

    sum_d = {}
    cache = []
    for fn in run_fn_lis: #['normal_words.txt']: #in target_fn_lis:
        logger.info('===starting opt on file %s===', fn)
        care_mode = ''
        if fn.startswith('normal_'):
            GIBBSENUM_DECAYLOSS_WEIGHT = 0
            care_mode = 'max'
        if fn.startswith('mal_') or fn.startswith('random_') or fn.startswith('testing_sam'):
            GIBBSENUM_DECAYLOSS_WEIGHT = 1 #weighted decay disabled beacuse we optimize for best_loss, instead of maxdecoding_hit
            care_mode = 'sample_min'
        logger.info('GF_DECAYLOSS_WEIGHT: %f care_mode: %s', GIBBSENUM_DECAYLOSS_WEIGHT, care_mode)
        if fn == 'normal_words.txt':
            lis_tgt_w[fn] = lis_tgt_w[fn][:100] #debug!!!
        logger.info('all_num_enum_hit: %d', np.sum([(m['co'] > 0) for m in lis_tgt_w[fn]]))
        mb_co = 0
        for m in lis_tgt_w[fn]:
            m['gf_max_success'] = False
            m['gf_best_loss'] = -100
            m['gf_best_sen_loss'] = -100
            m['gf_best_w_loss'] = -100
            if not 'tgt_len' in m: m['tgt_len'] =  (len(m['str'].strip().split()) - 1)
            #if m['co'] > 0: #hit by enum #for fast debug!
            cache.append(m)
            if len(cache) == bz:
                best_loss, best_idx, onehot_v, aux_d = globalenum_attack_mb(cache, care_mode = care_mode)
                logger.info('mb_co: %d, aux_d: %s', mb_co, str(aux_d))
                logger.info('save_dir is %s', save_dir)
                mb_co += 1
                cache = []        
                #break #debug!!
        if len(cache) > 0:
            best_loss, best_idx, onehot_v, aux_d = globalenum_attack_mb(cache, care_mode = care_mode)
            logger.info('mb_co: %d, aux_d: %s', mb_co, str(aux_d))
            logger.info('save_dir is %s', save_dir)
            mb_co += 1
            cache = []

        sum_d[fn] = {'max_hit_co': np.sum([m['gf_max_success'] for m in lis_tgt_w[fn]]),
            'avg_best_loss': np.mean([m['gf_best_loss'] for m in lis_tgt_w[fn]]),
            'avg_sen_best_loss': np.mean([m['gf_best_sen_loss'] for m in lis_tgt_w[fn]]),
            'avg_word_best_loss': np.sum([m['gf_best_sen_loss'] for m in lis_tgt_w[fn]]) / np.sum([(m['tgt_len']) for m in lis_tgt_w[fn]]),
            }
        enum_hit_co = np.sum([(m['co'] > 0) for m in lis_tgt_w[fn]])
        sum_d[fn]['enum_hit_co'] = enum_hit_co
        if enum_hit_co == 0:
            sum_d[fn]['max_hit_rate'] = 0
        else:
            sum_d[fn]['max_hit_rate'] = sum_d[fn]['max_hit_co'] * 1.0 / enum_hit_co
        
        sum_d[fn]['o_avg_sample1hit'] = np.sum([(m['gf_best_w_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
        sum_d[fn]['o_avg_sample2hit'] = np.sum([(m['gf_best_w_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
        sum_d[fn]['o_min_sample1hit'] = np.sum([(m['gf_best_o_min_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
        sum_d[fn]['o_min_sample2hit'] = np.sum([(m['gf_best_o_min_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
        if GE_I_DO == True:
            sum_d[fn]['io_avg_sample1hit'] = np.sum([(m['gf_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(1) and m['gf_best_w_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
            sum_d[fn]['io_avg_sample2hit'] = np.sum([(m['gf_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(2) and m['gf_best_w_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
            sum_d[fn]['io_min_sample1hit'] = np.sum([(m['gf_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(1) and m['gf_best_o_min_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
            sum_d[fn]['io_min_sample2hit'] = np.sum([(m['gf_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(2) and m['gf_best_o_min_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
            sum_d[fn]['i_avg_sample1hit'] = np.sum([(m['gf_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
            sum_d[fn]['i_avg_sample2hit'] = np.sum([(m['gf_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
     
        save_fn = save_dir + '/adv_globalenum_' + LOG_SUC + '_' + fn[:-4] + '_' + 'RanT{}EnumT{}MaxDecayW{}LambdaGEI{}GEIAvgSampleDecay{}Care{}Iterco{}'.format(str(GIBBSENUM_RANDOM_TIME), str(GIBBSENUM_E_NUM), str(GIBBSENUM_DECAYLOSS_WEIGHT), str(GE_I_LAMBDA), str(GE_I_SAMPLE_LOSSDECAY), care_mode, GE_MAX_ITER_CO) + '.data'
        logger.info('sum_d[%s]: %s', fn, str(sum_d[fn]))
        logger.info('saving lis_tgt_w[%s] to %s', fn, save_fn)
        torch.save(lis_tgt_w[fn], save_fn)

    logger.info('sum_d: %s', str(sum_d)) 
