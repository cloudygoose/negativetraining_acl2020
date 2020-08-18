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

from more_adv_attacks import *

logger = logging.getLogger()

sum_d = {}

def hashing(s):
    return int(hashlib.sha1(s).hexdigest(), 16)

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

def adv_enumerate_all(MODEL_TYPE, m_dict, HIDDEN_SIZE, DATA_SET, target_fn_lis, exclude_fn_lis, bz, vocab, vocab_inv, SRC_SEQ_LEN, TGT_SEQ_LEN, save_dir, ENUM_START_LENGTH = 1, ENUM_END_LENGTH = 100, ENUM_START_RATE = 0, ENUM_END_RATE = 1, log_fn = '', LOG_SUC = ''):
    m_embed, m_encode_w_rnn, m_decode_w_rnn = m_dict['m_embed'], m_dict['m_encode_w_rnn'], m_dict['m_decode_w_rnn']
    
    lis_tgt_w = build_lis_tgt_w(DATA_SET, vocab_inv, target_fn_lis) 
   
    hash_dir = {}
    for fn in lis_tgt_w:
        sum_d[fn] = {'hit_co':0, 'substr_hit_co':0}
        for m in lis_tgt_w[fn]:
            c = m['str']
            if c in hash_dir:
                print c, 'appeared twice'
                sys.exit(1)
            hash_dir[c] = m
    
    logger.info('building exclude_dict by %s', str(exclude_fn_lis))
    do_exclude = False
    exclude_dict = {'<s> < u n k > </s>':{'hit_co':0}, '<s> <unk> </s>':{'hit_co':0}} #attention, sometimes <unk> is not in the training file
    for fn in exclude_fn_lis:
        do_exclude = True
        for l in open(fn, 'r').readlines():
            ws = l.strip().lower().split()
            for w in ws:
                if len(w) > TGT_SEQ_LEN: w = w[:TGT_SEQ_LEN]
                w = '<s> ' + ' '.join(w) + ' </s>'
                exclude_dict[w] = {'hit_co':0}
    exclude_records = []

    lv = len(vocab)
    logger.info('search batch size is %d', bz)
    logger.info('length of vocab: %d', lv)
    logger.info('start length: %d end length: %d', ENUM_START_LENGTH, ENUM_END_LENGTH)
    logger.info('start rate : %f end rate : %f', ENUM_START_RATE, ENUM_END_RATE)

    def compare_output(lis_input):
        bz_now = len(lis_input)
        src_inputv = Variable(torch.LongTensor(lis_input)).cuda() 
        output, _ = m_encode_w_rnn(m_embed(idx2onehot(src_inputv, lv)), init_lstm_hidden(bz_now, HIDDEN_SIZE))
        latent = output[:, -1, :].unsqueeze(1).repeat(1, TGT_SEQ_LEN, 1)
        if MODEL_TYPE == 'latent': samples = m_decode_w_rnn.generate(latent, sample = False)
        if MODEL_TYPE == 'attention': samples, _ = m_decode_w_rnn.generate(output, latent, sample = False)
        samples = samples.cpu().numpy().tolist()
        for i in range(bz_now):
            sample_str = '<s> ' + ' '.join(clean_sen([vocab[kk] for kk in samples[i]])) + ' </s>'
            #if len(clean_sen([vocab[kk] for kk in samples[i]])) == 10:
            #    print sample_str, '10!!!!!!!!!'
            #print sample_str
            if sample_str in hash_dir:
                m = hash_dir[sample_str]
                input_str = ' '.join([vocab[kk] for kk in lis_input[i]])
                fn = m['fn']
                if m['co'] == 0:
                    logger.info('new_hit!! %d fn: %s str: [%s] input_str: [%s]', sum_d[fn]['hit_co'], fn, m['str'].replace(' ', ''), input_str.replace(' ', ''))
                    sum_d[fn]['hit_co'] += 1
                if m['co'] > 0 and m['co'] < 2:
                    logger.info('again_hit!!(first 2) %d fn: %s str: [%s] input_str: [%s]', sum_d[fn]['hit_co'], fn, m['str'].replace(' ', ''), input_str.replace(' ', ''))
                m['co'] += 1
                m['adv_input_str'] = input_str
            
            ss_lis = sample_str.split()
            for st in range(1, len(ss_lis) - 2):
                for ed in range(st + 1, len(ss_lis)):
                    ss_now = '<s> ' + ' '.join(ss_lis[st:ed]) + ' </s>'
                    if ss_now in hash_dir:
                        m = hash_dir[ss_now]
                        input_str = ' '.join([vocab[kk] for kk in lis_input[i]])
                        fn = m['fn']
                        if m['substr_co'] == 0:
                            logger.info('new_substr_hit!! %d fn: %s str: [%s] str_full: [%s] input_str: [%s]', sum_d[fn]['substr_hit_co'], fn, m['str'].replace(' ', ''), sample_str, input_str.replace(' ', ''))
                            sum_d[fn]['substr_hit_co'] += 1
                        m['substr_co'] += 1
                        m['substr_adv_input_str'] = input_str
                        m['substr_full'] = sample_str

            if do_exclude == True:
                if not sample_str in exclude_dict:
                    input_str = ' '.join([vocab[kk] for kk in lis_input[i]])
                    if len(exclude_records) <= 1000 or len(exclude_records) % 10000 == 0:
                        logger.info('new_exclude!! %d str: [%s] input_str: [%s]', len(exclude_records), sample_str, input_str.replace(' ', ''))
                    exclude_records.append({'str': sample_str, 'adv_input_str': input_str})
                else:
                    exclude_dict[sample_str]['hit_co'] += 1
     
    cache = [] 
    for length in range(ENUM_START_LENGTH, min(SRC_SEQ_LEN, ENUM_END_LENGTH)):
        logger.info('now search with length %d', length)
        k = int((lv ** length) * ENUM_START_RATE)
        end_k = int((lv ** length) * ENUM_END_RATE)
        while k < end_k:
            LOG_INTERVAL = 10 ** 6
            if k % LOG_INTERVAL == 0:
                logger.info('MTYPE: %s enum k at %d, l: %d, rate now: %f', MODEL_TYPE, k, length, k * 1.0 / (lv ** length))
                if k % (LOG_INTERVAL * 10) == 0:
                    log_str = ''
                    for fn in sum_d:
                        log_str += ' hit_co({}):'.format(fn) + str(sum_d[fn]['hit_co'])
                    logger.info('%s log_fn is %s save_dir is %s', log_str, log_fn, save_dir)
            
            m = k
            l_now = []
            for l in range(length):
                l_now.append(m % lv)
                m = m / lv
            k = k + 1
            
            if sum([idx == vocab_inv['<pad>'] or idx == vocab_inv['</s>'] or idx == vocab_inv['<s>'] or idx == vocab_inv['<eou>'] for idx in l_now]) > 0:
                continue #invalid input seq

            l_now = [vocab_inv['<pad>']] * (SRC_SEQ_LEN - 1 - length) + l_now + [vocab_inv['<eou>']]
            cache.append(l_now)
            if len(cache) == bz or k == end_k - 1:
                compare_output(cache)
                cache = []
    
    logger.info('==first 100 exclude records==')
    for i in range(min(len(exclude_records), 100)):
        m = exclude_records[i]
        logger.info('sample: %s input: %s', m['str'], m['adv_input_str'])     
        
    res_d = {} 
    for fn in lis_tgt_w:
        logger.info('==success summarize %s==', fn)
        hit_num, all_num, substr_hit_num = 0, 0, 0
        for m in lis_tgt_w[fn]:
            all_num += 1
            if m['co'] > 0: 
                if hit_num < 100 or hit_num % 100 == 0:
                    logger.info('success%d: %s adv_input: %s', hit_num, m['str'], m['adv_input_str'])
                hit_num += 1
                if m['substr_co'] == 0:
                    print 'SOMETHING WRONG SUB_STR!!!!!', m['str']
            if m['substr_co'] > 0:
                if substr_hit_num < 100 or substr_hit_num % 100 == 0:
                    logger.info('substr success%d: %s full_str: %s adv_input: %s', substr_hit_num, m['str'], m['substr_full'], m['substr_adv_input_str'])
                substr_hit_num += 1
        res_d[fn] = {'coverage_rate': (hit_num * 1.0 / all_num), 'substr_coverage_rate': (substr_hit_num * 1.0 / all_num)}
    
    logger.info('==numbers==')
    for fn in lis_tgt_w:
        logger.info('%s %s', fn, str(res_d[fn]))
    logger.info('exclude number: %d', len(exclude_records))
    
    save_fn = save_dir + '/adv_enum_' + LOG_SUC + '_exclude_records.data'
    logger.info('saving exclude_records to %s', save_fn)
    torch.save(exclude_records, save_fn)

    save_fn = save_dir + '/adv_enum_' + LOG_SUC + '_lis_tgt_w.data'
    logger.info('saving lis_tgt_w to %s', save_fn)
    torch.save(lis_tgt_w, save_fn)

    return res_d

def adv_gibbsenum(MODEL_TYPE, m_dict, HIDDEN_SIZE, DATA_SET, tgt_w_data_fn, run_fn_lis, bz, vocab, vocab_inv, SRC_SEQ_LEN, TGT_SEQ_LEN, save_dir, NORMAL_WORD_AVG_LOSS, GIBBSENUM_DECAYLOSS_WEIGHT, GIBBSENUM_E_NUM, GIBBSENUM_RANDOM_TIME, ENUM_START_LENGTH = 1, ENUM_END_LENGTH = 100, GE_I_DO = False, GE_I_WORD_AVG_LOSS = None, GE_I_LAMBDA = None, GE_I_LM = None, GE_I_SAMPLE_LOSSDECAY = False, GE_MAX_ITER_CO = 50, log_fn = '', LOG_SUC = ''):  
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
    
    if GE_I_DO == True:
        logger.info('GE_I_DO is true!! adding input lm score, also initializing from lm samples')
        logger.info('GE_I_LAMBDA: %f GE_I_SAMPLE_LOSSDECAY: %s', GE_I_LAMBDA, str(GE_I_SAMPLE_LOSSDECAY))  
    else:
        logger.info('ge_i_do is false, setting ge_i_do to zero')
        GE_I_LAMBDA = 0
    logger.info('RANDOM_TIME: %d E_NUM: %d MAX_ITER_CO: %d', GIBBSENUM_RANDOM_TIME, GIBBSENUM_E_NUM, GE_MAX_ITER_CO)
    
    def gibbsenum_attack_mb(mb, care_mode = ''):
        assert(care_mode in ('max', 'sample_min', 'sample_avg'))
        bz = len(mb)
        lis_tgt_w = [m['str'].split() for m in mb]
        tgt_len = [len(l) - 1 for l in lis_tgt_w]
        max_len = max(tgt_len)
        lis_tgt_w = [l + ['</s>'] * (max_len + 1 - len(l)) for l in lis_tgt_w]
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
            if MODEL_TYPE == 'attention': w_logit_rnn, attn_weights, _ = m_decode_w_rnn_dp(output, tgt_inputv)

            flat_output = w_logit_rnn.view(-1, len(vocab))
            flat_target = tgt_targetv.contiguous().view(-1)
            flat_logpdf = lib_pdf.logsoftmax_idxselect(flat_output, flat_target)
            batch_logpdf = flat_logpdf.view(bz, -1)
            #print 'tgt_targetv[8,9,10]', tgt_targetv[8], tgt_targetv[9], tgt_targetv[10]
            #print lis_tgt_w[8], lis_tgt_w[9]

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
                i_w_loss_lm = GE_I_LM.calMeanLogp(idx_lis, onehot_v, mode = 'input_eou', train_flag = do_train)
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
                best_idx = []
                for kk in range(bz):
                    ini_seq = [vocab_inv['<eou>']]
                    for k in range(SRC_SEQ_LEN - 1):
                        ini_seq = [random.randint(5, len(vocab) - 1)] + ini_seq
                    best_idx.append(ini_seq)
                #best_idx = [([vocab_inv['e'] for k in range(SRC_SEQ_LEN - 1)] + [vocab_inv['<eou>']]) for kk in range(bz)]
            else:
                best_idx = GE_I_LM.sampleBatch(SRC_SEQ_LEN, bz, full_length = True)
            onehot_v = torch.zeros(bz, SRC_SEQ_LEN, len(vocab)).cuda().detach()
            for i in range(bz): 
                for pos in range(SRC_SEQ_LEN):
                    onehot_v[i, pos, best_idx[i][pos]] = 1
     
            best_loss = [-1000 for k in range(bz)]
            w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(onehot_v, best_idx, bz, decay_loss = True, do_train = False)
            #print w_loss_rnn
            w_loss_lis = [[w_loss_rnn.mean().item()], [np.mean(o_mins)]]; # print w_loss_lis #for figure drawing, not actually used, 0 for avg, 1 for min
            print 'w_loss_lis[0] now:', w_loss_lis[0]
            w_loss_now = [[-1000] * bz, [-1000] * bz]
            iter_co = 0
            while True:
                last_mean_best = np.mean(best_loss)
                for pos in range(SRC_SEQ_LEN - 1):
                    cur_idx = copy.deepcopy(best_idx)            
                    if GIBBSENUM_E_NUM != -1:
                        onehot_opt = Variable(onehot_v, requires_grad = True)
                        #print onehot_opt.grad[9, pos, :10] #gradeint wrong!!
                        w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(onehot_opt, best_idx, bz, decay_loss = True, do_train = True)
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
                        w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(onehot_v, cur_idx, bz, decay_loss = True, do_train = False)
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
                if np.mean(best_loss) > last_mean_best and iter_co <= GE_MAX_ITER_CO:
                    last_mean_best = np.mean(best_loss)
                else: break
            for i in range(bz):
                if best_loss[i] > best_loss_glo[i]: 
                    best_loss_glo[i] = best_loss[i]; best_idx_glo[i] = best_idx[i]
        
        onehot_v_glo = torch.zeros(bz, SRC_SEQ_LEN, len(vocab)).cuda().detach()
        for i in range(bz): 
            for pos in range(SRC_SEQ_LEN):
                onehot_v_glo[i, pos, best_idx_glo[i][pos]] = 1  
        w_logit_rnn, w_logit_pred, sen_loss_rnn, w_loss_rnn, loss_combine, i_w_loss_lm, o_mins, batch_logpdf = model_forward(onehot_v_glo, best_idx_glo, bz, decay_loss = False, do_train = False) #here we want the true loss so decay_loss=False
         
        o_min_lis = []
        for i in range(bz): 
            best_loss_glo[i] = loss_combine[i].item() #best_loss_glo could be weighted
            mb[i]['gibbs_enum_best_loss'] = best_loss_glo[i]
            mb[i]['gibbs_enum_best_sen_loss'] = sen_loss_rnn[i].item()
            mb[i]['gibbs_enum_best_w_loss'] = w_loss_rnn[i].item()
            mb[i]['gibbs_enum_best_i_w_loss'] = i_w_loss_lm[i].item() if GE_I_DO == True else None
            mb[i]['gibbs_enum_best_o_min_loss'] = o_mins[i]
            o_min_lis.append(o_mins[i])
            mb[i]['gibbs_enum_best_o_logpdf'] = batch_logpdf[i].detach().cpu().numpy().tolist()
        print 'o_min_lis:', o_min_lis

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
            mb[i]['gibbs_enum_max_success'] = max_success
            mb[i]['enum_max_success'] = enum_success_co
            mb[i]['gibbs_enum_inputstr'] = ' '.join([vocab[best_idx_glo[i][kkk]] for kkk in range(SRC_SEQ_LEN)])
            print mb[i]['str'], 'best_loss:', mb[i]['gibbs_enum_best_loss'], 'w_loss:', mb[i]['gibbs_enum_best_w_loss'], 'o_min:', mb[i]['gibbs_enum_best_o_min_loss'], 'ge_inputstr:', mb[i]['gibbs_enum_inputstr'], 'i_w_loss:', mb[i]['gibbs_enum_best_i_w_loss'], '||', 'max_success:', max_success, '||', pred_str, '|| enum_hit_co:', mb[i]['co'], 'enum_input_str:', mb[i]['adv_input_str']
         
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
        logger.info('GIBBSENUM_DECAYLOSS_WEIGHT: %f care_mode: %s', GIBBSENUM_DECAYLOSS_WEIGHT, care_mode)
        if fn == 'normal_words.txt':
            lis_tgt_w[fn] = lis_tgt_w[fn][:100] #debug!!!
        logger.info('all_num_enum_hit: %d', np.sum([(m['co'] > 0) for m in lis_tgt_w[fn]]))
        mb_co = 0
        for m in lis_tgt_w[fn]:
            m['gibbs_enum_max_success'] = False
            m['gibbs_enum_best_loss'] = -100
            m['gibbs_enum_best_sen_loss'] = -100
            m['gibbs_enum_best_w_loss'] = -100
            if not 'tgt_len' in m: m['tgt_len'] =  (len(m['str'].strip().split()) - 1)
            #if m['co'] > 0: #hit by enum #for fast debug!
            cache.append(m)
            if len(cache) == bz:
                best_loss, best_idx, onehot_v, aux_d = gibbsenum_attack_mb(cache, care_mode = care_mode)
                logger.info('mb_co: %d, aux_d: %s', mb_co, str(aux_d))
                logger.info('save_dir is %s', save_dir)
                mb_co += 1
                cache = []        
                #break #debug!!
        if len(cache) > 0:
            best_loss, best_idx, onehot_v, aux_d = gibbsenum_attack_mb(cache, care_mode = care_mode)
            logger.info('mb_co: %d, aux_d: %s', mb_co, str(aux_d))
            logger.info('save_dir is %s', save_dir)
            mb_co += 1
            cache = []

        sum_d[fn] = {'max_hit_co': np.sum([m['gibbs_enum_max_success'] for m in lis_tgt_w[fn]]),
            'avg_best_loss': np.mean([m['gibbs_enum_best_loss'] for m in lis_tgt_w[fn]]),
            'avg_sen_best_loss': np.mean([m['gibbs_enum_best_sen_loss'] for m in lis_tgt_w[fn]]),
            'avg_word_best_loss': np.sum([m['gibbs_enum_best_sen_loss'] for m in lis_tgt_w[fn]]) / np.sum([(m['tgt_len']) for m in lis_tgt_w[fn]]),
            }
        enum_hit_co = np.sum([(m['co'] > 0) for m in lis_tgt_w[fn]])
        sum_d[fn]['enum_hit_co'] = enum_hit_co
        if enum_hit_co == 0:
            sum_d[fn]['max_hit_rate'] = 0
        else:
            sum_d[fn]['max_hit_rate'] = sum_d[fn]['max_hit_co'] * 1.0 / enum_hit_co
        
        sum_d[fn]['o_avg_sample1hit'] = np.sum([(m['gibbs_enum_best_w_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
        sum_d[fn]['o_avg_sample2hit'] = np.sum([(m['gibbs_enum_best_w_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
        sum_d[fn]['o_min_sample1hit'] = np.sum([(m['gibbs_enum_best_o_min_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
        sum_d[fn]['o_min_sample2hit'] = np.sum([(m['gibbs_enum_best_o_min_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
        if GE_I_DO == True:
            sum_d[fn]['io_avg_sample1hit'] = np.sum([(m['gibbs_enum_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(1) and m['gibbs_enum_best_w_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
            sum_d[fn]['io_avg_sample2hit'] = np.sum([(m['gibbs_enum_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(2) and m['gibbs_enum_best_w_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
            sum_d[fn]['io_min_sample1hit'] = np.sum([(m['gibbs_enum_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(1) and m['gibbs_enum_best_o_min_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
            sum_d[fn]['io_min_sample2hit'] = np.sum([(m['gibbs_enum_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(2) and m['gibbs_enum_best_o_min_loss'] >= NORMAL_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
            sum_d[fn]['i_avg_sample1hit'] = np.sum([(m['gibbs_enum_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(1)) for m in lis_tgt_w[fn]])
            sum_d[fn]['i_avg_sample2hit'] = np.sum([(m['gibbs_enum_best_i_w_loss'] >= GE_I_WORD_AVG_LOSS - math.log(2)) for m in lis_tgt_w[fn]])
     
        save_fn = save_dir + '/adv_gibbsenum_' + LOG_SUC + '_' + fn[:-4] + '_' + 'RanT{}EnumT{}MaxDecayW{}LambdaGEI{}GEIAvgSampleDecay{}Care{}Iterco{}'.format(str(GIBBSENUM_RANDOM_TIME), str(GIBBSENUM_E_NUM), str(GIBBSENUM_DECAYLOSS_WEIGHT), str(GE_I_LAMBDA), str(GE_I_SAMPLE_LOSSDECAY), care_mode, GE_MAX_ITER_CO) + '.data'
        logger.info('sum_d[%s]: %s', fn, str(sum_d[fn]))
        logger.info('saving lis_tgt_w[%s] to %s', fn, save_fn)
        torch.save(lis_tgt_w[fn], save_fn)

    logger.info('sum_d: %s', str(sum_d))
    

