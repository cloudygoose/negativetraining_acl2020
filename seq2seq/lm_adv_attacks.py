import torch 
import numpy as np
from torch.autograd import Variable
import os, sys
import logging
import math
from models import LSTMLM_onehot
from myutil import *
import nltk
import lib_pdf

logger = logging.getLogger()

def lm_model_forward(model, inputv, targetv, b_len, vocab, do_train = False):
    if do_train == False:
        model.eval()
    else:
        model.train()
    
    bz = inputv.size(0)
    maskv = Variable(mask_gen(b_len, ty = 'Float')).cuda()
    #size(batch, length)
    output, _ = model(idx2onehot(inputv, len(vocab)), model.initHidden(batch_size = inputv.size(0)))
    
    w_logpdf = lib_pdf.logsoftmax_idxselect(output.view(-1, len(vocab)), targetv.contiguous().view(-1)).view(bz, -1)
    w_logpdf = w_logpdf * maskv

    return w_logpdf

def lm_adv_check(model, targetfn_list, config):
    logger.info('start lm_adv_check')
    BATCH_SIZE, vocab, vocab_inv, SEQ_LEN = config['BATCH_SIZE'], config['vocab'], config['vocab_inv'], config['SEQ_LEN']
    LM_NORMAL_AVGLOGP = config['LM_NORMAL_AVGLOGP'] 

    bz = BATCH_SIZE
    hit_lists = {'o_min':[], 'o_avg':[]}
    for targetfn in targetfn_list:
        logger.info('targetfn now: %s', targetfn)
        b_co, all_num = 0, 0
        batches = MyBatchSentences_v2([targetfn], BATCH_SIZE, config['SEQ_LEN'], vocab_inv)
        for b_idx, b_w, b_len in batches:
            inputv = Variable(b_idx[:, :-1]).cuda() 
            targetv = Variable(b_idx[:, 1:]).cuda() 
            w_logpdf = lm_model_forward(model, inputv, targetv, b_len, vocab, do_train = False)
            for i in range(bz):
                logps = w_logpdf[i][:b_len[i]].detach().cpu().numpy()    
                if np.min(logps) >= LM_NORMAL_AVGLOGP:
                    hit_lists['o_min'].append(b_w[i])
                if np.mean(logps) >= LM_NORMAL_AVGLOGP:
                    hit_lists['o_avg'].append(b_w[i])
            b_co += 1
            all_num += len(b_len)
            #print len(hit_lists['o_min']), len(hit_lists['o_avg'])
        logger.info('all_num: %d hit_rate(percentage) o_min: %f o_avg: %f', all_num, len(hit_lists['o_min']) * 100.0 / all_num, len(hit_lists['o_avg']) * 100.0 / all_num)
        for ss in ['o_min', 'o_avg']:
            if len(hit_lists[ss]) > 0:
                for j in range(min(len(hit_lists[ss]), 5)):
                    logger.info('%s example: %s', ss, ' '.join(hit_lists[ss][j]))

