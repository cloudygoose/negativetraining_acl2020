import torch 
import numpy as np
from torch.autograd import Variable
import os, sys
import logging
import math
from myutil import *
#from models.attnSeq2seq import Seq2SeqAttentionSharedEmbedding
import nltk
import models
import lib_pdf
import torch.nn.functional as F
#import advidx_draw
import adv_attacks
import advinput_seq2seq
from advinput_seq2seq import get_adv_seq2seq_mb

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

import text_eval

DATA_SET = 'swda_dialogue' #'ptb_chars' #'swda_dialogue' #'ubuntu_dialogue' #'dailydialogue'
COMMAND = 'post_advtrain'

EXP_ROOT = '../exps/'

import socket
print('hostname:', socket.gethostname())

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print 'CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES']

MT = 'attention' #'latent' or 'attention'
if len(sys.argv) > 1:
    print 'execing sys.argv[1] for setting:', sys.argv[1]
    exec(sys.argv[1])

BASE_ITER_NUM = 20 #iteration num for loading baseline model

ADV_TARGET_FN_TRAIN = '../para_adv_lists/res_500/train_500_cp1.txt' 
ADV_TARGET_FN_TEST = '../para_adv_lists/res_500/test_500.txt'
ADV_LAMBDA = 1
ADV_ATTACK = 'gibbs_enum' #'random' #gibbs_enum
DO_INITIAL_TEST = True
MIDDLE_FOCUS = False
FREQ_AVOID_LIS = ['</s>', 'you', 'i', 'me', 'are', 'to', 'do']
FREQ_AVOID = True

if DATA_SET == 'ubuntu_dialogue_np':
    TRAIN_FN = '../data/ubuntuDialog/res_np/dialogues.200k.train.txt'
    VOCAB_FN = '../data/ubuntuDialog/res/vocab_choose.txt'
    VALID_FN = '../data/ubuntuDialog/res_np/dialogues.5k.valid.txt'
    TEST_FN = '../data/ubuntuDialog/res_np/dialogues.5k.test.txt'
    
    #ADV_TARGET_FN = '../adv_lists/ubuntu_dialogue_np/' + 'mal_words_all_500P1_2.txt' #'normal_samples_att_h500.txt' #'mal_words_all_500P1_2.txt'
   
    GE_I_LM_FN = '../exps//201806_adversarial_seq2seq/lm_baseline/ubuntu_lm_np/LSTM_LR1H600L1DR0OPTsgd/iter20.checkpoint'
    if MT == 'latent': NORMAL_WORD_AVG_LOSS = -4.243; GE_I_WORD_AVG_LOSS = -4.194185
    if MT == 'attention': NORMAL_WORD_AVG_LOSS = -4.085891; GE_I_WORD_AVG_LOSS = -4.194185
    #if MT == 'latent': NORMAL_WORD_AVG_LOSS = -3.953987; GE_I_WORD_AVG_LOSS = -4.1221
    #if MT == 'attention': NORMAL_WORD_AVG_LOSS = -3.930928; GE_I_WORD_AVG_LOSS = -4.1221
    
    DATA_CONFIG = {'filter_turn': 1, 'skip_unk': False} #2} #only the answering turns
    LOG_INTERVAL = 100
    BATCH_SIZE = 64
    ADV_BATCH_SIZE = 100
    ADV_CARE_MODE = 'sample_min'
    ADV_I_LM_FLAG = False
    #one working config, uses SGD
    OPT = 'sgd'
    DROPOUT = 0
    HIDDEN_SIZE = 600
    EMBED_SIZE = 300
    LAYER_NUM = 1
    BASE_LR = 1
    POS_LR = 0.01 #ADV_LR
    NEG_LR = 0.1
    POSITIVE_RATIO = 1  
    ITER_NUM = 20 #ITER_NUM
    TEST_ITER = ITER_NUM
    #HALF_LR_ITER = 10

    HIS = 1 #2 #2
    SRC_SEQ_LEN = 15 #30 for his2
    TGT_SEQ_LEN = 20
elif DATA_SET == 'os_dialogue_np':
    TRAIN_FN = '../data/opensubtitles/res_np/train.5k.txt'
    VOCAB_FN = '../data/opensubtitles/vocab.h30k'
    VALID_FN = '../data/opensubtitles/res_np/valid.50.txt'
    TEST_FN = '../data/opensubtitles/res_np/test.50.txt'
    
    GE_I_LM_FN = '../exps//201806_adversarial_seq2seq/lm_baseline/os_lm_np/LSTM_LR0.1H600L1DR0OPTsgd/iter20.checkpoint'
    GE_I_WORD_AVG_LOSS = -4.314027 #for test.100 -4.2794;
    if MT == 'latent': NORMAL_WORD_AVG_LOSS = None #for test.100 -4.218682;
    if MT == 'attention': NORMAL_WORD_AVG_LOSS = -4.260035;  #for test.100 -4.221512;    
     
    DATA_CONFIG = {'filter_turn': 1, 'skip_unk': False} #2} #only the answering turns
    LOG_INTERVAL = 100
    BATCH_SIZE = 64
    ADV_BATCH_SIZE = 100
    ADV_CARE_MODE = 'sample_min'
    ADV_I_LM_FLAG = False
    #one working config, uses SGD
    OPT = 'sgd'
    DROPOUT = 0
    HIDDEN_SIZE = 600
    EMBED_SIZE = 300
    LAYER_NUM = 1
    NEG_LR = 0.001 #ADV_LR, for test_500 (0.001, 0.001) gives more stable PPL
    POS_LR = 0.001 #POS_LR larger than 0.0001 will make test PPL fluctuate for os data, even if only doing positive training
    POSITIVE_RATIO = 1 #Other than 0.01 0.01, The config of neg0.001 pos0.0001 seems to work well, without much PPL loss
    BASE_LR = 0.1
    ITER_NUM = 20
    TEST_ITER = ITER_NUM

    HIS = 1
    SRC_SEQ_LEN = 15 #30 for his2 #15 for his1
    TGT_SEQ_LEN = 20
elif DATA_SET == 'swda_dialogue':
    TRAIN_FN = '../data/swda_dialogue/process_oneline/train/train.txt'
    VOCAB_FN = '../data/swda_dialogue/vocab_h10k.txt'
    VALID_FN = '../data/swda_dialogue/process_oneline/valid_25/valid.txt'
    TEST_FN = '../data/swda_dialogue/process_oneline/test_25/test.txt'
    
    GE_I_LM_FN = '../exps/lm_baseline/swda_lm/LSTM_LR1H600L1DR0.3OPTsgd/iter20.checkpoint'
 
    GE_I_WORD_AVG_LOSS = -3.792652 
    if MT == 'attention': NORMAL_WORD_AVG_LOSS = -3.756853 
     
    DATA_CONFIG = {'filter_turn': 1, 'skip_unk': False} #all turns

    LOG_INTERVAL = 100
    BATCH_SIZE = 32
    #one working config, uses SGD
    OPT = 'sgd'
    DROPOUT = 0.3
    HIDDEN_SIZE = 600
    EMBED_SIZE = 300
    LAYER_NUM = 1
    BASE_LR = 1
    POS_LR = 0.01 #ADV_LR
    NEG_LR = 0.01
    POSITIVE_RATIO = 1  
    ADV_CARE_MODE = 'sample_avg'
 
    ITER_NUM = 20
    HALF_LR_ITER = 10
    ADV_BATCH_SIZE = 100
    ADV_I_LM_FLAG = False

    HIS = 1 #2
    SRC_SEQ_LEN = 15 #30 for his2
    TGT_SEQ_LEN = 20
else:
    sys.exit(1)

START_ITER = 0
START_LR = None

SUC = ''
LOG_SUC = ''
ADV_MODE = 'sigmoid_idx'
WEIGHT_LOSS_DECAY = 0.01
SOFTMAX_IDX_EL = -1 #(for large ubuntu dataset)due to optimization diffculty, I only optimize on effective length vocab #after some testing on good_words, 125 seems to be good
ADV_SRC_LEN_TRY = SRC_SEQ_LEN
LASSO_LAMBDA = 1
DRAW_FN = 'figs/tmp.jpg'
FORCE_ONEHOT = True #force to be onehot when testing

ENUM_START_LENGTH = 1
ENUM_END_LENGTH = 100
ENUM_START_RATE = 0
ENUM_END_RATE = 1

GIBBSENUM_E_NUM = 100 #100 #-1 means not activated
GIBBSENUM_RANDOM_TIME = 1 #5
GIBBSENUM_MAX_ITER_CO = 5
#{R10E100:203 R1E100:187 R10E100:175}
GIBBSENUM_I_LM_LAMBDA = 1
BEAM_SIZE = 20
#POSITIVE_RATIO = 1 #RATIO0 will make test_PPL become worse. RATIO1 gives better performance than RATIO10 #one adversarial batch, then POSITIVE_RATIO positive mbs
if len(sys.argv) > 1:
    print 're-execing sys.argv[1] for setting:', sys.argv[1]
    exec(sys.argv[1])

torch.manual_seed(1234) #just to be different from the random generator
#'ILM' + str(ADV_I_LM_FLAG) +
DD = DATA_SET
if DD[-4:] == '_gen': DD = DD[:-4]
    
if MT == 'latent':
    base_save_dir = EXP_ROOT + '/201806_adversarial_seq2seq/mle_baseline/' + DD + '/LSTM_' + 'LR' + str(BASE_LR) + 'BA' + str(BATCH_SIZE) + 'EM' + str(EMBED_SIZE) + 'H' + str(HIDDEN_SIZE) + 'L' + str(LAYER_NUM) + 'DR' + str(DROPOUT) + 'OPT' + str(OPT) + '_HIS' + str(HIS) + '_SLEN' + str(SRC_SEQ_LEN) + '_ITERNUM' + str(BASE_ITER_NUM) #+ SUC
if MT == 'attention':
    base_save_dir = EXP_ROOT + '/201806_adversarial_seq2seq/mle_att_baseline/' + DD + '/LSTM_' + 'LR' + str(BASE_LR) + 'BA' + str(BATCH_SIZE) + 'EM' + str(EMBED_SIZE) + 'H' + str(HIDDEN_SIZE) + 'L' + str(LAYER_NUM) + 'DR' + str(DROPOUT) + 'OPT' + str(OPT) + '_HIS' + str(HIS) + '_SLEN' + str(SRC_SEQ_LEN) + '_ITERNUM' + str(BASE_ITER_NUM) #+ SUC

add = ''
if MIDDLE_FOCUS == True:
    add = add + 'MFOCUS' + str(MIDDLE_FOCUS)
if FREQ_AVOID == True:
    add = add + 'FAVOID' + str(FREQ_AVOID)
if ADV_I_LM_FLAG == True:
    add = add + 'ILM' + str(ADV_I_LM_FLAG) + 'ILAM' + str(GIBBSENUM_I_LM_LAMBDA)
save_dir = EXP_ROOT + '/201806_adversarial_seq2seq/post_advtrain_seq2seq/' + DATA_SET + '/' + 'MT' + MT + '/ADV' + str(ADV_ATTACK) + 'E' + str(GIBBSENUM_E_NUM) + 'R' + str(GIBBSENUM_RANDOM_TIME) + 'MIC' + str(GIBBSENUM_MAX_ITER_CO) + 'IC' + str(ITER_NUM) + 'PRATIO' + str(POSITIVE_RATIO) + 'LAMBDA' + str(ADV_LAMBDA) + add + 'CM' + str(ADV_CARE_MODE) + 'NEGLR' + str(NEG_LR) + 'POSLR' + str(POS_LR) + 'BA' + str(ADV_BATCH_SIZE) + '_LSTM_' + 'BASELR' + str(BASE_LR) + 'BA' + str(BATCH_SIZE) + 'EM' + str(EMBED_SIZE) + 'H' + str(HIDDEN_SIZE) + 'L' + str(LAYER_NUM) + 'DR' + str(DROPOUT) + 'OPT' + str(OPT) + '_HIS' + str(HIS) + '_SLEN' + str(SRC_SEQ_LEN) + 'TRAINF' + os.path.basename(ADV_TARGET_FN_TRAIN)[:-4] + 'TESTF' + os.path.basename(ADV_TARGET_FN_TEST)[:-4] + 'SUC' + SUC

log_fn = save_dir + '/logC' + COMMAND + LOG_SUC
print 'save_dir is', save_dir
print 'log_fn is', log_fn

os.system('mkdir -p ' + save_dir)

logging.basicConfig(level = 0)
logger = logging.getLogger()
setLogger(logger, log_fn)

vocab, vocab_inv = getVocab(VOCAB_FN)
logger.info('len of vocab: %d', len(vocab))

m_embed = nn.Linear(len(vocab), EMBED_SIZE).cuda()
m_encode_w_rnn = nn.LSTM(EMBED_SIZE, HIDDEN_SIZE, LAYER_NUM, batch_first = False, dropout = DROPOUT).cuda() #I'm not using batch_first to get easy on dataparallel!!
if MT == 'latent':
    m_decode_w_rnn = models.RNNLatentDecoder(EMBED_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, vocab, vocab_inv, TGT_SEQ_LEN, dropout_rate = DROPOUT, layer_num = LAYER_NUM).cuda()
if MT == 'attention':
    m_decode_w_rnn = models.RNNAttDecoder(EMBED_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, vocab, vocab_inv, TGT_SEQ_LEN, dropout_rate = DROPOUT, layer_num = LAYER_NUM).cuda()

m_embed_dp, m_encode_w_rnn_dp, m_decode_w_rnn_dp = m_embed, m_encode_w_rnn, m_decode_w_rnn

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    print("MultiGPU is not fully understood and supported yet.")
    m_embed_dp = nn.DataParallel(m_embed, dim = 1)
    m_encode_w_rnn_dp = nn.DataParallel(m_encode_w_rnn, dim = 0)
    m_decode_w_rnn_dp = nn.DataParallel(m_decode_w_rnn, dim = 0)

m_dict = {'m_embed': m_embed, 'm_embed_dp': m_embed_dp, 
    'm_encode_w_rnn': m_encode_w_rnn, 'm_encode_w_rnn_dp': m_encode_w_rnn_dp,
    'm_decode_w_rnn': m_decode_w_rnn, 'm_decode_w_rnn_dp': m_decode_w_rnn_dp,
}

if START_ITER > 0:
    MODEL_FILE = save_dir + '/it{}.checkpoint'.format(START_ITER - 1)
    logger.info('loading form %s.', MODEL_FILE)
    load_d = torch.load(MODEL_FILE)
    for nn in m_dict:
        if not nn.endswith('_dp'):
            logger.info('loading for %s', nn)
            m_dict[nn].load_state_dict(load_d[nn])

all_params = set()
for m in m_dict.values():
    all_params = all_params | set(m.parameters())

adv_config = {
    'MT': MT,
    'HIDDEN_SIZE': HIDDEN_SIZE, 
    'vocab': vocab,
    'vocab_inv': vocab_inv,
    'ADV_CARE_MODE': ADV_CARE_MODE,
    'ADV_I_LM_FLAG': ADV_I_LM_FLAG,
    'ADV_I_LM': None,
    'SRC_SEQ_LEN': SRC_SEQ_LEN,
    'TGT_SEQ_LEN': TGT_SEQ_LEN,
    'LAYER_NUM': LAYER_NUM,
    'NORMAL_WORD_AVG_LOSS': NORMAL_WORD_AVG_LOSS, 
    'GE_I_WORD_AVG_LOSS': GE_I_WORD_AVG_LOSS, 
    'GIBBSENUM_E_NUM': GIBBSENUM_E_NUM,
    'GIBBSENUM_RANDOM_TIME': GIBBSENUM_RANDOM_TIME,
    'GIBBSENUM_MAX_ITER_CO': GIBBSENUM_MAX_ITER_CO,
    'GIBBSENUM_I_LM_LAMBDA': GIBBSENUM_I_LM_LAMBDA,
    'BEAM_SIZE': BEAM_SIZE,
    'MIDDLE_FOCUS': MIDDLE_FOCUS,
}

def inf_data_gen(file_lis, bz, ty = 'LM', name = ''): 
    assert(ty == 'LM' or ty == 'DIALOGUE')
    iter_count = 0
    while True:
        if ty == 'LM':
            batches = MyBatchSentences_v2(file_lis, bz, TGT_SEQ_LEN, vocab_inv)
            for b_idx, b_w, b_len in batches:
                yield b_idx, b_w, b_len
        if ty == 'DIALOGUE':
            batches = DialogueBatches(file_lis, bz, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS)
            for src_mb, tgt_mb, tgt_len, src_w, tgt_w in batches:
                yield src_mb, tgt_mb, tgt_len, src_w, tgt_w 
        iter_count = iter_count + 1
        logger.info('data sweep time %d name: %s', iter_count, name)

def mask_gen(lengths):
    max_len = lengths[0]
    size = len(lengths)
    mask = torch.ByteTensor(size, max_len).zero_()
    for i in range(size):
        mask[i][:lengths[i]].fill_(1)
    return mask

def get_decay_co(batch_logpdf, w_logit_rnn, tgt_mb, tgt_len, adv_config):
    ADV_CARE_MODE, NORMAL_WORD_AVG_LOSS = adv_config['ADV_CARE_MODE'], adv_config['NORMAL_WORD_AVG_LOSS']
    assert(ADV_CARE_MODE == 'sample_min' or ADV_CARE_MODE == 'sample_avg' or ADV_CARE_MODE == 'max')
    bz = batch_logpdf.size(0)
    co = [0.01 for kk in range(bz)]
    _, pred = torch.max(w_logit_rnn, 2)
    #print batch_logpdf[:10], tgt_len[:10]
    #sys.exit(1)
    for i in range(bz):
        if ADV_CARE_MODE == 'max':
            co[i] = 1
            for j in range(tgt_len[i]):
                if pred[i][j] != tgt_mb[i][j + 1]: #tgt_mb starts with <s>
                    #print 'pred:', vocab[pred[i][j]], 'tgt:', vocab[tgt_mb[i][j + 1]]
                    co[i] = 0.01; break
            #if co[i] == 1:
            #    print 'max hit!', ' '.join([vocab[tgt_mb[i][j+1]] for j in range(tgt_len[i])])
        if ADV_CARE_MODE == 'sample_min' and torch.min(batch_logpdf[i][:tgt_len[i]]).item() > NORMAL_WORD_AVG_LOSS:
            co[i] = 1
        if ADV_CARE_MODE == 'sample_avg' and torch.sum(batch_logpdf[i][:tgt_len[i]]).item() / tgt_len[i] > NORMAL_WORD_AVG_LOSS:
            co[i] = 1
    return torch.FloatTensor(co).cuda()

def adv_train(adv_batches, positive_batches, opts, do_train = True, do_log = True, do_adv = False, adv_config = None):
    #adv_train put adversarial training first
    #assert(do_train == True)
    opt_pos, opt_neg = opts
    all_loss, all_num, adv_all_loss, adv_all_num, nodecay_sennum, success0_co, success1_co = 0, 0, 0, 0, 0, 0, 0
    b_count, adv_b_count, adv_sen_count = 0, 0, 0
    attack_results = []
    loss_sen = []
    all_target_set = {}
    for target_mb in adv_batches:
        adv_src_mb, adv_tgt_mb, adv_tgt_len, adv_src_lis, adv_tgt_lis = get_adv_seq2seq_mb(target_mb, ADV_ATTACK, m_dict, adv_config)
        for l in adv_tgt_lis: all_target_set[' '.join(l[1:])] = True
        nondecay_lis_now = []
        co_now = 0
        while 1 == 1:
            #print tgt_lis[0], tgt_mb[0], tgt_len[0]
            for m in m_dict: #get_adv_seq2seq_mb could change the flags
                m_dict[m].zero_grad()
                m_dict[m].train() 
            aux = {}; adv_batch_logpdf = models.encoder_decoder_forward(adv_src_mb, adv_tgt_mb, adv_tgt_len, m_dict, adv_config, aux_return = aux); w_logit_rnn = aux['w_logit_rnn']; 
            decay_co = get_decay_co(adv_batch_logpdf, w_logit_rnn, adv_tgt_mb, adv_tgt_len, adv_config)
            attack_results.append((adv_src_mb, adv_tgt_mb, adv_tgt_len, adv_src_lis, adv_tgt_lis, decay_co, adv_batch_logpdf))
            final_mask = torch.FloatTensor(adv_batch_logpdf.size()).cuda()
            final_mask[:] = 1
            if MIDDLE_FOCUS == True:
                for i in range(len(adv_tgt_len)):
                    final_mask[i][0] = 0
                    final_mask[i][adv_tgt_len[i] - 1] = 0
            if FREQ_AVOID == True:
                for i in range(len(adv_tgt_len)):
                    for j in range(adv_tgt_len[i]):
                        if adv_tgt_lis[i][j + 1] in FREQ_AVOID_LIS:
                            final_mask[i][j] = 0
                """
                for i in range(20):
                    print i, 'adv_tgt_lis:', adv_tgt_lis[i]
                    print 'mask:', final_mask[i]
                    print 'adv_tgt_len', adv_tgt_len[i]
                    print 'adv_batch_logpdf', adv_batch_logpdf[i]
                    print '*final_mask', (adv_batch_logpdf * final_mask)[i]
                """
            
            sen_logpdf = torch.sum(adv_batch_logpdf * final_mask, dim = 1) * decay_co
            (ADV_LAMBDA * torch.sum(sen_logpdf) / sum(adv_tgt_len)).backward()
            for m in m_dict.values():
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5)
            if do_train == True:
                opt_neg.step()
            
            """
            #just for debug!!
            for i in range(decay_co.size(0)):
                if decay_co[i] == 1:
                    _, tgt_w, _ = target_mb
                    s_lis = adv_src_mb[i].cpu().numpy().tolist()
                    print 'hit! target:', ' '.join(tgt_w[i]), 't_input:', ' '.join([vocab[ii] for ii in s_lis])
            print 'hit hit hit in first mb'
            """

            nondecay_num_now = np.sum(decay_co.cpu().numpy() == 1)
            #logger.info('co_now: %d nondecay_num_now: %d', co_now, nondecay_num_now)
            nondecay_lis_now.append(nondecay_num_now)
            if co_now == 0:
                logger.info('displaying hit targets')
                for i in range(len(adv_tgt_len)):
                    if decay_co[i] == 1:
                        print "hit! target:", ' '.join(adv_tgt_lis[i]), 'trigger:', ' '.join(adv_src_lis[i])
            if co_now == 0:
                adv_sen_count += adv_src_mb.size(0)
                adv_all_loss += torch.sum(adv_batch_logpdf).item()
                adv_all_num += sum(adv_tgt_len)
                nodecay_sennum += nondecay_num_now
                adv_b_count += 1
                if do_train == False:
                    break
            co_now += 1

            for kk in range(POSITIVE_RATIO):
                src_mb, tgt_mb, tgt_len, src_w, tgt_w = positive_batches.next()
                loss = 0
                b_count = b_count + 1
                bz = src_mb.size(0)
                all_num = all_num + sum(tgt_len)
                
                batch_logpdf = models.encoder_decoder_forward(src_mb, tgt_mb, tgt_len, m_dict, adv_config)
                
                w_loss_rnn = torch.sum(batch_logpdf)
                loss_sen.extend(torch.sum(batch_logpdf, dim = 1).detach().cpu().numpy().tolist())
                    
                all_loss = all_loss + w_loss_rnn.data.item()

                if do_train == True:
                    for m in m_dict.values():
                        m.zero_grad()
                    (- w_loss_rnn / sum(tgt_len)).backward()            
                    for m in m_dict.values():
                        torch.nn.utils.clip_grad_norm_(m.parameters(), 5)
                    opt_pos.step()
            if nondecay_num_now == 0 or co_now > 20: 
                if nondecay_num_now <= 0: success0_co += 1
                if nondecay_num_now <= 1: success1_co += 1
                break 
        logger.info('current mb nondecay_lis_now(through each update): %s', str(nondecay_lis_now))
                
    logger.info('all_num: %d', all_num)
    if all_num > 0:
        logger.info('sen_avg_loss: %f', np.mean(loss_sen))
    if do_adv == True:
        logger.info('iter: %d adv_all_num: %d avg_nodecay_sennum: %d success0_rate: %f success1_rate: %f', adv_config['iter_now'], adv_all_num, float(nodecay_sennum * 1.0) / adv_b_count, float(success0_co * 100.0 / adv_b_count), float(success1_co * 100.0 / adv_b_count))
        logger.info('debug for avg_nodecay_sennum: %d / %d', nodecay_sennum, adv_b_count)
        logger.info('avg_adv_loss: %f', adv_all_loss / adv_all_num)

    res = {
        'positive_avg_loss': 0 if all_num == 0 else all_loss * 1.0 / all_num,
        'nodecay_sennum': nodecay_sennum,
        'adv_sen_count': adv_sen_count,
        'attack_results': attack_results,
        'all_target_set': all_target_set,
    }
    return res

def mle_train(batches, opt, do_train = True, do_log = True, do_adv = False, adv_config = None):
    for m in m_dict:
        if do_train == True:
            m_dict[m].train()
        else:
            m_dict[m].eval()
    
    if do_adv == True:
        inf_adv_batches = adv_config['inf_adv_batches']

    all_loss, all_num, adv_all_loss, adv_all_num, nodecay_sennum = 0, 0, 0, 0, 0
    b_count, adv_b_count = 0, 0
    loss_sen = []
    for src_mb, tgt_mb, tgt_len, src_w, tgt_w in batches:
        #print src_w[0], src_w[1]; sys.exit(1)
        loss = 0
        b_count = b_count + 1
        bz = src_mb.size(0)
        all_num = all_num + sum(tgt_len)
        
        batch_logpdf = models.encoder_decoder_forward(src_mb, tgt_mb, tgt_len, m_dict, adv_config)
        
        if 1 == 0:
            #===drawing debug===
            logprob_lis = []
            w_lis = []
            for i in range(bz / 2):
                ss = ""
                for j in range(tgt_len[i]):
                    ss = ss + tgt_w[i][j + 1] + '(' + '%.2f' % batch_logpdf[i][j].item() + ') '
                    logprob_lis.append(batch_logpdf[i][j].item())
                    if batch_logpdf[i][j] < -1000:
                        print tgt_w[i][j + 1], batch_logpdf[i][j]
                    w_lis.append(tgt_w[i][j + 1])
                #print ss
            #torch.save(logprob_lis, 'figs/advtrain_wordlogps/naivetrainR' + str(ADV_RATIO) + '_' + SUC + '.data')
            #torch.save(w_lis, 'figs/advtrain_wordlogps/naivetrainR' + str(ADV_RATIO) + '_' + SUC + '_w.data')        
            #sys.exit(1)
            #===end===
        
        #print torch.min(batch_logpdf)

        w_loss_rnn = torch.sum(batch_logpdf)
        loss_sen.extend(torch.sum(batch_logpdf, dim = 1).detach().cpu().numpy().tolist())
            
        all_loss = all_loss + w_loss_rnn.data.item()

        if do_train == True:
            for m in m_dict.values():
                m.zero_grad()
            (- w_loss_rnn / sum(tgt_len)).backward()            
            for m in m_dict.values():
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5)
            opt.step() 
        
        if do_train == True and do_adv == True and b_count % ADV_RATIO == 0:
            target_mb = next(inf_adv_batches)
            src_mb, tgt_mb, tgt_len, src_lis, tgt_lis = get_adv_seq2seq_mb(target_mb, ADV_ATTACK, m_dict, adv_config)
            #print tgt_lis[0], tgt_mb[0], tgt_len[0]
            for m in m_dict: #get_adv_seq2seq_mb could change the flags
                m_dict[m].zero_grad()
                m_dict[m].train() 
            adv_batch_logpdf = models.encoder_decoder_forward(src_mb, tgt_mb, tgt_len, m_dict, adv_config)
            decay_co = get_decay_co(adv_batch_logpdf, tgt_len, adv_config)
            nodecay_sennum += np.sum(decay_co.cpu().numpy() == 1)
            adv_b_count += 1
            sen_logpdf = torch.sum(adv_batch_logpdf, dim = 1) * decay_co
            (ADV_LAMBDA * torch.sum(sen_logpdf) / sum(tgt_len)).backward()
            for m in m_dict.values():
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5)
            opt.step() 
            adv_all_loss += torch.sum(adv_batch_logpdf).item()
            adv_all_num += sum(tgt_len)
   
        if do_log == True and b_count % LOG_INTERVAL == 0:
            logger.info('avg loss at b: %d , %f', b_count, all_loss * 1.0 / all_num)

    logger.info('all_num: %d', all_num)
    logger.info('sen_avg_loss: %f', np.mean(loss_sen))
    if do_adv == True:
        logger.info('adv_all_num: %d avg_nodecay_sennum: %d', adv_all_num, float(nodecay_sennum * 1.0) / adv_b_count)
        logger.info('avg_adv_loss: %f', adv_all_loss / adv_all_num)
    if all_num != 0:
        return float(all_loss * 1.0 / all_num)
    else:
        return 0

def sample_compare_bleu(batches, outfn = None, ty = 'max', extend_num = 1):
    if outfn != None:
        logger.info('writing decoding result to %s ty: %s', outfn, ty)
        outfn = open(outfn, 'w') 
    if ty == 'max' and extend_num > 1:
        logger.info('something wrong with extend_num')
        sys.exit(1) #somthing wrong
    bleu = {}; bleu[2] = []; bleu[3] = []; 
    b_co = 0
    s_co = 0
    for src_mb, tgt_mb, tgt_len, src_w, tgt_w in batches: 
        b_co = b_co + 1
        bz = src_mb.size(0)
        input_src = Variable(src_mb).cuda()
        input_src = torch.cat([input_src] * extend_num, dim = 0)
        output, _ = m_encode_w_rnn(m_embed(idx2onehot(input_src, len(vocab))).permute(1,0,2), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM))
        output = output.permute(1, 0, 2)
        maxlen = tgt_mb.size(1)
        
        if MT == 'latent': latent = output[:, -1, :].unsqueeze(1).repeat(1, tgt_mb.size(1), 1)
        if MT == 'attention': latent = output
        
        if ty == 'sample_min':
            if MT == 'attention': samples, _ = m_decode_w_rnn.generate_samplemin(latent, maxlen, NORMAL_WORD_AVG_LOSS)
            if MT == 'latent': samples = m_decode_w_rnn.generate_samplemin(latent, maxlen, NORMAL_WORD_AVG_LOSS)
        else:
            samples = m_decode_w_rnn.generate(latent, maxlen, sample = (False if ty == 'max' else True))
        samples = samples.cpu().numpy().tolist()
        
        #print samples
        if extend_num > 1:
            samples_best = range(src_mb.size(0)) #placeholder
            for i in range(src_mb.size(0)):
                best_score = -1
                for k in range(extend_num):
                    idx = i + k * src_mb.size(0)
                    b = nltk.translate.bleu_score.sentence_bleu([tgt_w[i]], samples[idx])
                    if b > best_score:
                        samples_best[i] = samples[idx]
                        best_score = b
            samples = samples_best 
        for i in range(bz):
            samples[i] = clean_sen([vocab[kk] for kk in samples[i]])
            tgt_w[i] = clean_sen(tgt_w[i])
            if b_co <= 1 and i <= 5:
                logger.info('src%d: %s', i, ' '.join(src_w[i]))
                logger.info('ref%d: %s', i, ' '.join(tgt_w[i]))
                logger.info('sample%d: %s', i,  ' '.join(samples[i]))
            if outfn != None:
                outfn.write('src' + str(s_co) + ': ' + ' '.join(src_w[i]) + '\n')
                outfn.write('ref' + str(s_co) + ': ' + ' '.join(tgt_w[i]) + '\n')
                outfn.write('sample' + str(s_co) + ': ' + ' '.join(samples[i]) + '\n')
                s_co = s_co + 1
            for ngram in [2,3]:
                weight = tuple((1. / ngram for _ in range(ngram)))
                b = nltk.translate.bleu_score.sentence_bleu([tgt_w[i]], samples[i], weight) 
                bleu[ngram].append(b)
        """
        refs = get_lines(TEST_FILE)
     
        bleu = {}; bleu[2] = []; bleu[3] = [];
        for sa in range(num):
            sample = rnn.sampleOne(SEQ_LEN)
            sample = [vocab[idx] for idx in sample]
            #assert(sample[0] != '<s>')
            for ngram in [2,3]:
                weight = tuple((1. / ngram for _ in range(ngram)))
                b = nltk.translate.bleu_score.sentence_bleu(refs, sample, weight) 
                bleu[ngram].append(b)
        """
    
    res = {'bleu2':np.mean(bleu[2]), 'bleu3':np.mean(bleu[3])} 
    logger.info('ty: %s extend_num: %d result %s', str(ty), extend_num, str(res)) 
    if outfn != None:
        outfn.close()
    return res

def test_entropy(it):
    logger.info('it: %d testing entropy of TEST_FN', it)
    batches_test = DialogueBatches([TEST_FN], 20, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS) 
    res = models.get_samples(batches_test, 'max', m_dict, adv_config)
    #for i in range(50):
    #    print (' '.join(res['sample_lis'][i]))
    ref_lis, sample_lis = res['ref_lis'], res['sample_lis']
    for ss in ('ref_lis', 'sample_lis'):
        for kk in (2, 3):
            entro = text_eval.text_entropy(res[ss], kk)
            logger.info('it: %d text_entropy(TEST_FN) for %s with k %d: %f', it, ss, kk, entro)
    return res

if COMMAND.startswith('test'): #test
    MODEL_FILE = save_dir + '/it{}.checkpoint'.format(ITER_NUM)
    logger.info('loading form %s.', MODEL_FILE)
    save_d = torch.load(MODEL_FILE)
    for nn in m_dict:
        if not nn.endswith('_dp'):
            m_dict[nn].load_state_dict(save_d[nn])
            m_dict[nn].eval()
    logger.info('doing ppl test')
    batches_test = DialogueBatches([TEST_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS)
    opt = torch.optim.SGD(all_params, momentum=0.9, lr = 0, weight_decay = 1e-5)
    loss_test = mle_train(batches_test, opt, do_train = False, adv_config = adv_config)        
    logger.info('test PPL: %f log-likelihood: %f', math.exp(-loss_test), loss_test)
    batches_test = DialogueBatches([TEST_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS) 
    #ty = 'sample_min' #'sample_min' #'sample' #'max'
    #out_fn = save_dir + '/samples_test_ty' + ty + '.txt'
    #logger.info('sampling out_fn : %s', out_fn)
    #sample_compare_bleu(batches_test, ty = ty, outfn = out_fn, extend_num = 1)
    
    if COMMAND == 'testout':
        batches_test = DialogueBatches([TEST_FN], 20, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS) 
        logger.info('TEST_SAMPLE_TYPE: %s', TEST_SAMPLE_TYPE)
        res = models.get_samples(batches_test, TEST_SAMPLE_TYPE, m_dict, adv_config)
        for i in range(50):
            print (' '.join(res['sample_lis'][i]))
        sys.exit(1) #debug
 
    """ 
    for k in range(70):
        src_mb, tgt_mb, src_w, tgt_w = next(batches_test)
    logger.info('src: %s', str(src_w[0]))
    logger.info('ref: %s', str(tgt_w[0]))
    for k in range(20):
        input_src = Variable(src_mb).cuda()
        samples, logp = model.sampleBatch(input_src, TGT_SEQ_LEN, sample_type = 'sampling')
        for i in range(src_mb.size(0)):
            samples[i] = clean_sen([vocab[kk] for kk in samples[i]])
            tgt_w[i] = clean_sen(tgt_w[i])
            logger.info('sample%d: %s', i, str(samples[i]))
            logger.info('logp%d: %s', i, str(logp[i]))
    """
lstmlm = None
if ADV_I_LM_FLAG == True:
    logger.info('ADV_I_LM_FLAG is true! loading lm from %s', GE_I_LM_FN)
    lstmlm = models.LSTMLM_onehot(EMBED_SIZE, HIDDEN_SIZE, len(vocab), vocab_inv, dropout_rate = 0, layer_num = 1)
    #for name, p in rnn.state_dict().iteritems():
    lstmlm = lstmlm.cuda()
    lstmlm.load_state_dict(torch.load(GE_I_LM_FN))
    adv_config['ADV_I_LM'] = lstmlm

if COMMAND == 'post_advtrain':
    if START_ITER == 0:
        MODEL_FILE = base_save_dir + '/iter{}.checkpoint'.format(BASE_ITER_NUM)
        logger.info('START_ITER is 0, loading form base %s.', MODEL_FILE)
        save_d = torch.load(MODEL_FILE)
        for nn in m_dict:
            if not nn.endswith('_dp'):
                m_dict[nn].load_state_dict(save_d[nn])
                m_dict[nn].eval()
    
    logger.info('ADV_TARGET_FN_TRAIN: %s FN_TEST: %s ADV_CARE_MODE: %s MIDDLE_FOCUS: %s FREQ_AVOID: %s', ADV_TARGET_FN_TRAIN, ADV_TARGET_FN_TEST, ADV_CARE_MODE, str(MIDDLE_FOCUS), str(FREQ_AVOID))
    inf_positive_batches = inf_data_gen([TRAIN_FN], ADV_BATCH_SIZE, ty = 'DIALOGUE')
   
    if DO_INITIAL_TEST == True: 
        for FN, ss in [(ADV_TARGET_FN_TRAIN, 'train'), (ADV_TARGET_FN_TEST, 'test')]:
            logger.info('initial test on %s file with do_train = False!' % ss)
            opt_pos = torch.optim.SGD(all_params, momentum = 0.9, lr = POS_LR, weight_decay = 1e-5)
            opt_neg = torch.optim.SGD(all_params, momentum = 0.9, lr = NEG_LR, weight_decay = 1e-5)
            adv_batches = MyBatchSentences_v2([FN], ADV_BATCH_SIZE, TGT_SEQ_LEN, vocab_inv, do_sort = False)
            adv_config['iter_now'] = -1
            res = adv_train(adv_batches, inf_positive_batches, (opt_pos, opt_neg), do_train = False, do_adv = True, adv_config = adv_config)        
            logger.info('(initial test on %s file)it: %d, hit_num found in train: %d hit_rate: %f per target_num: %d', ss, -1, res['nodecay_sennum'], float(res['nodecay_sennum']) / res['adv_sen_count'], res['adv_sen_count'])
        test_entropy(-1)
         
    pos_lr, neg_lr = POS_LR, NEG_LR 
    for it in range(START_ITER, ITER_NUM + 1):          
        #if it > 0 and it % 5 == 0: #seems to be a bad idea for open_subtitle
        #    pos_lr, neg_lr = pos_lr * 0.5, neg_lr * 0.5  
        logger.info('starting iter %s POS_LR: %f NEG_LR: %f POSITIVE_RATIO: %d MIDDLE_FOCUS: %s FREQ_AVOID: %s ADV_I_LM_FLAG: %s', it, pos_lr, neg_lr, POSITIVE_RATIO, str(MIDDLE_FOCUS), str(FREQ_AVOID), str(ADV_I_LM_FLAG))
        logger.info('command is %s', ' '.join(sys.argv))
        #global opt
        opt_pos = torch.optim.SGD(all_params, momentum = 0.9, lr = pos_lr, weight_decay = 1e-5)
        opt_neg = torch.optim.SGD(all_params, momentum = 0.9, lr = neg_lr, weight_decay = 1e-5)
        logger.info('adv_train_fn is %s', ADV_TARGET_FN_TRAIN)
        adv_batches = MyBatchSentences_v2([ADV_TARGET_FN_TRAIN], ADV_BATCH_SIZE, TGT_SEQ_LEN, vocab_inv, do_sort = False)
        adv_config['iter_now'] = it
        res = adv_train(adv_batches, inf_positive_batches, (opt_pos, opt_neg), do_train = True, do_adv = True, adv_config = adv_config)        
        batches_test = DialogueBatches([TEST_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS)
        loss_test = mle_train(batches_test, opt_pos, do_train = False, adv_config = adv_config)        
        logger.info('it: %d, base_test PPL: %f', it, math.exp(-loss_test))
        logger.info('it: %d, hit_num found in train: %d hit_rate: %f per target_num: %d', it, res['nodecay_sennum'], float(res['nodecay_sennum']) / res['adv_sen_count'], res['adv_sen_count'])
        
        res_entro = test_entropy(it)
        save_fn = save_dir + '/it{}_testentropy_samples.save'.format(it)
        logger.info('saving testentropy_res to %s', save_fn)
        torch.save(res_entro, save_fn)
        if DATA_SET.endswith('_gen'):
            logger.info('printing all distinct samples')
            dd = {}
            for s in res_entro['sample_lis']:
                if not ' '.join(s) in dd:
                    dd[' '.join(s)] = 0
                dd[' '.join(s)] += 1
            for s in dd:
                logger.info('sample: %s time: %d ratio: %f in_targets %s', s, dd[s], float(dd[s] * 1.0) / len(res_entro['sample_lis']), str(s in res['all_target_set']))

        if it % 10 == 0 and it != 0:
            try:
                save_d = {}
                for m in m_dict:
                    save_d[m] = m_dict[m].state_dict()
                model_fn = save_dir + '/it{}.checkpoint'.format(it)
                logger.info('saving to %s', model_fn)
                torch.save(save_d, model_fn)
                save_fn = save_dir + '/it{}_attackresults.save'.format(it)
                logger.info('saving attack results to %s', save_fn)
                torch.save(res['attack_results'], save_fn)
            except IOError:
                print '!!!!!!!!!!!!!!!!!!!IOError!!!!!!!!!!!!!!!!!!!!!!!!!!'
                print '!!!!!!!!!!!!!!!!!!!IOError!!!!!!!!!!!!!!!!!!!!!!!!!!'
            
        comm = 'cp ' + log_fn + ' ' + log_fn + '_it' + str(it)
        logger.info('saving log file: %s', comm)
        os.system(comm)
        logger.info('grepping log file %s', log_fn)
        comm = 'grep ' + "'hit_num found in train:'" + ' ' + log_fn
        os.system(comm)
        comm = 'grep ' + "'test PPL:'" + ' ' + log_fn
        os.system(comm)
        logger.info('grepping log file %s for text_entropy', log_fn)
        comm = 'grep ' + "'text_entropy(TEST_FN)'" + ' ' + log_fn
        os.system(comm)
    
    logger.info('final test on test file with do_train = False! %s', ADV_TARGET_FN_TEST)
    adv_batches = MyBatchSentences_v2([ADV_TARGET_FN_TEST], ADV_BATCH_SIZE, TGT_SEQ_LEN, vocab_inv, do_sort = False)
    adv_config['iter_now'] = -1
    res = adv_train(adv_batches, inf_positive_batches, (opt_pos, opt_neg), do_train = False, do_adv = True, adv_config = adv_config)        
    logger.info('(final test) hit_num found in test: %d hit_rate: %f per target_num: %d', res['nodecay_sennum'], float(res['nodecay_sennum']) / res['adv_sen_count'], res['adv_sen_count'])
 
if COMMAND in 'new_gibbsenum':
    COMMAND = 'new_gibbsenum'
    MODEL_FILE = save_dir + '/it{}.checkpoint'.format(TEST_ITER)
    logger.info('loading form %s.', MODEL_FILE)
    save_d = torch.load(MODEL_FILE)
    for nn in m_dict: 
        if not nn.endswith('_dp'):
            m_dict[nn].load_state_dict(save_d[nn])
    
    lstmlm = None
    if ADV_I_LM_FLAG == True:
        logger.info('ADV_I_LM_FLAG is true! loading lm from %s', GE_I_LM_FN)
        lstmlm = models.LSTMLM_onehot(EMBED_SIZE, HIDDEN_SIZE, len(vocab), vocab_inv, dropout_rate = 0, layer_num = 1)
        #for name, p in rnn.state_dict().iteritems():
        lstmlm = lstmlm.cuda()
        lstmlm.load_state_dict(torch.load(GE_I_LM_FN))
        adv_config['ADV_I_LM'] = lstmlm
    
    hit_co = {
        'o_min_k1': 0
    }
    """ #not sure!
    param_use = {
        'GIBBSENUM_RANDOM_TIME': 10,
        'GIBBSENUM_MAX_ITER_CO': 5,
        'GIBBSENUM_E_NUM': 100,
    }
    for p in param_use: adv_config[p] = param_use[p]
    logger.info('We are testing! using this param set: %s, setting adv_config....', str(param_use))
    """
    logger.info('We use the same adv setting as training, to be consistent.')
    logger.info('Doing gibbsenum for %s', ADV_TARGET_FN)
    batches = MyBatchSentences_v2([ADV_TARGET_FN], ADV_BATCH_SIZE, TGT_SEQ_LEN, vocab_inv)
    for b_idx, b_w, b_len in batches:
        target_mb = b_idx, b_w, b_len
        adv_idx, stat_mb = advinput_seq2seq.adv_gibbs_enum_mb(target_mb, m_dict, adv_config)
        o_min_lis = np.array([m['gibbs_enum_best_o_min_loss'] for m in stat_mb])
        print 'o_min_lis:', o_min_lis
        hit_co['o_min_k1'] += np.sum(o_min_lis > NORMAL_WORD_AVG_LOSS)
        print 'hit_co for now:', str(hit_co)
             
if COMMAND in 'get_attackres':
    if TEST_ITER == -1:
        MODEL_FILE = base_save_dir + '/iter{}.checkpoint'.format(BASE_ITER_NUM)
    else:
        MODEL_FILE = save_dir + '/it{}.checkpoint'.format(TEST_ITER)
    print 'loading from', MODEL_FILE
    save_d = torch.load(MODEL_FILE)
    for nn in m_dict:
        if not nn.endswith('_dp'):
            m_dict[nn].load_state_dict(save_d[nn])
            m_dict[nn].eval()
    ADV_BATCH_SIZE = 50
    analyze_target_fn = '../para_adv_lists/analyze_list/test_500.shuf.h100.txt'
    #analyze_target_fn = '../para_adv_lists/analyze_list/train_500_cp1.shuf.h100.txt'
    print 'analyze_target_fn:', analyze_target_fn
    
    inf_positive_batches = inf_data_gen([TRAIN_FN], ADV_BATCH_SIZE, ty = 'DIALOGUE')
    adv_batches = MyBatchSentences_v2([analyze_target_fn], ADV_BATCH_SIZE, TGT_SEQ_LEN, vocab_inv, do_sort = False)
    adv_config['iter_now'] = TEST_ITER
    opt_pos = torch.optim.SGD(all_params, momentum = 0.9, lr = POS_LR, weight_decay = 1e-5)
    opt_neg = torch.optim.SGD(all_params, momentum = 0.9, lr = NEG_LR, weight_decay = 1e-5)
    res = adv_train(adv_batches, inf_positive_batches, (opt_pos, opt_neg), do_train = False, do_adv = True, adv_config = adv_config)       
    for mb in res['attack_results']:
        adv_src_mb, adv_tgt_mb, adv_tgt_len, adv_src_lis, adv_tgt_lis, decay_co, adv_batch_logpdf = mb
        for i in range(ADV_BATCH_SIZE):
            if decay_co[i] == 1:
                print 'hit!', ' '.join(adv_tgt_lis[i]), adv_batch_logpdf[i]
                print 'trigger input:', ' '.join(adv_src_lis[i])
    
    res_save_fn = 'figs/post_mal_advtrain/' + DATA_SET + '_IT' + str(TEST_ITER) + '_NEGLR' + str(NEG_LR) + '_POSLR' + str(POS_LR) + '_FAVOID' + str(FREQ_AVOID) + '_ILM' + str(ADV_I_LM_FLAG) + '_MFOCUS' + str(MIDDLE_FOCUS) + '_TARGET' + os.path.basename(analyze_target_fn)[:-4] + '_CARE' + ADV_CARE_MODE + '_attackres.save'
    print 'saving to res_save_fn:', res_save_fn
    torch.save(res, res_save_fn)
 
if COMMAND in 'get_testlogp':
    if TEST_ITER == -1:
        MODEL_FILE = base_save_dir + '/iter{}.checkpoint'.format(BASE_ITER_NUM)
    else:
        MODEL_FILE = save_dir + '/it{}.checkpoint'.format(TEST_ITER)
    print 'loading from', MODEL_FILE
    save_d = torch.load(MODEL_FILE)
    for nn in m_dict:
        if not nn.endswith('_dp'):
            m_dict[nn].load_state_dict(save_d[nn])
            m_dict[nn].eval()

    batches_test = DialogueBatches([TEST_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS)
    opt = torch.optim.SGD(all_params, momentum=0.9, lr = 0, weight_decay = 1e-5)
    loss_test = mle_train(batches_test, opt, do_train = False, adv_config = adv_config)        
    logger.info('test PPL: %f log-likelihood: %f', math.exp(-loss_test), loss_test)
    
    res = {
        'loss_test': loss_test,
        'mb_lis': []
    }
    batches_test = DialogueBatches([TEST_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS)
    for src_mb, tgt_mb, tgt_len, src_w, tgt_w in batches_test:
        batch_logpdf = models.encoder_decoder_forward(src_mb, tgt_mb, tgt_len, m_dict, adv_config)
        res['mb_lis'].append((src_mb.detach().cpu(), tgt_mb.detach().cpu(), tgt_len, src_w, tgt_w, batch_logpdf.detach().cpu()))

    res_save_fn = 'figs/post_mal_advtrain/' + DATA_SET + '_IT' + str(TEST_ITER) + '_NEGLR' + str(NEG_LR) + '_POSLR' + str(POS_LR) + '_FAVOID' + str(FREQ_AVOID) + '_ILM' + str(ADV_I_LM_FLAG) + '_MFOCUS' + str(MIDDLE_FOCUS) + '_CARE' + ADV_CARE_MODE + '_testlogp.save'
    print 'saving to res_save_fn:', res_save_fn
    torch.save(res, res_save_fn)
 
