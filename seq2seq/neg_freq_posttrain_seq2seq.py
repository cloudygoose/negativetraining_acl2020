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
import time

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
D_RATIO = 0 #we do not train D for now
D_TYPE = 'CNN'
NEG_LAMBDA_TYPE = 'log'
TRAIN_SAMPLE_TYPE = 'max'
TEST_SAMPLE_TYPE = 'max'
DEBUG_INFO = True
BEAM_SIZE = 20

FREQ_AVOID_LIS = ['</s>']
FREQ_AVOID = True
FREQ_AVOID_SCAL = 0.1

if DATA_SET == 'ubuntu_dialogue_np':
    TRAIN_FN = '../data/ubuntuDialog/res_np/dialogues.200k.train.txt'
    VOCAB_FN = '../data/ubuntuDialog/res/vocab_choose.txt'
    VALID_FN = '../data/ubuntuDialog/res_np/dialogues.5k.valid.txt'
    TEST_FN = '../data/ubuntuDialog/res_np/dialogues.5k.test.txt'
    
    #ADV_TARGET_FN = '../adv_lists/ubuntu_dialogue_np/' + 'mal_words_all_500P1_2.txt' #'normal_samples_att_h500.txt' #'mal_words_all_500P1_2.txt'
   
    GE_I_LM_FN = None #'../exps//201806_adversarial_seq2seq/lm_baseline/ubuntu_lm/LSTM_LR1H600L1DR0OPTsgd/iter20.checkpoint'
    if MT == 'latent': NORMAL_WORD_AVG_LOSS = -4.243; GE_I_WORD_AVG_LOSS = -4.194185
    if MT == 'attention': NORMAL_WORD_AVG_LOSS = -4.085891; GE_I_WORD_AVG_LOSS = -4.194185
    #if MT == 'latent': NORMAL_WORD_AVG_LOSS = -3.953987; GE_I_WORD_AVG_LOSS = -4.1221
    #if MT == 'attention': NORMAL_WORD_AVG_LOSS = -3.930928; GE_I_WORD_AVG_LOSS = -4.1221
    
    DATA_CONFIG = {'filter_turn': 1, 'skip_unk': False} #2} #only the answering turns
    LOG_INTERVAL = 100
    BATCH_SIZE = 64
    BASE_BATCH_SIZE = 64
    ADV_BATCH_SIZE = 100
    ADV_CARE_MODE = 'max'
    ADV_I_LM_FLAG = False
    #one working config, uses SGD
    OPT = 'sgd'
    DROPOUT = 0
    HIDDEN_SIZE = 600
    EMBED_SIZE = 300
    LAYER_NUM = 1
    BASE_LR = 1
    POS_LR = 0.001 #ADV_LR
    NEG_LR = 0.001
    R_THRES = 0.001 
    D_LR = 0.01 #For CNN, for RNN, 0.1 is best
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
    BASE_BATCH_SIZE = 64
    BATCH_SIZE = 64
    ADV_CARE_MODE = 'max'
    ADV_I_LM_FLAG = False
    #one working config, uses SGD
    OPT = 'sgd'
    DROPOUT = 0
    HIDDEN_SIZE = 600
    EMBED_SIZE = 300
    LAYER_NUM = 1
    NEG_LR = 0.001 #ADV_LR, for test_500 (0.01, 0.001) words best 
    POS_LR = 0.001 #POS_LR larger than 0.0001 will make test PPL fluctuate for os data, even if only doing positive training
    D_LR = 0.01
    POSITIVE_RATIO = 1 #Other than 0.01 0.01, The config of neg0.001 pos0.0001 seems to work well, without much PPL loss
    BASE_LR = 0.1
    ITER_NUM = 10
    TEST_ITER = ITER_NUM
    DEBUG_INFO = True
    R_THRES = 0.001 #ratio of 0.01 will be penalized
 
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
    BASE_BATCH_SIZE = 32
    BATCH_SIZE = 32
    #one working config, uses SGD
    OPT = 'sgd'
    DROPOUT = 0.3
    HIDDEN_SIZE = 600
    EMBED_SIZE = 300
    LAYER_NUM = 1
    BASE_LR = 1
    POS_LR = 0.001 #ADV_LR
    NEG_LR = 0.001
    R_THRES = 0.01 #ratio of 0.01 will be penalized
    D_LR = 0.01 #For CNN or RNN, 0.1 is best
    POSITIVE_RATIO = 1  
    ADV_CARE_MODE = 'max'
 
    ITER_NUM = 20
    HALF_LR_ITER = 10
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

sd_add = ''
if FREQ_AVOID == True:
    sd_add = sd_add + 'FAVOID' + str(FREQ_AVOID) + str(FREQ_AVOID_SCAL)

save_dir = EXP_ROOT + '/201806_adversarial_seq2seq/neg_freq_post_seq2seq/' + DATA_SET + '/' + 'MT' + MT + '/IC' + str(ITER_NUM) + 'PRATIO' + str(POSITIVE_RATIO) + 'RTHRES' + str(R_THRES) + 'CM' + str(ADV_CARE_MODE) + sd_add + 'TRSTYPE' + str(TRAIN_SAMPLE_TYPE) + 'NLTY' + str(NEG_LAMBDA_TYPE) + 'NEGLR' + str(NEG_LR) + 'POSLR' + str(POS_LR) + 'BA' + str(BATCH_SIZE) + '_LSTM_' + 'BASELR' + str(BASE_LR) + 'BA' + str(BASE_BATCH_SIZE) + 'EM' + str(EMBED_SIZE) + 'H' + str(HIDDEN_SIZE) + 'L' + str(LAYER_NUM) + 'DR' + str(DROPOUT) + 'OPT' + str(OPT) + '_HIS' + str(HIS) + '_SLEN' + str(SRC_SEQ_LEN) + 'SUC' + SUC

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
if D_TYPE == 'RNN':
    m_d = models.LSTM_onehot_D(EMBED_SIZE, HIDDEN_SIZE, vocab_inv, dropout_rate = 0.3, layer_num = 2, final_layer_num = 2).cuda(); print 'using rnn_d';
if D_TYPE == 'RNNSIMPLE':
    m_d = models.LSTM_onehot_D(10, 10, vocab_inv, dropout_rate = 0.3, layer_num = 1, final_layer_num = 1).cuda(); print 'using rnn_simple_d';
if D_TYPE == 'CNN':
    m_d = models.CNN_KIM_TWOINPUT_D(vocab_inv, {'dropout':0.3, 'final_sigmoid':True}).cuda(); print 'using cnn_d'

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
    'm_d': m_d
}

if START_ITER > 0:
    MODEL_FILE = save_dir + '/it{}.checkpoint'.format(START_ITER - 1)
    logger.info('loading form %s.', MODEL_FILE)
    load_d = torch.load(MODEL_FILE)
    for nn1 in m_dict:
        if not nn1.endswith('_dp'):
            logger.info('loading for %s', nn1)
            m_dict[nn1].load_state_dict(load_d[nn1])

all_params, g_params, d_params = set(), set(), set()
for m in m_dict:
    mm = m_dict[m]
    all_params = all_params | set(mm.parameters())
    if m != 'm_d':
        g_params = g_params | set(mm.parameters())
    else:
        d_params = d_params | set(mm.parameters())

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
    'GIBBSENUM_E_NUM': GIBBSENUM_E_NUM,
    'GIBBSENUM_RANDOM_TIME': GIBBSENUM_RANDOM_TIME,
    'GIBBSENUM_MAX_ITER_CO': GIBBSENUM_MAX_ITER_CO,
    'GIBBSENUM_I_LM_LAMBDA': GIBBSENUM_I_LM_LAMBDA,
    'BEAM_SIZE': BEAM_SIZE,
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

def sen_left2right(mb, target_len):
    bz = mb.size(0)
    res = torch.LongTensor(bz, target_len).cuda()
    res[:, :] = vocab_inv['<pad>']
    for i in range(bz):
        first_w_idx = 0
        if vocab[mb[i, first_w_idx]] == '<s>': first_w_idx += 1
        last_w_idx = 0
        while last_w_idx < mb.size(1) - 1 and vocab[mb[i, last_w_idx + 1]] != '</s>': last_w_idx += 1
        if last_w_idx > target_len - 2: last_w_idx = target_len - 2
        res[i, target_len - 1 - (last_w_idx - first_w_idx + 1) : target_len - 1] = mb[i, first_w_idx : (last_w_idx + 1)]
        res[i, target_len - 1] = vocab_inv['</s>']
        if target_len - 1 - (last_w_idx - first_w_idx + 1) - 1 >= 0:
            res[i, target_len - 1 - (last_w_idx - first_w_idx + 1) - 1] = vocab_inv['<s>']
    #print 'mb:', mb[:3], 'res:', res[:3]
    return res

bce_criterion = nn.BCELoss()
def d_train_mb(mb, opt_d, do_train = True):
    src_mb, tgt_mb, tgt_len, src_w, tgt_w = mb  
    bz = src_mb.size(0)
    res_sample = models.get_samples([mb], TRAIN_SAMPLE_TYPE, m_dict, adv_config) 
    sample_mb = res_sample['raw_sample_lis'][0]
    if do_train == True:
        m_d.train()
    else:
        m_d.eval()

    input_idx_tgt = torch.cat([sen_left2right(tgt_mb, TGT_SEQ_LEN), sen_left2right(sample_mb, TGT_SEQ_LEN)], dim = 0)
    #debug
    #t_zero = torch.LongTensor(bz, TGT_SEQ_LEN); t_zero[:] = 0
    #t_one = torch.LongTensor(bz, TGT_SEQ_LEN); t_one[:] = 1
    #input_idx_tgt = torch.cat([t_zero, t_one], dim = 0)
    
    input_idx_src = torch.cat([src_mb, src_mb], dim = 0)
    y_label = torch.cat([torch.ones(bz), torch.zeros(bz)], dim = 0).cuda()
    net_out = m_d(idx2onehot(input_idx_src, len(vocab)).detach(), idx2onehot(input_idx_tgt, len(vocab)).detach())
    loss = bce_criterion(net_out.view(-1), y_label)
    #loss = ((net_out - y_label) * (net_out - y_label)).mean()
    acc_num = (torch.sum(net_out[:bz] >= 0.5) + torch.sum(net_out[bz:] <= 0.5)).item()
    if do_train == True:
        opt_d.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_d.parameters(), 5)
        #for n, m in rnn_d.named_parameters():
        #    if hasattr(m, 'grad')==True:
        #        print n, torch.norm(m.grad.view(-1)).item()
        opt_d.step()
    #print 'mean_out', net_out[:BATCH_SIZE].mean(), net_out[BATCH_SIZE:].mean()
    res = {
        'net_out': net_out,
        'd_loss': loss.item(),
        'd_acc_num': acc_num,
        'd_acc_ratio': float(acc_num) / (bz * 2.0),
    }
    return res

def form_tgtmb(s_mb):
    bz = s_mb.size(0)
    res_mb = torch.LongTensor(bz, TGT_SEQ_LEN + 1).cuda()
    res_mb[:, :] = vocab_inv['<pad>']
    res_tgt_len = [0 for kk in range(bz)]
    for i in range(bz):
        res_mb[i][0] = vocab_inv['<s>']
        for j in range(s_mb.size(1)):
            #assert(s_mb[i][j] != vocab_inv['<s>']) #in bad situations, <s> will appear
            res_mb[i][j + 1] = s_mb[i][j]
            res_tgt_len[i] += 1
            if s_mb[i][j] == vocab_inv['</s>']:
                break
    return res_mb, res_tgt_len

def make_ratio_dict(long_his):
    dd = {}
    all_num = 0
    for s_mb in long_his:
        for s_w in s_mb:
            s = ' '.join(s_w)
            if s not in dd: dd[s] = 0
            dd[s] += 1
            all_num += 1
    for s in dd:
        dd[s] = float(dd[s]) * 1.0 / all_num
    return dd 

def neg_d_train(aux_batches, train_batches, opts, do_train = True, do_log = True, do_adv = False, adv_config = None):
    #adv_train put adversarial training first
    #assert(do_train == True)
    opt_pos, opt_neg, opt_d = opts
    all_loss, all_num, adv_all_loss, adv_all_num, nodecay_sennum, success0_co, success1_co = 0, 0, 0, 0, 0, 0, 0
    attack_results = []
    loss_sen = []
    all_target_set = {}
    inf_pos_batches, inf_d_batches = aux_batches
    
    d_b_co, pos_b_co, neg_b_co = 0, 0, 0
    sample_mb_longhis = []
    stat_dic = MyStatDic() 
    for train_mb in train_batches:
        neg_b_co += 1
        src_mb, tgt_mb, tgt_len, src_w, tgt_w = train_mb  
        bz = src_mb.size(0)
        res_sample = models.get_samples([train_mb], TRAIN_SAMPLE_TYPE, m_dict, adv_config) 
        sample_mb, samples_w = res_sample['raw_sample_lis'][0], res_sample['sample_lis']

        neg_lambda = [0 for kk in range(len(samples_w))]
        sample_mb_longhis.append(samples_w)
        if len(sample_mb_longhis) > 200: 
            sample_mb_longhis = sample_mb_longhis[1:]
            d_r = make_ratio_dict(sample_mb_longhis) 
            for i in range(BATCH_SIZE):
                #print samples_w[i]
                if d_r[' '.join(samples_w[i])] > R_THRES:
                    neg_lambda[i] = 1
                    #print 'found frequent response:', ' '.join(samples_w[i]), 'ratio:', d_r[' '.join(samples_w[i])]
        neg_lambda = torch.FloatTensor(neg_lambda).cuda()
                     
        sample_tgt_mb, sample_tgt_len = form_tgtmb(sample_mb)
        for m in m_dict.values(): m.zero_grad(); m.train();
        aux = {}; adv_batch_logpdf = models.encoder_decoder_forward(src_mb, sample_tgt_mb, sample_tgt_len, m_dict, adv_config, aux_return = aux); w_logit_rnn = aux['w_logit_rnn'];
        
        final_mask = torch.FloatTensor(adv_batch_logpdf.size()).cuda()
        final_mask[:] = 1
        if FREQ_AVOID == True:
            for i in range(len(sample_tgt_len)):
                final_mask[i][sample_tgt_len[i] - 1] = FREQ_AVOID_SCAL
            #for i in range(5):
            #    print sample_tgt_mb[i]
            #    print adv_batch_logpdf[i]
            #    print final_mask[i]
            #sys.exit(1)

        neg_logpdf = torch.sum(adv_batch_logpdf * final_mask, dim = 1) * neg_lambda
        if do_train == True:
            (torch.sum(neg_logpdf) / sum(sample_tgt_len)).backward()
            for m in m_dict.values(): torch.nn.utils.clip_grad_norm_(m.parameters(), 5)
            opt_neg.step()

        stat_dic.append_dict({'neg_lambda':neg_lambda.mean().item()})
        stat_dic.append_dict({'neg_loss':(torch.sum(adv_batch_logpdf).item() / sum(sample_tgt_len))}) 
        if neg_b_co % 2000 == 0 and DEBUG_INFO == True:
            print 'peek mb sample!'
            for i in range(sample_tgt_mb.size(0) / 5):
                logger.info('neg_lambda: %f %s', neg_lambda[i].item(), ' '.join([vocab[sample_tgt_mb[i][j]] for j in range(sample_tgt_len[i] + 1)]))
            #wait = raw_input("PRESS ENTER")

        src_mb, tgt_mb, tgt_len, src_w, tgt_w = train_mb 
        for kk in range(D_RATIO):
            res_d = d_train_mb(inf_d_batches.next(), opt_d, do_train = do_train)     
            d_b_co += 1
            stat_dic.append_dict(res_d, keys = ['d_loss', 'd_acc_ratio'])

        for kk in range(POSITIVE_RATIO):
            src_mb, tgt_mb, tgt_len, src_w, tgt_w = inf_pos_batches.next()
            pos_b_co += 1
            bz = src_mb.size(0)
            
            for m in m_dict.values():
                m.train()
                m.zero_grad()
        
            batch_logpdf = models.encoder_decoder_forward(src_mb, tgt_mb, tgt_len, m_dict, adv_config)
            
            w_loss_rnn = torch.sum(batch_logpdf)
            loss_sen.extend(torch.sum(batch_logpdf, dim = 1).detach().cpu().numpy().tolist())
                
            #all_loss = all_loss + w_loss_rnn.data.item()
            stat_dic.append_dict({'pos_loss':(- w_loss_rnn.item() / sum(tgt_len))})
            
            if do_train == True:
                (- w_loss_rnn / sum(tgt_len)).backward()            
                for m in m_dict.values():
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 5)
                opt_pos.step()
        
        if neg_b_co % 200 == 0:
            stat_dic.log_mean(last_num = 200, log_pre = 'it{} neg_b_co: {}'.format(adv_config['iter_now'], neg_b_co)) 
           
    return stat_dic

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

def test_entropy(it, print_sample = True):
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

	save_fn = save_dir + '/it{}_testentropy_samples.save'.format(it)
	logger.info('saving testentropy_res to %s', save_fn)
	torch.save(res, save_fn)
	
    dd = {}
    for s in res['sample_lis']:
        if not ' '.join(s) in dd:
            dd[' '.join(s)] = 0
        dd[' '.join(s)] += 1
    r_count, max_r = 0, 0
    for s in dd:
        r = float(dd[s] * 1.0) / len(res['sample_lis']) 
        if r > max_r: max_r = r
        if r > R_THRES:
            r_count += 1
            logger.info('SAMPLE >R_THRES : %s ratio: %f', s, r)

    logger.info('it%d R_THRES_SAMPLE_COUNT: %d R_MAX_SAMPLE: %f', it, r_count, max_r)

    if print_sample == True:
        logger.info('printing first 200 samples with ratio > 0.0001')
        co = 0
        for s in dd:
            r = float(dd[s] * 1.0) / len(res['sample_lis']) 
            if r > 0.0001:
                co += 1
                logger.info('co: %d sample: %s time: %d ratio: %f', co, s, dd[s], float(dd[s] * 1.0) / len(res['sample_lis']))
                if co > 200: break

    return res

if COMMAND.startswith('test'): #test
    if TEST_ITER == -1:
        MODEL_FILE = base_save_dir + '/iter{}.checkpoint'.format(BASE_ITER_NUM)
    else:
        MODEL_FILE = save_dir + '/it{}.checkpoint'.format(TEST_ITER)
 
    logger.info('loading from %s.', MODEL_FILE)
    save_d = torch.load(MODEL_FILE)
    for nn in m_dict:
        if not nn.endswith('_dp') and nn != 'm_d':
            m_dict[nn].load_state_dict(save_d[nn])
            m_dict[nn].eval()
    
    """
    logger.info('doing ppl test TEST_FN: %s', TEST_FN)
    batches_test = DialogueBatches([TEST_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS)
    opt = torch.optim.SGD(all_params, momentum=0.9, lr = 0, weight_decay = 1e-5)
    loss_test = mle_train(batches_test, opt, do_train = False, adv_config = adv_config)        
    logger.info('test PPL: %f log-likelihood: %f', math.exp(-loss_test), loss_test)
    """
    #batches_test = DialogueBatches([TEST_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS) 
    #ty = 'sample_min' #'sample_min' #'sample' #'max'
    #out_fn = save_dir + '/samples_test_ty' + ty + '.txt'
    #logger.info('sampling out_fn : %s', out_fn)
    #sample_compare_bleu(batches_test, ty = ty, outfn = out_fn, extend_num = 1)

       
    """
    batches_test = DialogueBatches([TEST_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS) 
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
    if COMMAND == 'testout':
        batches_test = DialogueBatches([TEST_FN], 20, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS) 
        logger.info('TEST_SAMPLE_TYPE: %s', TEST_SAMPLE_TYPE)
        res = models.get_samples(batches_test, TEST_SAMPLE_TYPE, m_dict, adv_config)
        for i in range(50):
            print (i, 'src:', ' '.join(res['src_lis'][i]))
            print (' '.join(res['sample_lis'][i]))
        fn = 'saves/negfreq_testouts/' + DATA_SET + '_TESTITER' + str(TEST_ITER) + '_IC' + str(ITER_NUM) + 'PRATIO' + str(POSITIVE_RATIO) + 'RTHRES' + str(R_THRES) + 'CM' + str(ADV_CARE_MODE) + sd_add + 'TRSTYPE' + str(TRAIN_SAMPLE_TYPE) + 'NLTY' + str(NEG_LAMBDA_TYPE) + 'NEGLR' + str(NEG_LR) + 'POSLR' + str(POS_LR) + '.testout'
        logger.info('saving to %s', fn)
        torch.save(res, fn)
        sys.exit(1)
         
    if COMMAND == 'test_entropy':
        batches_test = DialogueBatches([TEST_FN], 20, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS) 
        logger.info('TEST_SAMPLE_TYPE: %s', TEST_SAMPLE_TYPE)
        res = models.get_samples(batches_test, TEST_SAMPLE_TYPE, m_dict, adv_config)
        ref_lis, sample_lis = res['ref_lis'], res['sample_lis']
        for ss in ('ref_lis', 'sample_lis'):
            for kk in (2, 3):
                entro = text_eval.text_entropy(res[ss], kk)
                logger.info('text_entropy(TEST_FN) for %s with k %d: %f', ss, kk, entro)
        
            logger.info('printing all distinct samples with ratio > 0.01')
            dd = {}
            for s in res[ss]:
                if not ' '.join(s) in dd:
                    dd[' '.join(s)] = 0
                dd[' '.join(s)] += 1
            max_r = 0; max_s = '';
            for s in dd:
                r = float(dd[s] * 1.0) / len(res[ss])
                if r > max_r: max_r = r; max_s = s;
                if r > 0.01:
                    logger.info('sample: %s time: %d ratio: %f', s, dd[s], r)
            logger.info('%s max_r: %f max_s: %s', ss, max_r, max_s)

    if COMMAND == 'testout_beam_search':
        BEAM_OUT_FILE = './generic_lists/beam_out_files/' + DATA_SET + '/' + MT + '_beamsize' + str(BEAM_SIZE) + '.txt'
        logger.info('doing beam_search on VALID_FN %s !! BEAM_SIZE: %d', VALID_FN, BEAM_SIZE)
        batches_test = DialogueBatches([VALID_FN], 20, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS) 
        b_count = 0
        logger.info('Outputing beams to file %s', BEAM_OUT_FILE)
        outf = open(BEAM_OUT_FILE, 'w')
        for src_mb, tgt_mb, tgt_len, src_w, tgt_w in batches_test:
            b_count = b_count + 1
            if b_count % 20 == 0:
                logger.info('b_count: %d', b_count)
            for i in range(len(tgt_len)):
                beam = models.beam_search(src_mb[i].unsqueeze(0), TGT_SEQ_LEN, BEAM_SIZE, m_dict, adv_config)
                if b_count == 1 and i < 5:
                    print 'src:', ' '.join(src_w[i])
                    print 'ref:', ' '.join(tgt_w[i])
                    for j in range(BEAM_SIZE):
                        print '\t', 'beam ll:', beam[j]['ll'], ' '.join(beam[j]['w_lis'])
                for j in range(BEAM_SIZE):
                    outf.write(str(j) + '\t' + ' '.join(beam[j]['w_lis']) + '\n')
        outf.close() 

if COMMAND == 'post_advtrain':
    if START_ITER == 0:
        MODEL_FILE = base_save_dir + '/iter{}.checkpoint'.format(BASE_ITER_NUM)
        logger.info('START_ITER is 0, loading form base %s.', MODEL_FILE)
        save_d = torch.load(MODEL_FILE)
        for nn in m_dict:
            if not nn.endswith('_dp') and nn != 'm_d':
                m_dict[nn].load_state_dict(save_d[nn])
                m_dict[nn].eval()
    
    logger.info('ADV_CARE_MODE: %s', ADV_CARE_MODE)
    inf_d_batches = inf_data_gen([TRAIN_FN], BATCH_SIZE, ty = 'DIALOGUE')
    inf_pos_batches = inf_data_gen([TRAIN_FN], BATCH_SIZE, ty = 'DIALOGUE')
    
    if DO_INITIAL_TEST == True: 
       	test_entropy(-1)

    if START_ITER == 0 and D_RATIO > 0:
        logger.info('START_ITER == 0, doing pre-training for m_d!') 
        stat_dic = MyStatDic()
        opt_d = torch.optim.SGD(d_params, momentum = 0.9, lr = D_LR, weight_decay = 1e-5)
        for kk in range(2000 + 1):
            res_d = d_train_mb(inf_d_batches.next(), opt_d, do_train = True)     
            stat_dic.append_dict(res_d, keys = ['d_loss', 'd_acc_ratio'])
            if kk % 200 == 0 and kk != 0:
                stat_dic.log_mean(keys = ['d_loss', 'd_acc_ratio'], last_num=200, log_pre='b_co{}'.format(kk))
         
    pos_lr, neg_lr, d_lr = POS_LR, NEG_LR, D_LR
    for it in range(START_ITER, ITER_NUM + 1):          
        #if it > 0 and it % 5 == 0: #seems to be a bad idea for open_subtitle
        #    pos_lr, neg_lr = pos_lr * 0.5, neg_lr * 0.5  
        logger.info('starting iter %s POS_LR: %f NEG_LR: %f D_LR:%f POSITIVE_RATIO: %d D_RATIO: %d R_THRES: %f', it, pos_lr, neg_lr, d_lr, POSITIVE_RATIO, D_RATIO, R_THRES)
        logger.info('command is %s', ' '.join(sys.argv))
        print 'log_fn:', log_fn
        #global opt
        opt_pos = torch.optim.SGD(g_params, momentum = 0.9, lr = pos_lr, weight_decay = 1e-5)
        opt_neg = torch.optim.SGD(g_params, momentum = 0.9, lr = neg_lr, weight_decay = 1e-5)
        opt_d = torch.optim.SGD(d_params, momentum = 0.9, lr = d_lr, weight_decay = 1e-5)
        batches_train = DialogueBatches([TRAIN_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS)       
        adv_config['iter_now'] = it
        stat_dic = neg_d_train((inf_pos_batches, inf_d_batches), batches_train, (opt_pos, opt_neg, opt_d), do_train = True, do_adv = True, adv_config = adv_config)       
        stat_dic.log_mean(last_num = 0, log_pre = 'STAT_FINISH_TRAIN_ITER: {}'.format(it)) 
 
        batches_test = DialogueBatches([TEST_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS)
        loss_test = mle_train(batches_test, opt_pos, do_train = False, adv_config = adv_config)        
        logger.info('it: %d, base_test PPL: %f', it, math.exp(-loss_test))
        if math.exp(-loss_test) > 400:
            logger.info('test_PPL > 400, exiting...')
            sys.exit(1)
        #logger.info('it: %d, hit_num found in train: %d hit_rate: %f per target_num: %d', it, res['nodecay_sennum'], float(res['nodecay_sennum']) / res['adv_sen_count'], res['adv_sen_count'])
        
        res_entro = test_entropy(it)
        if it % 10 == 0 and it != 0:
            try:
                save_d = {}
                for m in m_dict:
                    save_d[m] = m_dict[m].state_dict()
                model_fn = save_dir + '/it{}.checkpoint'.format(it)
                logger.info('saving to %s', model_fn)
                torch.save(save_d, model_fn)
                save_fn = save_dir + '/it{}res_entro.save'.format(it)
                logger.info('saving res_entro to %s', save_fn)
                torch.save(res_entro, save_fn)
            except IOError:
                print '!!!!!!!!!!!!!!!!!!!IOError!!!!!!!!!!!!!!!!!!!!!!!!!!'
                print '!!!!!!!!!!!!!!!!!!!IOError!!!!!!!!!!!!!!!!!!!!!!!!!!'
            
        comm = 'cp ' + log_fn + ' ' + log_fn + '_it' + str(it)
        logger.info('saving log file: %s', comm)
        os.system(comm)
        logger.info('grepping log file %s', log_fn)
        comm = 'grep ' + "STAT_FINISH_TRAIN_ITER" + ' ' + log_fn
        os.system(comm)
        comm = 'grep ' + "'test PPL:'" + ' ' + log_fn
        os.system(comm)
        logger.info('grepping log file %s for text_entropy', log_fn)
        comm = 'grep ' + "'text_entropy(TEST_FN)'" + ' ' + log_fn
        os.system(comm)
        logger.info('grepping log file for R_THRES_SAMPLE_COUNT:')
        comm = 'grep ' + "'R_THRES_SAMPLE_COUNT:'" + ' ' + log_fn
        os.system(comm)
