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

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import text_eval

DATA_SET = 'swda_dialogue' #'ptb_chars' #'swda_dialogue' #'ubuntu_dialogue' #'dailydialogue'
COMMAND = 'train'

EXP_ROOT = '../exps/'

import socket
print('hostname:', socket.gethostname())

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print 'CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES']

MT = 'attention' #'latent' or 'attention'
if len(sys.argv) > 1:
    print 'execing sys.argv[1] for setting:', sys.argv[1]
    exec(sys.argv[1])

if DATA_SET in 'ubuntu_dialogue_np':
    DD = 'res'
    if DATA_SET == 'ubuntu_dialogue_np': DD = 'res_np'
    print 'ubuntu DD:', DD
    TRAIN_FN = '../data/ubuntuDialog/'+DD+'/dialogues.200k.train.txt'
    VOCAB_FN = '../data/ubuntuDialog/res/vocab_choose.txt'
    VALID_FN = '../data/ubuntuDialog/'+DD+'/dialogues.5k.valid.txt'
    TEST_FN = '../data/ubuntuDialog/'+DD+'/dialogues.5k.test.txt'
    
    if DATA_SET == 'ubuntu_dialogue_np':
        if MT == 'latent': NORMAL_WORD_AVG_LOSS = -4.243; GE_I_WORD_AVG_LOSS = -4.194185
        if MT == 'attention': NORMAL_WORD_AVG_LOSS = -4.085891; GE_I_WORD_AVG_LOSS = -4.194185
    if DATA_SET == 'ubuntu_dialogue':
        if MT == 'latent': NORMAL_WORD_AVG_LOSS = -3.953986; GE_I_WORD_AVG_LOSS = -4.1221;
        if MT == 'attention': NORMAL_WORD_AVG_LOSS = -3.930928; GE_I_WORD_AVG_LOSS = -4.1221;

    DATA_CONFIG = {'filter_turn': 1, 'skip_unk': False} #2} #only the answering turns
    LOG_INTERVAL = 100
    BATCH_SIZE = 64
    #one working config, uses SGD
    OPT = 'sgd'
    DROPOUT = 0
    HIDDEN_SIZE = 600
    EMBED_SIZE = 300
    LAYER_NUM = 1
    LR = 1
    ADV_BZ = 100
    ITER_NUM = 20
    HALF_LR_ITER = 10

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
    #one working config, uses SGD
    OPT = 'sgd'
    DROPOUT = 0
    HIDDEN_SIZE = 600
    EMBED_SIZE = 300
    LAYER_NUM = 1
    LR = 0.1
    ADV_BZ = 100
    ITER_NUM = 20
    HALF_LR_ITER = 10

    HIS = 1 #2 #2
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
    LR = 1

    ITER_NUM = 20
    HALF_LR_ITER = 10
    ADV_BZ = 100
 
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
#ADV_TARGET_FN = 'mal_words_all_500P1_2.txt' #'normal_samples_att_h500.txt' #'mal_words_all_500P1_2.txt'
TEST_ITER = ITER_NUM
LASSO_LAMBDA = 1
DRAW_FN = 'figs/tmp.jpg'
FORCE_ONEHOT = True #force to be onehot when testing

ENUM_START_LENGTH = 1
ENUM_END_LENGTH = 100
ENUM_START_RATE = 0
ENUM_END_RATE = 1

GIBBSENUM_DECAYLOSS_WEIGHT = 0
GIBBSENUM_E_NUM = 100 #-1 means not activated
GIBBSENUM_RANDOM_TIME = 5

GE_I_LAMBDA = 1
GE_I_DO = False
GE_I_SAMPLE_LOSSDECAY = True
GE_MAX_ITER_CO = 5

PARA_FILE = '../para_adv_lists/res_500/ori_pair.txt'
PARA_OUT_FILE = '../para_adv_lists/res_500/train_para_all.txt'

BEAM_SIZE = 20
TEST_SAMPLE_TYPE = 'max'

if len(sys.argv) > 1:
    print 're-execing sys.argv[1] for setting:', sys.argv[1]
    exec(sys.argv[1])

torch.manual_seed(1234) #just to be different from the random generator

if MT == 'latent':
    save_dir = EXP_ROOT + '/201806_adversarial_seq2seq/mle_baseline/' + DATA_SET + '/LSTM_' + 'LR' + str(LR) + 'BA' + str(BATCH_SIZE) + 'EM' + str(EMBED_SIZE) + 'H' + str(HIDDEN_SIZE) + 'L' + str(LAYER_NUM) + 'DR' + str(DROPOUT) + 'OPT' + str(OPT) + '_HIS' + str(HIS) + '_SLEN' + str(SRC_SEQ_LEN) + '_ITERNUM' + str(ITER_NUM) + SUC
if MT == 'attention':
    save_dir = EXP_ROOT + '/201806_adversarial_seq2seq/mle_att_baseline/' + DATA_SET + '/LSTM_' + 'LR' + str(LR) + 'BA' + str(BATCH_SIZE) + 'EM' + str(EMBED_SIZE) + 'H' + str(HIDDEN_SIZE) + 'L' + str(LAYER_NUM) + 'DR' + str(DROPOUT) + 'OPT' + str(OPT) + '_HIS' + str(HIS) + '_SLEN' + str(SRC_SEQ_LEN) + '_ITERNUM' + str(ITER_NUM) + SUC

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
    MODEL_FILE = save_dir + '/iter{}.checkpoint'.format(START_ITER - 1)
    logger.info('loading form %s.', MODEL_FILE)
    load_d = torch.load(MODEL_FILE)
    for nn in m_dict:
        if not nn.endswith('_dp'):
            logger.info('loading for %s', nn)
            m_dict[nn].load_state_dict(load_d[nn])

all_params = set()
for m in m_dict.values():
    all_params = all_params | set(m.parameters())

def mask_gen(lengths):
    max_len = lengths[0]
    size = len(lengths)
    mask = torch.ByteTensor(size, max_len).zero_()
    for i in range(size):
        mask[i][:lengths[i]].fill_(1)
    return mask

adv_config = {
    'MT': MT,
    'HIDDEN_SIZE': HIDDEN_SIZE, 
    'vocab': vocab,
    'vocab_inv': vocab_inv,
    'SRC_SEQ_LEN': SRC_SEQ_LEN,
    'TGT_SEQ_LEN': TGT_SEQ_LEN,
    'BEAM_SIZE': BEAM_SIZE,
    'LAYER_NUM': LAYER_NUM,
}

def decoder_forward(e_output, tgt_inputv, tgt_len):
    if MT == 'latent':
        latent = e_output[:, -1, :].squeeze(1)
        latent = latent.unsqueeze(1).repeat(1, tgt_inputv.size(1), 1)
        w_logit_rnn = m_decode_w_rnn_dp(latent, tgt_inputv, tgt_len) #change from decode to forward for data parallel
        return w_logit_rnn, None
    if MT == 'attention':
        w_logit_rnn, attn_weights, _ = m_decode_w_rnn_dp(e_output, tgt_inputv)
        return w_logit_rnn, attn_weights

def mle_train(batches, opt, do_train = True, do_log = True):
    for m in m_dict:
        if do_train == True:
            m_dict[m].train()
        else:
            m_dict[m].eval()
    
    all_loss = 0 
    all_num = 0
    b_count = 0
    loss_sen = []
    for src_mb, tgt_mb, tgt_len, src_w, tgt_w in batches:
        #print src_w[0], src_w[1]; sys.exit(1)
        loss = 0
        b_count = b_count + 1
        bz = src_mb.size(0)
        all_num = all_num + sum(tgt_len)
        
        src_inputv = Variable(src_mb).cuda() 
        tgt_inputv = Variable(tgt_mb[:, :-1]).cuda() 
        tgt_targetv = Variable(tgt_mb[:, 1:]).cuda()
        tgt_mask = Variable(torch.FloatTensor([([1] * l + [0] * (tgt_inputv.size(1) - l)) for l in tgt_len]).cuda(), requires_grad = False)
 
        #mask = Variable(mask_gen(b_len)).cuda()

        #size(batch, length)
        output, _ = m_encode_w_rnn_dp(m_embed_dp(idx2onehot(src_inputv, len(vocab))).permute(1, 0, 2), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM)) #for parallel!
        output = output.permute(1, 0, 2) #for parallel!
        w_logit_rnn, attn_weights = decoder_forward(output, tgt_inputv, tgt_len)
        flat_output = w_logit_rnn.view(-1, len(vocab))
        flat_target = tgt_targetv.contiguous().view(-1)
        flat_logpdf = lib_pdf.logsoftmax_idxselect(flat_output, flat_target)
        batch_logpdf = flat_logpdf.view(bz, -1) * tgt_mask
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
        
        if do_log == True and b_count % LOG_INTERVAL == 0:
            logger.info('avg loss at b: %d , %f', b_count, all_loss * 1.0 / all_num)

    logger.info('all_num: %d', all_num)
    logger.info('sen_avg_loss: %f', np.mean(loss_sen))
    return float(all_loss * 1.0 / all_num)

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

if COMMAND == 'train': #train
    res_lis = []
    if START_LR != None:
        LR = START_LR
    for iter in range(START_ITER, ITER_NUM + 1):            
        logger.info('command is: %s', ' '.join(sys.argv)) 
        if OPT == 'sgd' and iter >= HALF_LR_ITER:
            LR *= 0.6
        logger.info('starting iter %s LR: %f', iter, LR)
        #global opt
        opt =torch.optim.SGD(all_params, momentum=0.9, lr = LR, weight_decay = 1e-5)
        logger.info('train_fn is %s', TRAIN_FN)
        batches_train = DialogueBatches([TRAIN_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS)
        loss_train = mle_train(batches_train, opt, do_train = True)        
        batches_test = DialogueBatches([TEST_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS)
        loss_test = mle_train(batches_test, opt, do_train = False)        
        logger.info('iter: %d, train PPL: %f, test PPL: %f', iter, math.exp(-loss_train), math.exp(-loss_test))
        #batches_test = DialogueBatches([TEST_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS)
        #res_now = sample_compare_bleu(batches_test, save_dir + '/samples_it' + str(iter) + '.txt')
        #res_now['it'] = iter
        #r2 = sample_compare_bleu(batches_test, save_dir + '/samples_best10_it' + str(iter) + '.txt', ty = 'sampling', extend_num = 10)
        #res_now['bleu2_best10'] = r2['bleu2']
        #res_now['bleu3_best10'] = r2['bleu3']
        #res_lis.append(res_now)
        #logger.info('res history so far: %s', str(res_lis))
        if iter % 1 == 0:
            if iter % 5 == 0 or iter % 2 == 0:
                save_d = {}
                for m in m_dict:
                    save_d[m] = m_dict[m].state_dict()
                model_fn = save_dir + '/iter{}.checkpoint'.format(iter)
                logger.info('saving to %s', model_fn)
                torch.save(save_d, model_fn)
            comm = 'cp ' + log_fn + ' ' + log_fn + '_it' + str(iter)
            logger.info('saving log file: %s', comm)
            os.system(comm)

if 'test' in COMMAND: #test
    MODEL_FILE = save_dir + '/iter{}.checkpoint'.format(ITER_NUM)
    logger.info('loading form %s.', MODEL_FILE)
    save_d = torch.load(MODEL_FILE)
    for nn in m_dict:
        if not nn.endswith('_dp'):
            m_dict[nn].load_state_dict(save_d[nn])
            m_dict[nn].eval()
    
    logger.info('doing ppl test TEST_FN: %s', TEST_FN)
    batches_test = DialogueBatches([TEST_FN], BATCH_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS)
    opt = torch.optim.SGD(all_params, momentum=0.9, lr = 0, weight_decay = 1e-5)
    loss_test = mle_train(batches_test, opt, do_train = False)        
    logger.info('test PPL: %f log-likelihood: %f', math.exp(-loss_test), loss_test)
    
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

    if COMMAND == 'test_entropy':
        batches_test = DialogueBatches([TEST_FN], 20, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS) 
        logger.info('TEST_SAMPLE_TYPE: %s', TEST_SAMPLE_TYPE)
        res = models.get_samples(batches_test, TEST_SAMPLE_TYPE, m_dict, adv_config)
        #for i in range(50):
        #    print (' '.join(res['sample_lis'][i]))
        ref_lis, sample_lis = res['ref_lis'], res['sample_lis']
        for ss in ('ref_lis', 'sample_lis'):
            for kk in (2, 3):
                entro = text_eval.text_entropy(res[ss], kk)
                logger.info('text_entropy(TEST_FN) for %s with k %d: %f', ss, kk, entro)
        logger.info('printing all distinct samples with ratio > 0.01')
        dd = {}
        for s in res['sample_lis']:
            if not ' '.join(s) in dd:
                dd[' '.join(s)] = 0
            dd[' '.join(s)] += 1
        max_r = 0
        for s in dd:
            r = float(dd[s] * 1.0) / len(res['sample_lis'])
            if r > max_r: max_r = r
            if r > 0.01:
                logger.info('sample: %s time: %d ratio: %f', s, dd[s], r)
        logger.info('max_r: %f', max_r)

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

if COMMAND == 'para_adv':
    MODEL_FILE = save_dir + '/iter{}.checkpoint'.format(ITER_NUM)
    logger.info('loading form %s.', MODEL_FILE)
    save_d = torch.load(MODEL_FILE)
    for nn in m_dict:
        if not nn.endswith('_dp'):
            m_dict[nn].load_state_dict(save_d[nn])
            m_dict[nn].eval()
    
    logger.info('PARA_FILE is %s', PARA_FILE)    
    batches_test = DialogueBatches([PARA_FILE], 20, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS) 
    dd = {} #checking for duplicates
    for src_mb, tgt_mb, tgt_len, src_w, tgt_w in batches_test:
        for i in range(len(tgt_len)):
            for ss in [' '.join(src_w[i][:-1]), ' '.join(tgt_w[i][1:])]:
                if ss.endswith('</s>'): ss = ss[:-4]
                ss = ss.strip()
                assert(ss not in dd)
                dd[ss] = True

    f_out = open(PARA_OUT_FILE, 'w')
    logger.info('writing to %s', PARA_OUT_FILE)
    b_count = 0
    batches_test = DialogueBatches([PARA_FILE], 20, SRC_SEQ_LEN, TGT_SEQ_LEN, vocab_inv, DATA_CONFIG, his_len = HIS) 
    for src_mb, tgt_mb, tgt_len, src_w, tgt_w in batches_test:
        b_count = b_count + 1
        print(b_count)
        for i in range(len(tgt_len)): #src_mb is the train seq, which is to be paraed
            beam = models.beam_search(src_mb[i].unsqueeze(0), TGT_SEQ_LEN, 30, m_dict, adv_config)
            id_now = 0
            p_lis = []
            for kk in range(10):
                while True:
                    p_now = ' '.join(beam[id_now]['w_lis'])
                    if p_now.endswith('</s>'): p_now = p_now[:-4]
                    p_now = p_now.strip()
                    id_now = id_now + 1
                    if p_now not in dd: break
                dd[p_now] = True
                p_lis.append(p_now)
                #print 'src:', ' '.join(src_w[i])
                #print 'ref:', ' '.join(tgt_w[i])
                #for j in range(10):
                #print '\t', 'beam ll:', beam[j]['ll'], ' '.join(beam[j]['w_lis'])
            f_out.write('\t'.join(p_lis) + '\n')
    f_out.close()

def adversarial_latent_optimize(lis_tgt_w):
    bz = len(lis_tgt_w)
    tgt_len = [len(l) - 1 for l in lis_tgt_w]
    max_len = max(tgt_len)
    #latent_v = Variable(torch.randn(bz, max_len, HIDDEN_SIZE).cuda(), requires_grad = True)
    latent_v_ori = Variable(torch.randn(bz, 1, HIDDEN_SIZE).cuda(), requires_grad = True)
    latent_v = latent_v_ori.repeat(1, max_len, 1) 
    latent_opt = torch.optim.SGD([latent_v_ori], momentum = 0.9, lr = 1, weight_decay = 1e-5) #the larger the better, it seems
    lis_tgt_w = [l + ['<pad>'] * (max_len + 1 - len(l)) for l in lis_tgt_w]
    lis_tgt_idx = [[vocab_inv[w] for w in l] for l in lis_tgt_w]
    tgt_mb = Variable(torch.LongTensor(lis_tgt_idx).cuda())
    tgt_inputv = tgt_mb[:, :-1]
    tgt_targetv = tgt_mb[:, 1:]
    tgt_mask_ori = Variable(torch.FloatTensor([([1] * l + [0] * (tgt_inputv.size(1) - l)) for l in tgt_len]).cuda(), requires_grad = False)
 
    for it in range(4000 + 1):
        #print latent_v.size(), tgt_inputv.size(), tgt_len
        latent_v = latent_v_ori.repeat(1, max_len, 1)
        w_logit_rnn = m_decode_w_rnn.forward(F.tanh(latent_v), tgt_inputv)
        flat_output = w_logit_rnn.view(-1, len(vocab))
        flat_target = tgt_targetv.contiguous().view(-1)
        flat_logpdf = lib_pdf.logsoftmax_idxselect(flat_output, flat_target)
        tgt_mask = Variable(torch.FloatTensor([([1] * l + [0] * (tgt_inputv.size(1) - l)) for l in tgt_len]).cuda(), requires_grad = False)
        w_logit_pred = torch.max(w_logit_rnn, dim = 2)[1]
        for i in range(bz):
            for j in range(tgt_len[i]):
                if w_logit_pred[i][j] == tgt_targetv[i][j]:
                    tgt_mask[i][j] = WEIGHT_LOSS_DECAY
        
        batch_logpdf = flat_logpdf.view(bz, -1) * tgt_mask
        w_loss_rnn = torch.sum(batch_logpdf)

        w_loss_rnn_ori = torch.sum(flat_logpdf.view(bz, -1) * tgt_mask_ori)
        avg_loss = (- w_loss_rnn_ori / sum(tgt_len)).cpu().data[0]
 
        latent_opt.zero_grad()
        (- w_loss_rnn / sum(tgt_len)).backward()
        latent_opt.step()
        
        avg_loss = (- w_loss_rnn_ori / sum(tgt_len)).cpu().data[0]
        if 1 == 1 and it % 500 == 0:
            logger.info('it %d avg_loss: %f', it, avg_loss)

    return latent_v, avg_loss
    
if COMMAND == 'adv_latent':
    MODEL_FILE = save_dir + '/iter{}.checkpoint'.format(ITER_NUM)
    logger.info('loading form %s.', MODEL_FILE)
    save_d = torch.load(MODEL_FILE)
    for nn in m_dict:
        m_dict[nn].load_state_dict(save_d[nn])
    
    lis_tgt_w = []
    causal_lis = open('../adv_lists/' + DATA_SET + '/bad_words.txt', 'r').readlines()
    lis_tgt_w.extend([l.split() for l in causal_lis])

    success_lis = []   
    bz = 20
    loss_lis = []
    bleu_lis = []
    co = 0
     
    for it in range(len(lis_tgt_w) / bz):
        if 2 == 1 and it > 1: #debug!!
            break 
        bz_tgt_w = lis_tgt_w[bz * it : bz * (it + 1)]
        bz_len = [len(l) for l in bz_tgt_w]
        latent_v, loss_now = adversarial_latent_optimize(bz_tgt_w)
        loss_lis.append(loss_now)
        samples = m_decode_w_rnn.generate(F.tanh(latent_v), sample = False) #don't forget this F.tanh!!!
        samples = samples.cpu().numpy().tolist()
        
        for i in range(len(bz_tgt_w)): 
            co = co + 1
            logger.info('target(%d): %s', (bz * it + i), ' '.join(bz_tgt_w[i]))
            sample = clean_sen([vocab[kk] for kk in samples[i][:(bz_len[i] - 1)]])
            bleu_lis.append(nltk.translate.bleu_score.sentence_bleu([bz_tgt_w[i][1:-1]], sample))
            logger.info('!!!decoding result: |%s|', ' '.join(sample))
            if ' '.join(bz_tgt_w[i][1:-1]) == ' '.join(sample):
                logger.info('^^success!')
                success_lis.append(' '.join(sample))
 
    logger.info('===summarize===')
    logger.info('avg loss : %f', np.mean(loss_lis))
    logger.info('avg bleu: %f', np.mean(bleu_lis))
    logger.info('success rate: %f', len(success_lis) * 1.0 / co)
    logger.info('displaying success lis:')
    for i, s in enumerate(success_lis):
        logger.info('%d: %s', i, s) 

def softmax_idx_el_glue(onehot_v):
    if SOFTMAX_IDX_EL >= 0:
        return torch.cat([F.softmax(onehot_v, dim = 2), torch.zeros(bz, ADV_SRC_LEN_TRY, len(vocab) - SOFTMAX_IDX_EL).cuda()], dim = 2)
    else:
        return F.softmax(onehot_v, dim = 2)

def adversarial_input_optimize(lis_tgt_w, mode = 'embed'):
    assert(mode == 'embed' or mode == 'softmax_idx' or mode == 'linear_idx' or mode == 'softplus_idx' or mode == 'sigmoid_idx')
    bz = len(lis_tgt_w)
    tgt_len = [len(l) - 1 for l in lis_tgt_w]
    max_len = max(tgt_len)
    v_list = []
    if mode == 'embed':
        embed_v = Variable(torch.randn(bz, ADV_SRC_LEN_TRY, EMBED_SIZE).cuda(), requires_grad = True)
        v_list.append(embed_v)
    elif mode == 'softmax_idx': 
        ll = len(vocab)
        if SOFTMAX_IDX_EL >= 0:
            ll = SOFTMAX_IDX_EL
        onehot_v = Variable(torch.randn(bz, ADV_SRC_LEN_TRY, ll).cuda(), requires_grad = True)
        v_list.append(onehot_v)
    elif mode == 'sigmoid_idx':
        onehot_v = Variable(torch.randn(bz, ADV_SRC_LEN_TRY, len(vocab)).cuda() - 3, requires_grad = True)
        v_list.append(onehot_v)
    elif mode == 'linear_idx' or mode == 'softplus_idx':
        onehot_v = Variable(torch.randn(bz, ADV_SRC_LEN_TRY, len(vocab)).cuda(), requires_grad = True)
        v_list.append(onehot_v)
 
    #latent_v_ori = Variable(torch.randn(bz, 1, HIDDEN_SIZE).cuda(), requires_grad = True)
    #latent_v = latent_v_ori.repeat(1, max_len, 1) 
    lis_tgt_w = [l + ['<pad>'] * (max_len + 1 - len(l)) for l in lis_tgt_w]
    lis_tgt_idx = [[vocab_inv[w] for w in l] for l in lis_tgt_w]
    tgt_mb = Variable(torch.LongTensor(lis_tgt_idx).cuda())
    tgt_inputv = tgt_mb[:, :-1]
    tgt_targetv = tgt_mb[:, 1:]
    
    tgt_mask_ori = Variable(torch.FloatTensor([([1] * l + [0] * (tgt_inputv.size(1) - l)) for l in tgt_len]).cuda(), requires_grad = False)

    lr = 1 * bz #when the batch is larger than one, the learning rate becomes small, for mode 'embed' lr=1 works best
    for epoch in range(8): #8!!
        opt_v = torch.optim.SGD(v_list, momentum = 0.9, lr = lr, weight_decay = 1e-5) #the larger the better, it seems
        #opt_v = torch.optim.Adam(v_list, lr = 1e-4) #debug adam!
        #lr = lr * 0.6 #exp shows const 1 is better~
        for it in range(500 * epoch, 500 * (epoch + 1)):
            #print latent_v.size(), tgt_inputv.size(), tgt_len
            #latent_v = latent_v_ori.repeat(1, max_len, 1)
            if mode == 'embed':
                output, _ = m_encode_w_rnn(embed_v, init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM))
            elif mode == 'softmax_idx':
                #output, _ = m_encode_w_rnn(m_embed(F.softmax(onehot_v, dim = 2)), init_lstm_hidden(bz, HIDDEN_SIZE))
                output, _ = m_encode_w_rnn(m_embed(softmax_idx_el_glue(onehot_v)), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM))
            elif mode == 'linear_idx':
                output, _ = m_encode_w_rnn(m_embed(onehot_v), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM))
            elif mode == 'softplus_idx':
                output, _ = m_encode_w_rnn(m_embed(F.softplus(onehot_v)), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM))
            elif mode == 'sigmoid_idx':
                output, _ = m_encode_w_rnn(m_embed(F.sigmoid(onehot_v)).permute(1,0,2), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM))
            output = output.permute(1,0,2)
            #latent = output[:, -1, :].unsqueeze(1).repeat(1, tgt_targetv.size(1), 1)
            #latent = Variable(torch.zeros(latent.size())).cuda() #debug!
            #print 'hn:', hn.size() #[1, bz, HIDDEN_SIZE]
    
            w_logit_rnn, attn_weights = decoder_forward(output, tgt_inputv, tgt_len)
            #w_logit_rnn = m_decode_w_rnn.decode(latent, tgt_inputv, tgt_len)
            flat_output = w_logit_rnn.view(-1, len(vocab))
            flat_target = tgt_targetv.contiguous().view(-1)
            flat_logpdf = lib_pdf.logsoftmax_idxselect(flat_output, flat_target)
            tgt_mask = Variable(torch.FloatTensor([([1] * l + [0] * (tgt_inputv.size(1) - l)) for l in tgt_len]).cuda(), requires_grad = False)
            w_logit_pred = torch.max(w_logit_rnn, dim = 2)[1]
            for i in range(bz):
                for j in range(tgt_len[i]):
                    if w_logit_pred[i][j] == tgt_targetv[i][j]:
                        tgt_mask[i][j] = WEIGHT_LOSS_DECAY
                        #if tgt_targetv[i][j] != 0:
                        #    logger.info('setting to 0.01: |%s|', vocab[tgt_targetv[i][j]])
            batch_logpdf = flat_logpdf.view(bz, -1) * tgt_mask
            w_loss_rnn = torch.sum(batch_logpdf)
            
            w_loss_rnn_ori = torch.sum(flat_logpdf.view(bz, -1) * tgt_mask_ori)

            opt_v.zero_grad() 
            (- w_loss_rnn / sum(tgt_len)).backward(retain_graph = True)
            
            if (mode == 'sigmoid_idx' or mode == 'softmax_idx') and LASSO_LAMBDA > 0:
                if mode == 'sigmoid_idx':
                    sv = F.sigmoid(onehot_v)
                if mode == 'softmax_idx':
                    sv = F.softmax(onehot_v, dim = 2)
                lasso_loss = LASSO_LAMBDA * torch.sum(sv, dim = 2).view(-1).mean()
                lasso_loss += (-LASSO_LAMBDA * 2) * (torch.max(sv, dim = 2)[0]).view(-1).mean()
                lasso_loss.backward(retain_graph = True)

            #print torch.max(torch.abs(onehot_v.grad)) #softmaxed gradient is very small
            opt_v.step()
            
            avg_loss = (- w_loss_rnn_ori / sum(tgt_len)).cpu().data[0]
            if 1 == 1 and it % 200 == 0:
                logger.info('it %d avg_loss: %f', it, avg_loss)
        
    if mode == 'embed':
        return embed_v, avg_loss
    elif mode == 'softmax_idx' or mode == 'linear_idx' or mode == 'softplus_idx' or mode == 'sigmoid_idx':
        return onehot_v, avg_loss

if COMMAND == 'adv_input':
    MODEL_FILE = save_dir + '/iter{}.checkpoint'.format(TEST_ITER)
    logger.info('loading form %s.', MODEL_FILE)
    save_d = torch.load(MODEL_FILE)
    if 1 == 1:
        for nn in m_dict:
            if not nn.endswith('_dp'):
                m_dict[nn].load_state_dict(save_d[nn])
    else:
        logger.info('!!!!!!!!!!!!!!!!!!!!!using RANDOM INTIALIZATION')
        scal = 5
        logger.info('magnifying params by %f', scal)
        for i, p in enumerate(all_params):
            (p.detach())[:] = p * scal #in_place modification


    logger.info('ADV_SRC_LEN_TRY: %d', ADV_SRC_LEN_TRY)
    logger.info('ADV_MODE: %s', ADV_MODE)
    logger.info('WEIGHT_LOSS_DECAY: %f', WEIGHT_LOSS_DECAY)
    logger.info('LASSO_LAMBDA: %f', LASSO_LAMBDA)
    logger.info('FORCE_ONEHOT: %s', str(FORCE_ONEHOT))
    if ADV_MODE == 'softmax_idx':
        logger.info('SOFTMAX_IDX_EL: %d', SOFTMAX_IDX_EL)

    logger.info('the target fn is %s', ADV_TARGET_FN)
    lis_tgt_w = []
    causal_lis = open('../adv_lists/' + DATA_SET + '/' + ADV_TARGET_FN, 'r').readlines()
    lis_tgt_w.extend([l.split() for l in causal_lis if (len(l.split()) > 2)])
    
    success_lis = []   
    bz = 20
    loss_lis = []
    bleu_lis = []
    
    onehotv_lis = [] #save for stats

    co = 0
    for it in range(len(lis_tgt_w) / bz):
        if 2 == 1 and it > 1: #debug!!
            break 
        bz_tgt_w = lis_tgt_w[bz * it : bz * (it + 1)]
        bz_len = [len(l) for l in bz_tgt_w]
        if ADV_MODE == 'embed':
            embed_v, loss_now = adversarial_input_optimize(bz_tgt_w, ADV_MODE)
            output, _ = m_encode_w_rnn(embed_v, init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM))
        elif ADV_MODE == 'softmax_idx':
            onehot_v, loss_now = adversarial_input_optimize(bz_tgt_w, ADV_MODE)
            sv = softmax_idx_el_glue(onehot_v)
            if FORCE_ONEHOT == True:
                sv = force_onehot(sv)
            output, _ = m_encode_w_rnn(m_embed(sv), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM))
        elif ADV_MODE == 'linear_idx':
            onehot_v, loss_now = adversarial_input_optimize(bz_tgt_w, ADV_MODE)
            output, _ = m_encode_w_rnn(m_embed(onehot_v), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM))
        elif ADV_MODE == 'softplus_idx':
            onehot_v, loss_now = adversarial_input_optimize(bz_tgt_w, ADV_MODE)
            output, _ = m_encode_w_rnn(m_embed(F.softplus(onehot_v)), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM))
        elif ADV_MODE == 'sigmoid_idx':
            onehot_v, loss_now = adversarial_input_optimize(bz_tgt_w, ADV_MODE)
            sv = F.sigmoid(onehot_v)
            idx = torch.max(sv, dim = 2)[1]
            logger.info('example input sentence: %s', ' '.join([vocab[d] for d in idx[0]]))
            embed_ori = m_embed(sv)
            output_ori, _ = m_encode_w_rnn(m_embed(sv).permute(1, 0, 2), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM)) #just for comparison
            output_ori = output_ori.permute(1, 0, 2)
            if FORCE_ONEHOT == True:
                sv = force_onehot(sv)
                embed = m_embed(sv)
            output, _ = m_encode_w_rnn(m_embed(sv).permute(1,0,2), init_lstm_hidden(bz, HIDDEN_SIZE, layer_num = LAYER_NUM))
            output = output.permute(1, 0, 2)
            if FORCE_ONEHOT == True:
                norm_lis_h, norm_lis_embed = [], []
                for p in range(output.size(1)):
                    dif = output_ori[:,p,:] - output[:,p,:]
                    dif_norm = torch.norm(dif, p = 2, dim = 1)
                    norm_lis_h.append(torch.mean(dif_norm).item())
                    
                    dif = embed_ori[:,p,:] - embed[:,p,:]
                    dif_norm = torch.norm(dif, p = 2, dim = 1)
                    norm_lis_embed.append(torch.mean(dif_norm).item())
                
                print 'diff in h:', norm_lis_h; print 'diff in embed:', norm_lis_embed; sys.exit(1)
     
        #output = output.permute(1,0,2)
        loss_lis.append(loss_now)
        #print 'output size', output.size()
        #print output[:, -1, :].size()
        #latent = output[:, -1, :].unsqueeze(1).repeat(1, max(bz_len) - 1, 1)
        #print 'latent size', latent.size()
        if MT == 'latent': 
            latent = output[:, -1, :].unsqueeze(1).repeat(1, max(bz_len), 1)
            samples = m_decode_w_rnn.generate(latent, max(bz_len), sample = False)
        if MT == 'attention': 
            latent = output
            samples, _ = m_decode_w_rnn.generate(output, max(bz_len), sample = False)
 
        samples = samples.cpu().numpy().tolist()
        
        if ADV_MODE == 'sigmoid_idx' or ADV_MODE == 'softmax_idx':
            onehotv_lis.append(onehot_v)
         
        for i in range(len(bz_tgt_w)): 
            co = co + 1
            logger.info('target(%d): %s', (bz * it + i), ' '.join(bz_tgt_w[i]))
            sample = clean_sen([vocab[kk] for kk in samples[i][:(bz_len[i] - 1)]])
            bleu_lis.append(nltk.translate.bleu_score.sentence_bleu([bz_tgt_w[i][1:-1]], sample))
            logger.info('!!!decoding result: |%s|', ' '.join(sample))
            if ' '.join(bz_tgt_w[i][1:-1]) == ' '.join(sample):
                logger.info('^^success!')
                success_lis.append(' '.join(sample))
    
    if ADV_MODE in ('sigmoid_idx', 'softmax_idx'):
        fn = 'saves/advinput_onehotvlis_latent_' + ADV_MODE + ADV_TARGET_FN[:-4] + '_FORCEOH' + str(FORCE_ONEHOT) + '_lambda' + str(LASSO_LAMBDA) + '.data' 
        logger.info('saving onehotvlis to %s', fn)
        torch.save(onehotv_lis, fn)
        #advidx_draw.draw_advidx_onehotv('sigmoid', fn = 'figs/advinput' + ADV_MODE + '_att_' + ADV_TARGET_FN[:5] + '_lambda' + str(LASSO_LAMBDA) + '.eps', onehotv_lis = onehotv_lis)
        
    logger.info('===summarize===')
    logger.info('avg loss : %f', np.mean(loss_lis))
    logger.info('avg bleu: %f', np.mean(bleu_lis))
    logger.info('success rate: %f', len(success_lis) * 1.0 / co)
    logger.info('displaying success lis:')
    for i, s in enumerate(success_lis):
        logger.info('%d: %s', i, s) 

if COMMAND == 'adv_enum':
    MODEL_FILE = save_dir + '/iter{}.checkpoint'.format(TEST_ITER)
    logger.info('loading form %s.', MODEL_FILE)
    save_d = torch.load(MODEL_FILE)

    add_log_fn('log_res/'+COMMAND+'/log_list', log_fn, save_dir)    
 
    for nn in m_dict:
        m_dict[nn].load_state_dict(save_d[nn])
    adv_attacks.adv_enumerate_all(MT, m_dict, HIDDEN_SIZE, DATA_SET, ['normal_words.txt', 'random_words.txt', 'reverse_words.txt', 'mal_words.txt'], [EXCLUDE_FN], 2000, vocab, vocab_inv, SRC_SEQ_LEN, TGT_SEQ_LEN, save_dir, ENUM_START_LENGTH = ENUM_START_LENGTH, ENUM_END_LENGTH = ENUM_END_LENGTH, ENUM_START_RATE = ENUM_START_RATE, ENUM_END_RATE = ENUM_END_RATE , log_fn = log_fn, LOG_SUC = LOG_SUC)
 
if COMMAND in 'adv_greedyflip':
    COMMAND = 'adv_greedyflip'
    MODEL_FILE = save_dir + '/iter{}.checkpoint'.format(TEST_ITER)
    logger.info('loading form %s.', MODEL_FILE)
    save_d = torch.load(MODEL_FILE)
    for nn in m_dict: 
        if not nn.endswith('_dp'):
            m_dict[nn].load_state_dict(save_d[nn])
 
    lstmlm = None
    if GE_I_DO == True:
        logger.info('GE_I_DO is true! loading lm from %s', GE_I_LM_FN)
        lstmlm = models.LSTMLM_onehot(EMBED_SIZE, HIDDEN_SIZE, len(vocab), vocab_inv, dropout_rate = 0, layer_num = 1)
        #for name, p in rnn.state_dict().iteritems():
        lstmlm = lstmlm.cuda()
        lstmlm.load_state_dict(torch.load(GE_I_LM_FN))
      
    lis_tgt_w_data_fn = None
   
    adv_attacks.adv_greedyflip(MT, m_dict, HIDDEN_SIZE, DATA_SET, lis_tgt_w_data_fn, [ADV_TARGET_FN], ADV_BZ, vocab, vocab_inv, SRC_SEQ_LEN, TGT_SEQ_LEN, save_dir, NORMAL_WORD_AVG_LOSS = NORMAL_WORD_AVG_LOSS, GIBBSENUM_DECAYLOSS_WEIGHT = GIBBSENUM_DECAYLOSS_WEIGHT, GIBBSENUM_E_NUM = GIBBSENUM_E_NUM, GIBBSENUM_RANDOM_TIME = GIBBSENUM_RANDOM_TIME, GREEDYFLIP_NUM = GREEDYFLIP_NUM, ENUM_START_LENGTH = ENUM_START_LENGTH, ENUM_END_LENGTH = ENUM_END_LENGTH, GE_I_DO = GE_I_DO, GE_I_WORD_AVG_LOSS = GE_I_WORD_AVG_LOSS, GE_I_LAMBDA = GE_I_LAMBDA, GE_I_LM = lstmlm, GE_I_SAMPLE_LOSSDECAY = GE_I_SAMPLE_LOSSDECAY, GE_MAX_ITER_CO = GE_MAX_ITER_CO, log_fn = log_fn, LOG_SUC = LOG_SUC)

if COMMAND in 'adv_gibbsenum':
    COMMAND = 'adv_gibbsenum'
    MODEL_FILE = save_dir + '/iter{}.checkpoint'.format(TEST_ITER)
    logger.info('loading form %s.', MODEL_FILE)
    save_d = torch.load(MODEL_FILE)
    for nn in m_dict: 
        if not nn.endswith('_dp'):
            m_dict[nn].load_state_dict(save_d[nn])
    
    lstmlm = None
    if GE_I_DO == True:
        logger.info('GE_I_DO is true! loading lm from %s', GE_I_LM_FN)
        lstmlm = models.LSTMLM_onehot(EMBED_SIZE, HIDDEN_SIZE, len(vocab), vocab_inv, dropout_rate = 0, layer_num = 1)
        #for name, p in rnn.state_dict().iteritems():
        lstmlm = lstmlm.cuda()
        lstmlm.load_state_dict(torch.load(GE_I_LM_FN))
      
    #add_log_fn('log_res/'+COMMAND+'/log_list', log_fn, save_dir)    
    lis_tgt_w_data_fn = save_dir + '/adv_enum_' + '' + '_lis_tgt_w.data'
    if not DATA_SET.startswith('ptb_chars'):
        lis_tgt_w_data_fn = None
   
    adv_attacks.adv_gibbsenum(MT, m_dict, HIDDEN_SIZE, DATA_SET, lis_tgt_w_data_fn, [ADV_TARGET_FN], ADV_BZ, vocab, vocab_inv, SRC_SEQ_LEN, TGT_SEQ_LEN, save_dir, NORMAL_WORD_AVG_LOSS = NORMAL_WORD_AVG_LOSS, GIBBSENUM_DECAYLOSS_WEIGHT = GIBBSENUM_DECAYLOSS_WEIGHT, GIBBSENUM_E_NUM = GIBBSENUM_E_NUM, GIBBSENUM_RANDOM_TIME = GIBBSENUM_RANDOM_TIME, ENUM_START_LENGTH = ENUM_START_LENGTH, ENUM_END_LENGTH = ENUM_END_LENGTH, GE_I_DO = GE_I_DO, GE_I_WORD_AVG_LOSS = GE_I_WORD_AVG_LOSS, GE_I_LAMBDA = GE_I_LAMBDA, GE_I_LM = lstmlm, GE_I_SAMPLE_LOSSDECAY = GE_I_SAMPLE_LOSSDECAY, GE_MAX_ITER_CO = GE_MAX_ITER_CO, log_fn = log_fn, LOG_SUC = LOG_SUC)

if COMMAND in 'adv_globalenum':
    COMMAND = 'adv_globalenum'
    MODEL_FILE = save_dir + '/iter{}.checkpoint'.format(TEST_ITER)
    logger.info('loading form %s.', MODEL_FILE)
    save_d = torch.load(MODEL_FILE)
    for nn in m_dict: 
        if not nn.endswith('_dp'):
            m_dict[nn].load_state_dict(save_d[nn])
    
    lstmlm = None
    if GE_I_DO == True:
        logger.info('GE_I_DO is true! loading lm from %s', GE_I_LM_FN)
        lstmlm = models.LSTMLM_onehot(EMBED_SIZE, HIDDEN_SIZE, len(vocab), vocab_inv, dropout_rate = 0, layer_num = 1)
        #for name, p in rnn.state_dict().iteritems():
        lstmlm = lstmlm.cuda()
        lstmlm.load_state_dict(torch.load(GE_I_LM_FN))
      
    #add_log_fn('log_res/'+COMMAND+'/log_list', log_fn, save_dir)    
    lis_tgt_w_data_fn = save_dir + '/adv_enum_' + '' + '_lis_tgt_w.data'
    if not DATA_SET.startswith('ptb_chars'):
        lis_tgt_w_data_fn = None
   
    adv_attacks.adv_globalenum(MT, m_dict, HIDDEN_SIZE, DATA_SET, lis_tgt_w_data_fn, [ADV_TARGET_FN], ADV_BZ, vocab, vocab_inv, SRC_SEQ_LEN, TGT_SEQ_LEN, save_dir, NORMAL_WORD_AVG_LOSS = NORMAL_WORD_AVG_LOSS, GIBBSENUM_DECAYLOSS_WEIGHT = GIBBSENUM_DECAYLOSS_WEIGHT, GIBBSENUM_E_NUM = GIBBSENUM_E_NUM, GIBBSENUM_RANDOM_TIME = GIBBSENUM_RANDOM_TIME, ENUM_START_LENGTH = ENUM_START_LENGTH, ENUM_END_LENGTH = ENUM_END_LENGTH, GE_I_DO = GE_I_DO, GE_I_WORD_AVG_LOSS = GE_I_WORD_AVG_LOSS, GE_I_LAMBDA = GE_I_LAMBDA, GE_I_LM = lstmlm, GE_I_SAMPLE_LOSSDECAY = GE_I_SAMPLE_LOSSDECAY, GE_MAX_ITER_CO = GE_MAX_ITER_CO, log_fn = log_fn, LOG_SUC = LOG_SUC)


