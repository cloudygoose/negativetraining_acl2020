import torch 
import numpy as np
from torch.autograd import Variable
import os, sys
import logging
import math
from models import LSTMLM_onehot
from myutil import *
import nltk
import lm_adv_attacks

COMMAND = 'train'
INPUT_SIZE = 300
HIDDEN_SIZE = 300
LR = 1
LAYER_NUM = 1
DROPOUT = 0.5
BATCH_SIZE = 20
ITER_NUM = 40
SEQ_LEN = 100
OPT = 'sgd' #adam
DO_BLEU = False
DO_EVAL_TRAINLM = False

EXP_ROOT = '../exps/'

import socket
print('hostname:', socket.gethostname())

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print 'CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES']

DATA_SET = 'swda_lm' #'ubuntu_lm' #'ptb_chars_noise'
if len(sys.argv) > 1:
    print 'execing sys.argv[1] for setting:', sys.argv[1]
    exec(sys.argv[1])

if DATA_SET == 'ubuntu_lm_np':
    VOCAB_FILE = '../data/ubuntuDialog/res/vocab_choose.txt'
    TRAIN_DIR = ['../data/ubuntuDialog_lm/process_lm_np/train.txt']
    TEST_FILE = '../data/ubuntuDialog_lm/process_lm_np/test.txt'
    LM_NORMAL_AVGLOGP = None #-4.1221
    TEST_DIR = [TEST_FILE] 
    BATCH_SIZE = 64
    LOG_INTERVAL = 100
    DO_EVAL_TRAINLM = False
    #one working config, uses SGD
    OPT = 'sgd'
    DROPOUT = 0
    HIDDEN_SIZE = 600
    INPUT_SIZE = 300
    LAYER_NUM = 1
    LR = 1
    ADV_TARGET_FN = 'mal_words_all_500P1_2.txt'
    ITER_NUM = 20
    HALF_LR_ITER = 10
    SEQ_LEN = 20
elif DATA_SET == 'os_lm_np':
    TRAIN_DIR = ['../data/opensubtitles/process_lm_np/train.5k.txt']
    VOCAB_FILE = '../data/opensubtitles/vocab.h30k'
    TEST_FILE = '../data/opensubtitles/process_lm_np/test.50.txt'
    TEST_DIR = [TEST_FILE] 
    LM_NORMAL_AVGLOGP = None #-3.87
 
    BATCH_SIZE = 64
    LOG_INTERVAL = 100
    DO_EVAL_TRAINLM = False
    #one working config, uses SGD
    OPT = 'sgd'
    DROPOUT = 0
    HIDDEN_SIZE = 600
    INPUT_SIZE = 300
    LAYER_NUM = 1
    LR = 0.1

    ITER_NUM = 20
    HALF_LR_ITER = 10
    SEQ_LEN = 20
elif DATA_SET == 'swda_lm':
    VOCAB_FILE = '../data/swda_dialogue/vocab_h10k.txt'
    TRAIN_DIR = ['../data/swda_dialogue/process_lm/train.txt']
    TEST_FILE = '../data/swda_dialogue/process_lm/test_25/test.txt'
    TEST_DIR = [TEST_FILE] 
    VALID_DIR = ['../data/swda_dialogue/process_lm/valid_25/valid.txt']
    #LM_NORMAL_AVGLOGP = -3.73
 
    BATCH_SIZE = 64
    LOG_INTERVAL = 100
    DO_EVAL_TRAINLM = False
    #one working config, uses SGD
    OPT = 'sgd'
    DROPOUT = 0.3
    HIDDEN_SIZE = 600
    INPUT_SIZE = 300
    LYAER_NUM = 1
    LR = 1

    ITER_NUM = 20
    HALF_LR_ITER = 10
    SEQ_LEN = 20
else:
    sys.exit(1)

if len(sys.argv) > 1:
    print 're-execing sys.argv[1] for setting:', sys.argv[1]
    exec(sys.argv[1])

torch.manual_seed(1234) #just to be different from the random generator

save_dir = EXP_ROOT + '/lm_baseline/' + DATA_SET + '/LSTM_' + 'LR' + str(LR) + 'H' + str(HIDDEN_SIZE) + 'L' + str(LAYER_NUM) + 'DR' + str(DROPOUT) + 'OPT' + str(OPT)
log_fn = save_dir + '/logC' + COMMAND + '.txt'
print 'save_dir is', save_dir
print 'log_fn is', log_fn

os.system('mkdir -p ' + save_dir)

logging.basicConfig(level = 0)
logger = logging.getLogger()
setLogger(logger, log_fn)

def get_opt(lr, model):
    if OPT == 'sgd':
        o = torch.optim.SGD(model.parameters(), momentum=0.9, lr = lr, weight_decay = 1e-5)
    elif OPT == 'adam':
        o = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-5)
    else:
        sys.exit(1)
    return o

vocab, vocab_inv = getVocab(VOCAB_FILE)
print 'len of vocab:', len(vocab)

criterion = nn.CrossEntropyLoss().cuda()

rnn = LSTMLM_onehot(INPUT_SIZE, HIDDEN_SIZE, len(vocab), vocab_inv, dropout_rate = DROPOUT, layer_num = LAYER_NUM)
#for name, p in rnn.state_dict().iteritems():
rnn = rnn.cuda()

"""moved to myutil
def mask_gen(lengths):
    max_len = lengths[0]
    size = len(lengths)
    mask = torch.ByteTensor(size, max_len).zero_()
    for i in range(size):
        mask[i][:lengths[i]].fill_(1)
    return mask
"""

def train(model, batches, lr, do_train = True, do_log = True):
    #batch should be a LongTensor
    rnn = model
    if do_train == True:
        rnn.train()
    else:
        rnn.eval()
    opt = get_opt(lr, rnn)
    all_loss = 0 
    all_num = 0
    b_count = 0
    for b_idx, b_w, b_len in batches:
        loss = 0
        b_count = b_count + 1
        all_num = all_num + np.sum(b_len)
        
        inputv = Variable(b_idx[:, :-1]).cuda() 
        target = Variable(b_idx[:, 1:]).cuda() 
        mask = Variable(mask_gen(b_len)).cuda()
        #size(batch, length)
        output, _ = rnn(idx2onehot(inputv, len(vocab)), rnn.initHidden(batch_size = inputv.size(0)))
        output = output.masked_select(
            mask.unsqueeze(dim=2).expand_as(output)
        )
         
        target = target.masked_select(mask)
        
        loss = criterion(output.view(target.size(0), -1), target)
        
        assert(target.size(0) == sum(b_len))
                         
        all_loss = all_loss + loss.data[0] * target.size(0)

        if do_train == True:
            rnn.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5)
            #for p in rnn.parameters():
            #    p.grad.data.clamp_(min = -5, max=5)
            #torch.nn.utils.clip_grad_norm(rnn.parameters(), 10, norm_type = inf)
            #print output.grad.data
            #for p in rnn.parameters():
            #    print p.grad.data
            #sys.exit(1)
            opt.step()
        
        if do_log == True and b_count % LOG_INTERVAL == 0:
            logger.info('avg loss at b: %d , %f', b_count, all_loss * 1.0 / all_num)

    logger.info('all_num: %d', all_num)

    return float(all_loss * 1.0 / all_num)

if COMMAND == 'train': #train
    for it in range(ITER_NUM + 1):            
        if OPT == 'sgd' and it >= HALF_LR_ITER:
            LR *= 0.6
        print 'starting iter', it, 'LR:', LR
        #global opt
        #opt = get_opt(LR) # torch.optim.SGD(rnn.parameters(), momentum=0.9, lr = LR, weight_decay = 1e-5)
        batches_train = MyBatchSentences_v2(TRAIN_DIR, BATCH_SIZE, SEQ_LEN, vocab_inv)
        loss_train = train(rnn, batches_train, LR, do_train = True)        
        batches_test = MyBatchSentences_v2(TEST_DIR, BATCH_SIZE, SEQ_LEN, vocab_inv)
        loss_test = train(rnn, batches_test, LR, do_train = False)        
        logger.info('iter: %d, train PPL: %f, test PPL: %f test_loss: %f', it, math.exp(loss_train), math.exp(loss_test), loss_test)
        if it % 1 == 0:
            model_fn = save_dir + '/iter{}.checkpoint'.format(it)
            logger.info('saving to %s', model_fn)
            torch.save(rnn.state_dict(), model_fn)

if COMMAND == 'test': #test
    model_file = save_dir + '/iter{}.checkpoint'.format(ITER_NUM)
    logger.info('loading form %s.', model_file)
    states = torch.load(model_file)
    rnn.load_state_dict(states)
    batches_test = MyBatchSentences_v2(TEST_DIR, BATCH_SIZE, SEQ_LEN, vocab_inv)
    loss_test = train(rnn, batches_test, LR, do_train = False)        
    logger.info('test PPL: %f avg_loss: %f', math.exp(loss_test), loss_test)
    
    """ 
    logger.info('sampling 20 samples')
    samples = rnn.sampleBatch(10, 5, full_length = True)
    for i in range(5):
        sw = [vocab[kk] for kk in samples[i]]
        print ' '.join(sw)

    onehot_v = Variable(idx2onehot(torch.LongTensor(samples).cuda(), len(vocab)), requires_grad = True)
    w_loss = rnn.calMeanLogp(samples, onehot_v, mode = 'input_eou', train_flag = True)
    
    w_loss.mean().backward()
    print onehot_v.grad.size()
    print onehot_v.grad
    """

if COMMAND == 'adv_check':
    config = {
        'LM_NORMAL_AVGLOGP': LM_NORMAL_AVGLOGP,
        'BATCH_SIZE': 50,
        'SEQ_LEN': SEQ_LEN,
        'vocab': vocab,
        'vocab_inv': vocab_inv,
        }    
    model_file = save_dir + '/iter{}.checkpoint'.format(ITER_NUM)
    logger.info('loading form %s.', model_file)
    states = torch.load(model_file)
    rnn.load_state_dict(states)
    lm_adv_attacks.lm_adv_check(rnn, ['../adv_lists/' + DATA_SET + '/' + ADV_TARGET_FN], config)

