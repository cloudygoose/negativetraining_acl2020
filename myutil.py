import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle
import logging
import os 
       
logger = logging.getLogger()

class DialogueBatches(object):
    def __init__(self, dirname, batch_size, src_max_len, tgt_max_len, vocab_inv, config, his_len = 2, lower = False, sw = '<eou>'):
        logger.info('initing DialogueBatches lower: %s his_len: %s filter_turn: %s skip_unk: %s', str(lower), str(his_len), str(config['filter_turn']), str(config['skip_unk']))
        self.dirname = dirname
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.batch_size = batch_size
        self.vocab_inv = vocab_inv
        self.lower = lower
        self.his_len = his_len 
        self.filter_turn = config['filter_turn']
        self.skip_unk = config['skip_unk']
         
    @staticmethod 
    def init_inf(dirname, batch_size, src_max_len, tgt_max_len, vocab_inv, config, his_len = 2, lower = False, sw = '<eou>', name = ''):
        iter_count = 0
        while True:
            batches = DialogueBatches(dirname, batch_size, src_max_len, tgt_max_len, vocab_inv, config, his_len = his_len, lower = lower, sw = sw)
            for a, b, c, d in batches:
                yield a, b, c, d
            iter_count = iter_count + 1
            logger.info('data sweep time %d %s', iter_count, name)

    def arrange_mb(self, res):
        vocab_inv = self.vocab_inv
        
        len_src = 0
        len_tgt = 0
        
        src_lis = []
        tgt_lis = []

        tgt_lens = []

        for i in range(self.batch_size):
            src = []
            his = res[i]['his']
            for h in his:
                src.extend(h)
                src.append('<eou>')
            if len(src) > self.src_max_len:
                src = src[len(src)-self.src_max_len:]
            len_src = max(len_src, len(src))
            src_lis.append(src)

            tgt = res[i]['tgt']
            tgt = ['<s>'] + tgt + ['</s>']
            if len(tgt) > self.tgt_max_len:
                tgt = tgt[:self.tgt_max_len]
            tgt_lens.append(len(tgt) - 1)
            len_tgt = max(len_tgt, len(tgt))
            tgt_lis.append(tgt) 

        mb_src = torch.LongTensor(self.batch_size, len_src)
        mb_tgt = torch.LongTensor(self.batch_size, len_tgt)
        
        for i in range(self.batch_size): 
            src = src_lis[i]
            tgt = tgt_lis[i]

            src = ['<pad>'] * (len_src - len(src)) + src
            tgt = tgt + ['<pad>'] * (len_tgt - len(tgt))
            
            #print 'src:', src 
            #print 'tgt:', tgt
             
            for j in range(len_src):
                mb_src[i][j] = vocab_inv[src[j]] if src[j] in vocab_inv else vocab_inv['<unk>']
            for j in range(len_tgt):
                mb_tgt[i][j] = vocab_inv[tgt[j]] if tgt[j] in vocab_inv else vocab_inv['<unk>']

        return mb_src, mb_tgt, tgt_lens, src_lis, tgt_lis

    def __iter__(self):
        
        vocab_inv = self.vocab_inv
        if type(self.dirname) != type([]):
            fns = os.listdir(self.dirname)
            fns = [os.path.join(self.dirname, fn) for fn in fns]
        else:
            fns = self.dirname
        
        res = []

        for fname in fns:
            for line in open(fname):
                his = []
                lines = line.split('<eou>')
                if len(lines) <= 1: #we only consider the dialogues of more than two sentences
                    continue
                co = 0
                for l in lines:
                    co = co + 1
                    l = l.split()
                    if self.skip_unk == True:
                        s_l = [w for w in l if w in self.vocab_inv]
                        if len(s_l) < len(l):
                            print s_l, l
                        l = s_l
                    if len(l) <= 0:
                        continue
                    if len(his) > 0:
                        if co % self.filter_turn == 0:
                            res.append({'his':list(his), 'tgt':list(l)})
                        if len(res) == self.batch_size:
                            yield self.arrange_mb(res) 
                            res = []
                    his.append(l)
                    if len(his) > self.his_len:
                        his = his[1:]

class MyBatchSentences_v2(object):
    def __init__(self, dirname, batch_size, max_len, vocab_inv, lower = True, do_sort = True):
        self.dirname = dirname
        self.max_len = max_len
        self.batch_size = batch_size
        self.vocab_inv = vocab_inv
        self.lower = lower
        self.do_sort = do_sort

    def __iter__(self):
        ss_now = []
        lens = []
        vocab_inv = self.vocab_inv
        if type(self.dirname) != type([]):
            fns = os.listdir(self.dirname)
            self.dirname = [os.path.join(self.dirname, fn) for fn in fns]
        for fname in self.dirname:
            for line in open(fname):
                if self.lower == True:
                    l = line.strip().lower().split()
                else:
                    l = line.strip().split()
                if len(l) == 0:
                    continue
                if l[-1] == '</s>' or l[-1] == '<eos>':
                    l = l[:-1]
                l.append('</s>')
                if len(l) > self.max_len - 1:
                    l = l[:self.max_len]
                if l[0] == '<s>':
                    l = l[1:]
                lens.append(len(l)) #len does not count <s>
                l = ['<s>'] + l 
                ss_now.append(l)

                if len(ss_now) == self.batch_size:
                    if self.do_sort == True:
                        ss_now, lens = length_sort(ss_now, lens)
                    ss_idx = [[(vocab_inv[w] if (w in vocab_inv) else vocab_inv['<unk>']) for w in l] for l in ss_now]
                    ss_idx = [(l + [0] * (max(lens) + 1 - len(l))) for l in ss_idx]
                    ss_idx = torch.LongTensor(ss_idx).cuda()
                    yield ss_idx, ss_now, lens
                    ss_now = [] 
                    lens = []

class MyStatDic(object):
    def __init__(self):
        self.dd = {}

    def append_dict(self, new_d, keys = None):
        dd = self.dd
        if keys == None:
            keys = new_d.keys()
        for k in keys:
            if not k in dd: dd[k] = []
            dd[k].append(new_d[k])

    def log_mean(self, keys = None, last_num = 0, log_pre = ''):
        ss = ''; dd = self.dd;
        if keys == None: keys = dd.keys()
        for k in keys:
            ss = ss + k + ': ' + str(np.mean(dd[k][-last_num:])) + ' '
        logger.info(log_pre + ' (last_num %d ) ' + ss, last_num)

def force_onehot(sv):
    pl = torch.zeros(sv.size()).cuda()
    idx = torch.max(sv, dim = 2)[1]
    for i1 in range(idx.size(0)):
        for i2 in range(idx.size(1)):
            pl[i1, i2, idx[i1, i2].item()] = 1
    return pl

def clean_sen(sen):
    if len(sen) == 0:
        return sen
    for i in range(len(sen)):
        if sen[i] == '</s>':
            sen = sen[:i]
            break
    while len(sen) > 0 and (sen[0] == '<pad>' or sen[0] == '<s>'):
        sen = sen[1:]
    while len(sen) > 0 and (sen[-1] == '<pad>' or sen[-1] == '</s>'):
        sen = sen[:-1]
    return sen

def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)

def getVocab(fn, lower = False):
    logger.info('learning vocab from %s, lower: %s', fn, str(lower))

    vocab = ['</s>', '<unk>', '<s>', '<pad>', '<eou>']
    vocab_inv = {'</s>':0, '<unk>':1, '<s>':2, '<pad>':3, '<eou>':4}

    for line in open(fn):
        if lower == True:
            l = line.strip().lower().split()
        else:
            l = line.strip().split()

        for w in l:
            if not w in vocab_inv:
                vocab.append(w)
                vocab_inv[w] = len(vocab) - 1

    return vocab, vocab_inv

def mask_gen(lengths, ty = 'Byte'):
    max_len = max(lengths)
    size = len(lengths)
    if ty == 'Byte':
        mask = torch.ByteTensor(size, max_len).zero_()
    elif ty == 'Float':
        mask = torch.FloatTensor(size, max_len).zero_()
    for i in range(size):
        mask[i][:lengths[i]].fill_(1)
    return mask

def setLogger(logger, LOG_FN):
    logger.handlers = []
    fileHandler = logging.FileHandler(LOG_FN, mode = 'w') #could also be 'a'
    logFormatter = logging.Formatter("%(asctime)s [%(funcName)-15s] [%(levelname)-5.5s]  %(message)s")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

def check_memory():
# return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    #logger.info('current memory : %d MB', mem)
    if mem > 7 * 1024: #large than 8G
        logger.info('too much memory, forcing shutdown...')
        sys.exit(1)
    return mem

def idx2onehot(idx, vocab_size):
    assert(len(idx.size()) == 2)
    #idx should be a integer tensor
    l = idx.size(0) * idx.size(1)
    oh = torch.zeros(l, vocab_size).cuda()
    oh[range(l), idx.contiguous().view(-1).cpu()] = 1
    oh = oh.view(idx.size(0), idx.size(1), vocab_size).cuda()
    return oh

def init_lstm_hidden(bsz, hz, layer_num = 1):
    zeros1 = Variable(torch.zeros(layer_num, bsz, hz)).cuda()
    zeros2 = Variable(torch.zeros(layer_num, bsz, hz)).cuda()
    return zeros1, zeros2

def add_log_fn(fn, log_fn, save_dir):
    logger.info('attaching log_fn %s to %s', log_fn, fn)
    ff = open(fn, 'a')
    ff.write('{} \t {}\n'.format(log_fn, save_dir))
    ff.close()

def countSenAcc(target, dec, vocab_inv):
    #accepts two lists of lists
    c = 0
    eos_id = vocab_inv['</s>']
    for l in range(len(target)):    
        for j in range(len(target[l])):
            if target[l][j] == dec[l][j]:
                c = c + 1
            if target[l][j] == eos_id:
                break
    return c
