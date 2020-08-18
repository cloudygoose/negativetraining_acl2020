import sys
import torch
import numpy as np
import scipy.stats
from torch.nn import Softmax
import torch.nn.functional
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from matplotlib import pyplot
import math

#import pyro.distributions.normal

def onehot_maskgen(sz, idx, ty = 'byte'): #or float
    #sz: list of size
    #idx: LongTensor
    if ty == 'byte':
        msk = torch.ByteTensor(sz)
    elif ty == 'float':
        msk = torch.FloatTensor(sz)       
    msk.fill_(0)
    msk[torch.LongTensor(range(sz[0])), idx.cpu()] = 1
    if idx.is_cuda == True:
        msk = msk.cuda()
    return Variable(msk)

def sample_gumbel(input_x):
    #from https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530
    noise = torch.rand(input_x.size())
    if input_x.is_cuda == True:
        noise = noise.cuda()
    eps = 1e-9
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise)

def gumbel_softmax_sample(input_pi, temp):
    temperature = temp #0.1 for sharper, 1 for smooth
    noise = sample_gumbel(input_pi)
    x = (torch.log(input_pi) + noise) / temperature
    x = F.softmax(x, dim = len(input_pi.size()) - 1)
    return x.view_as(input_pi)

def ex_gumbel_softmax_sample():
    m1 = Variable(torch.FloatTensor([[3.0, 1.0, 1.0]] * 10))
    print gumbel_softmax_sample(m1, 0.1)

def sample_normal(mu, sigma, reparameterize = True):
    #mu and sigma should be of the same size of generated data
    sam = Variable(torch.randn(mu.size(0), mu.size(1)))
    if mu.is_cuda == True:
        sam = sam.cuda()
    sam = sam * sigma
    sam = sam + mu
    if reparameterize == False:
        sam = Variable(sam.data, requires_grad = False)
    return sam

def ex_sample_normal():
    num = 5
    mu = Variable(torch.ones(num, 2))
    mu[1, :] = 2
    mu[2, :] = 3
    sig = Variable(torch.ones(num, 2) * 0.1)
    print sample_normal(mu, sig)

def logsoftmax_idxselect(logits, idx):
    #logits: Variable, FloatTensor
    #idx: Variable, LongTensor
    out = torch.nn.functional.log_softmax(logits, dim = 1)
    msk = onehot_maskgen(out.size(), idx.data)
    out = torch.masked_select(out, msk)
    return out

def ex_logsoftmax_idxselect():
    aa = Variable(torch.ones(3, 4), requires_grad = True)
    opt = optim.SGD([aa], lr=0.1)
    idx = Variable(torch.LongTensor([1,2,3]))
    loss = logsoftmax_idxselect(aa, idx)
    loss = (-loss.mean())
    loss.backward()
    print aa.grad
    opt.step()
    print aa

def kl_discrete(p, q):
    mid = torch.log(p + 1e-8) - torch.log(q + 1e-8)
    mid = p * mid
    return torch.sum(mid, 1)

def ex_kl_discrete():
    p1 = Variable(torch.FloatTensor([[0.5, 0.5], [0.3, 0.7]])) 
    p2 = Variable(torch.FloatTensor([[0.5, 0.5], [0.7, 0.3]]))
    
    kl = kl_discrete(p1, p2)
    print 'kl:', kl
 
def kl_normal_diag(m1, s1, m2, s2):
    #kl(p1||p2)
    dim = m1.size(1)
    p3 = torch.sum((m2 - m1) * (m2 - m1) / (s2 * s2), 1)
    p2 = torch.sum(s1 * s1 / (s2 * s2), 1)
    p1 = 2 * torch.sum(torch.log(s2 + 1e-8) - torch.log(s1 + 1e-8), 1)
    res = (p1 - dim + p2 + p3) / 2.0
    return res

def ex_kl_normal_diag():
    m1 = Variable(torch.FloatTensor([[1.0, 1.0], [0.0, 0.0]])) 
    s1 = Variable(torch.FloatTensor([[0.1, 0.1], [1.0, 1.0]]))
    
    m2 = Variable(torch.FloatTensor([[2.0, 2.0], [0.0, 0.0]])) 
    s2 = Variable(torch.FloatTensor([[1, 1], [3.0, 3.0]]))
    
    kl = kl_normal_diag(m1, s1, m2, s2)
    print 'kl:', kl
 
def normal_diag_logp(data, mu, sigma):
    #data, mu, sigma: all Variable Float with the same size
    res = - (data - mu) * (data - mu) / (2.0) / (sigma * sigma)
    res = res - 0.5 * math.log(2 * 3.1415926) - torch.log(sigma)
    res = torch.sum(res, 1)
    return res

def ex_normal_diag_logp():
    print 'pyro answer'
    mu = Variable(torch.FloatTensor([1.0, 1.0]), requires_grad = True) 
    sigma = Variable(torch.FloatTensor([0.1, 0.1]), requires_grad = True)
    normal = pyro.distributions.Normal(mu, sigma) 
    data = Variable(torch.FloatTensor([[1.0, 1.1],[2.0, 2.1],[3.0, 3.1]]), requires_grad = True)
    loss = - normal.batch_log_pdf(data)
    print '-logp:', loss
    loss.mean().backward()
    print 'mu.grad', mu.grad
    print 'sig.grad', sigma.grad
    print 'data.grad', data.grad

    mu.grad.fill_(0)
    sigma.grad.fill_(0)
    data.grad.fill_(0)
    print 'trying mypyro'
    loss = - normal_diag_logp(data, mu.expand_as(data), sigma.expand_as(data))
    print '-logp:', loss
    loss.mean().backward()
    print 'mu.grad', mu.grad
    print 'sig.grad', sigma.grad
def gaussian_kernel(x, y, s2 = None):
    if s2 == None:
        s2 = x.size(1) * 1.0
    l2 = - torch.sum(((x - y) * (x - y)), dim = 1) / s2 
    return torch.exp(l2)

def compute_mmd(x, y, s2 = None):    
    """ my old simple way
    assert(x.size(0) % 2 == 0)
    hh = x.size(0) / 2
    xx = gaussian_kernel(x[:hh], x[hh:], s2).mean()
    yy = gaussian_kernel(y[:hh], y[hh:], s2).mean()
    xy = gaussian_kernel(x, y, s2).mean()
    return xx + yy - 2 * xy """
    #"""
    bz = x.size(0)
    dim = x.size(1)
    bz_f = bz * 1.0
    #print 'x:', x
    half_size = (bz * bz - bz) / 2
    norms_x = torch.sum(x * x, dim = 1).view(bz, 1).expand(bz, bz)
    dots_x = torch.mm(x, torch.t(x))
    dis_x = norms_x + torch.t(norms_x) - 2. * dots_x
    #print 'dis_x:', dis_x

    #print 'y:', y
    norms_y = torch.sum(y * y, dim = 1).view(bz, 1).expand(bz, bz)
    dots_y = torch.mm(y, torch.t(y))
    dis_y = norms_y + torch.t(norms_y) - 2. * dots_y
    #print 'dis_y:', dis_y

    dots_xy = torch.mm(x, torch.t(y))
    dis_xy = norms_x + torch.t(norms_y) - 2. * dots_xy
    #print 'dis_xy', dis_xy 
    if s2 == None:
        s2 = sorted(dis_xy.view(-1).data.cpu().numpy().tolist())[half_size]
        s2 += sorted(dis_y.view(-1).data.cpu().numpy().tolist())[half_size]
    #print 's2:', s2
    res1 = torch.exp(- dis_x / 2. / s2)
    res1 = res1 + torch.exp(- dis_y / 2. / s2)
    res1 = res1 * (1 - torch.eye(bz).cuda())
    res1 = res1.sum() / (bz_f * bz_f - bz_f)
    #print 'res1:', res1
    res2 = torch.exp(- dis_xy / 2. / s2)
    res2 = res2.sum() * 2. /(bz_f * bz_f)
    #print 'res2:', res2
    return res1 - res2
    #"""

def ex_compute_mmd():
    """
    num = 100
    dim = 2
    mu = Variable(torch.ones(num, dim)).cuda()
    sig = Variable(torch.ones(num, dim) * 0.1).cuda()
    s1 = sample_normal(mu, sig)
    s2 = sample_normal(mu + 1, sig)
    s3 = sample_normal(mu + 2, sig)
    s4 = sample_normal(mu + 3, sig)
    s5 = sample_normal(mu + 4, sig)
    s6 = sample_normal(mu + 5, sig)
    s7 = sample_normal(mu + 6, sig)
    print compute_mmd(s1, s2)
    print compute_mmd(s1, s3)
    print compute_mmd(s1, s4)
    print compute_mmd(s1, s5)
    print compute_mmd(s1, s6)
    print compute_mmd(s1, s7)
    """
    sample_pz = torch.FloatTensor([[1.,1.], [2.,2.], [3.,3.]]).cuda()
    sample_qz = torch.FloatTensor([[1.5,1.], [2.5,2.], [3.5,3.]]).cuda()
    print compute_mmd(sample_pz, sample_qz)     

ex_compute_mmd()
#ex_kl_discrete()
#ex_kl_normal_diag()
#ex_logsoftmax_idxselect()
#ex_normal_diag_logp()
#ex_sample_normal()
#ex_gumbel_softmax_sample()
 
