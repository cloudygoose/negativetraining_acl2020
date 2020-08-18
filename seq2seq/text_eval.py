import os, sys, math

def text_entropy(sen_lis, k):
    #sen_lis is like [['i','am','you','</s>'] ...]
    #assume it is lowered case, and clean
    dd, num = {}, 0
    for sen in sen_lis:
        for i in range(0, len(sen) - k + 1):
            num += 1
            tt = ' '.join(sen[i:i+k])
            #print tt
            if not tt in dd: dd[tt] = 0
            dd[tt] += 1
    
    entro = 0.0
    for tt in dd:
        prob = float(dd[tt] * 1.0) / num
        entro = entro - math.log(prob) * prob
    return entro
    
def ex_text_entropy():
    s1 = 'i am you </s>'.split()
    s2 = 'hello guys </s>'.split()
    s3 = 'come on </s>'.split()
    s4 = 'i like you </s>'.split()
    ss = [s1, s2, s3, s4]
    print text_entropy(ss, 2)

#ex_text_entropy()


