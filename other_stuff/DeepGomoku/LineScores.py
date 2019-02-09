import numpy as np
from HeuristicScore import HeuristicScore

class LineScoresHelper:

    def __init__(self, heuristics=HeuristicScore()):
        self.heuristics = heuristics

        self.lm = [240, 0, 16, 0, 48, 0, 16, 0, 112, 0, 16, 0, 48, 0, 16, 0]
        self.rm = [15, 14, 12, 12, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0]
        self.rl = [4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.ll = [4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0]
        self.offsets=[0, 256, 256+128, 256+128+64, 256+128+64+32]

        # pre-compute the scores
        self.cscores = [
            heuristics.cscore([self.as_bits(line), self.as_bits(0x100 >> n)]) 
            for n in range(5)
            for line in range(0x100 >> n)
        ]
        mag = np.array(self.cscores).T[0]
        mul = np.array(self.cscores).T[1]
        self.scores = mul**(1/heuristics.kappa1) * mag        
        
    def as_bits(self, n):
        return [np.sign(n  & (1<<(7-i))) for i in range(8)]        
    
    def left_mask(self, n):
        i = (n&0xF0) >> 4
        return (self.lm[i], self.ll[i])

    def right_mask(self, n):
        i = n & 0xF
        return self.rm[i], self.rl[i]

    def mask(self, n):
        lm_ = self.left_mask(n)
        rm_ = self.right_mask(n)
        return np.add(lm_[0], rm_[0]), lm_[1], rm_[1]

    def free_range(self, line, c):
        to_mask = c
        use = 1 - c
        m, ll, lr = self.mask(line[use])
        return (line[to_mask] & m), ll, lr

    def lookup_score(self, line, c=0):
        fr, ll_, lr_ = self.free_range(line, c)
        len_ = lr_ + ll_
        if len_ < 4: 
            return (0,0), 0.0
        offset = self.offsets[8-len_]
        n_ = fr >> (4-lr_)
        index = offset + n_
        cscore_ = self.cscores[index]
        score = self.scores[index]
        return cscore_, score


