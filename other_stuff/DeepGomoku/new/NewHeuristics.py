import numpy as np
from GomokuTools import GomokuTools as gt

class NewHeuristics:
    
    def __init__(self, kappa):
        
        self.kappa = kappa

        self.color_scheme = [ # visualize the offensive/defensive score
            ['#F0F0F0', '#FFC0C0', '#FF9090', '#FF6060', '#FF0000'],
            ['#A0FFA0', '#E8D088', '#FFA080', '#F86040', '#F01808'],
            ['#00FF00', '#B0D818', '#EFB060', '#F07040', '#E03010'],
            ['#00CF00', '#80B014', '#C0A048', '#E08050', '#D04820'],
            ['#00A000', '#307810', '#607020', '#907828', '#C06030']
        ]

        self.compute_line_scores()
        self.compute_total_scores()
        

    def num_offensive(self, o, d):
        s, l, offset = gt.mask2(o, d)
        m2o_bits = gt.as_bit_array(s)[:l]
        max_count = 0
        for w in [2,1,0]:
            i = 0
            while i <= len(m2o_bits) - 2 - w:
                count = sum(m2o_bits[i:i+w+2])
                count = 3*count - (w+2)
                if count > max_count:
                    max_count = count
                i+=1
        if m2o_bits[0] == 0:
            max_count += 1.5
        if m2o_bits[-1] == 0:
            max_count += 1.5

        return max_count        
    
                
    def line_score_for(self, o,d):
        m = gt.mask(o,d)
        m2 = gt.mask2(o,d)
        if m2[1] >= 4 and sum(gt.as_bit_array(m2[0])) >= 1:
            return self.num_offensive(o,d) - 2
        else:
            return 0        

        
    def compute_line_scores(self):
        
        self.line_scores = np.zeros(256*256)
        
        # 81*81 is the range of values for 8-digit base 2 numbers, 
        # which represent arbitrary combinations of 8 stones in a row
        for n in range(81*81):
            xo = gt.base2_to_xo(n)
            o,d = gt.line_for_xo(xo)
            self.line_scores[256*o+d]=self.line_score_for(o,d)

                
    def lookup_line_score(self,o,d):
        return self.line_scores[256*o+d]


    def nhcombine(self, line):
        """
        The neighbourhood score or count.
        Heuristic function of the 4 line scores or counts
        score_or_count: a board of line evaluations: shape = (N,N,4)
        """
        l_ = sorted(line)

        if l_[-1]>7:
            return 8 # Done

        if l_[-1]==7 or (l_[-1] in [4.5,5.5,6,6.5,7.0] and l_[-2] >= 4):
            return 7 # truly strong

        if l_[-1] in [4,5] and l_[-2] in [4,5]:
            return 6.9 # can only be countered by strong counter-attack

        return (l_[-1]**self.kappa + l_[-2]**self.kappa)**(1/self.kappa)        
        

    def compute_total_scores(self):
        self.total_scores = np.zeros([160000])
        values = np.arange(20)/2
        for e in values:
            for ne in values:
                for n in values:
                    for nw in values:
                        v = self.nhcombine([e, ne, n, nw])
                        self.total_scores[int(2*(8000*e+400*ne+20*n+nw))]=v        

                        
    def lookup_total_scores(self, line_scores):
        e,ne,n,nw = np.rollaxis(line_scores, 2, 0)
        indices = (2*(8000*e+400*ne+20*n+nw)).astype(int)
        return self.total_scores[indices]
    
    
