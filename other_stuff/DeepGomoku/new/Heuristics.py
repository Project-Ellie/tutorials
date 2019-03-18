import numpy as np
from GomokuTools import GomokuTools as gt

class Heuristics:
    
    def __init__(self):

        self.color_scheme = [ # visualize the offensive/defensive score
            ['#F0F0F0', '#FFC0C0', '#FF9090', '#FF6060', '#FF0000'],
            ['#A0FFA0', '#E8D088', '#FFA080', '#F86040', '#F01808'],
            ['#00FF00', '#B0D818', '#EFB060', '#F07040', '#E03010'],
            ['#00CF00', '#80B014', '#C0A048', '#E08050', '#D04820'],
            ['#00A000', '#307810', '#607020', '#907828', '#C06030']
        ]

        self.compute_line_scores()
        
        
    def nhcombine(self, score_or_count, kappa=1.2):
        """
        The neighbourhood score or count.
        Heuristic function of the 4 line scores or counts
        score_or_count: a board of line evaluations: shape = (N,N,4)
        """
        e,ne,n,nw = np.rollaxis(score_or_count,2,0)
        return np.power(e**kappa + ne**kappa + n**kappa + nw**kappa, 1/kappa)
        
        
    def compute_line_scores(self):
        self._all_scores = np.zeros(256*256, dtype=int)
        self._all_counts = np.zeros(256*256, dtype=int)
        self._all_scores_and_more = [0 for _ in range(256*256)]
        
        # 81*81 is the range of values for 8-digit base 2 numbers, 
        # which represent arbitrary combinations of 8 stones in a row
        for n in range(81*81):
            xo = gt.base2_to_xo(n)
            o,d = gt.line_for_xo(xo)
            m = gt.mask(o,d)
            m2 = gt.mask2(o,d)
            if m2[1] >= 4 and sum(gt.as_bit_array(m2[0])) >= 1:
                densities = np.multiply(gt.as_bit_array(o), [0,1,2,3,3,2,1,0])
                density = sum(densities)
                no = gt.num_offensive(o,d)
                no = max(no - 2, 0)
                nf = min(sum(gt.as_bit_array(m[1])),5)                
                score = 256*no+16*nf+density
                self._all_scores_and_more[256*o+d]=(xo, score, no, nf, density)
                self._all_scores[256*o+d]=score
                self._all_counts[256*o+d]=no

        
    def lookup_score(self,o,d):
        return self._all_scores[256*o+d]

    
    def lookup_count(self,o,d):
        return self._all_counts[256*o+d]

    
    def lookup_score_and_more(self,o,d):
        return self._all_scores_and_more[256*o+d]

    
