import numpy as np
class HeuristicScore:
    
    def __init__(self, kappa0=2, kappa1=5):
        self.kappa0 = kappa0
        self.kappa1 = kappa1
        
    def f_range(self, line, c=0, edges=(None, None)):
        """
        The largest adversary-free range within a given line
        
        Args:
            line: 8x2 integer array that represents the stones
            fof:  friend or foe? 0 to look at black, 1 to consider white
        """

        i=3
        while i >= 0 and line[1-c][i] == 0 and i != edges[0]:
            i-=1
        left = i + 1
        i=4
        while i <= 7 and line[1-c][i] == 0 and i != edges[1]:
            i+=1
        right = i-1
        return np.array(line[c][left:right+1])


    def cscore(self, line, c=0, edges=(None, None)):
        """
        count how many sub-lines of 5 come with the max number of stones
        Example: "oo.x*xx.." : The max num of blacks if obviously 3. And there are
                 two different adversary-free sub-lines counting three, namely '.x*xx' and 'x*xx.'.
                 Thus the cscore would be (3,2)

        Args:
            line: 8x2 integer array that represents the stones 
            c:  color: 0 to look at black, 1 to consider white
        """

        fr = self.f_range(line, c, edges)
        counts = []
        for i in range(len(fr)-3):
            counts.append(sum(fr[i:i+4]))            
        m = max(counts) if counts else 0
        c_ = sum(np.array(counts) == max(counts)) if counts else 0
        c_ = min(c_,3)
        return (m, c_)
    

    def score(self, line, c=0, edges=(None, None)):
        """
        weighted sum of the count score
        """
        cscore = self.cscore(line, c, edges)
        mag, mul = cscore
        return mul**(1/self.kappa1) * mag
    
    
    def scores(self, lines, c=0, all_edges=None):
        if all_edges is None:
            all_edges = [(None, None), (None, None), (None, None), (None, None)]

            return [self.score(line, c, edges=edges) 
                for line, edges in zip(lines, all_edges)]
    
    
    def total_score(self, lines, c=0, all_edges=None):
        """
        total score of the given list of lines
        """
        if all_edges is None:
            all_edges = [(None, None), (None, None), (None, None), (None, None)]

        scores = [self.score(line, c, edges=edges) 
                  for line, edges in zip(lines, all_edges)]
        
        return self.euclidean_sum(scores)
    
    def euclidean_sum(self, scores):
        return sum(s**self.kappa0 for s in scores)**(1/self.kappa0) 