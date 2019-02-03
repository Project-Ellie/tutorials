import numpy as np
class HeuristicScore:
    
    def __init__(self, kappa0=2, kappa1=3):
        self.kappa0 = kappa0
        self.kappa1 = kappa1
        
    def f_range(self, line, fof=0):
        """
        The largest adversary-free range within a given line
        
        Args:
            line: 8x2 integer array that represents the stones
            fof:  friend or foe? 0 to look at black, 1 to consider white
        """

        i=3
        while i>=0 and line[1-fof][i]==0:
            i-=1
        left = i+1
        i=4
        while i<=7 and line[1-fof][i]==0:
            i+=1
        right = i-1
        return np.array(line[fof][left:right+1])


    def cscore(self, line, fof=0):
        """
        count how many sub-lines of 5 come with the max number of stones
        Example: "oo.x*xx.." : The max num of blacks if obviously 3. And there are
                 two different adversary-free sub-lines counting three, namely '.x*xx' and 'x*xx.'.
                 Thus the cscore would be (3,2)

        Args:
            line: 8x2 integer array that represents the stones 
            fof:  friend or foe? 0 to look at black, 1 to consider white
        """

        fr = self.f_range(line, fof)
        counts = []
        for i in range(len(fr)-4):
            counts.append(sum(fr[i:i+5]))            
        m = max(counts) if counts else 0
        c = sum(np.array(counts) == max(counts)) if counts else 0
        c = min(c,2)
        return (m, c)
    

    def score(self, line, fof=0):
        """
        weighted sum of the count score
        """
        cscore = self.cscore(line, fof)
        mag, mul = cscore
        return mul**(1/self.kappa1) * mag
    
    
    def scores(self, lines, fof=0):
        return [self.score(line, fof) for line in lines]
    
    
    def total_score(self, lines, fof=0):
        """
        total score of the given list of lines
        """
        scores = [self.score(line, fof) for line in lines]
        return sum(s**self.kappa0 for s in scores)**(1/self.kappa0) 