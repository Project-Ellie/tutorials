from operator import itemgetter
import numpy as np
from GomokuTools import GomokuTools as gt

class Move:
    def __init__(self, x, y, comment, status):
        self.status=status # 0: ongoing, -1: giving up, 1: declaring victory
        self.x = x
        self.y = y
        self.comment = comment
        
    def __repr__(self):
        return self.comment+ ("" if self.status == -1 
                              else ": (%s, %s)" % (chr(self.x+64), self.y))
    

class StochasticMaxSampler:
    """
    This class allows to sample from the top n of an array of scores, 
    with higher probability for the larger scores. With bias > 1.0,
    the sampler has an even higher bias toward the larger scores.
    """
    def __init__(self, array, topn, bias=1.0):
        self.array = array
        self.bias = bias
        
        top = sorted(list(array), key=itemgetter(1))[-topn:]
        values = [v for _,v in top]
        positions = [p for p,_ in top]

        biased = (values - min(values)) * self.bias
        probs = np.exp(np.asarray(biased))
        probs = probs / probs.sum(0)
        boundaries = [0.]+list(np.cumsum(probs))
        self.probs = probs
        self.choices = list(zip(boundaries[:-1], positions, probs, values))[::-1]

    def draw(self):        
        r = np.random.uniform(0,1)
        for i in self.choices:
            if r > i[0]:
                return i[1]
        
        
    
class HeuristicGomokuPolicy:
    def __init__(self, board, style):
        self.board = board
        self.style = style # 0=aggressive, 1=defensive, 2=mixed
    
    def pos_and_scores(self, index, viewpoint):
        "index: the index of the scored position in a flattened array"
        mpos = np.divmod(index, 15)
        bpos = gt.m2b(mpos, 15)
        return (bpos[0], bpos[1], 
            self.board.scores[viewpoint][mpos[0]][mpos[1]])
    
    def most_critical_pos(self):
        "If this function returns not None, take the move or die."
    
        viewpoint = self.board.current_color
        clean_scores = self.board.get_clean_scores()
        o = np.argmax(clean_scores[1-viewpoint])
        d = np.argmax(clean_scores[viewpoint]) 
        xo, yo, vo = self.pos_and_scores(o, 1-self.board.current_color)
        xd, yd, vd = self.pos_and_scores(d, self.board.current_color)
        #print(xo, yo, vo)
        #print(xd, yd, vd)
        if vo > 7.0:
            return Move(xo, yo, "Immediate win", 1)
        elif vd > 7.0:
            if sorted(clean_scores[viewpoint].reshape(15*15))[-2] > 7.0:
                return Move(0,0,"Two or more immediate threats. Giving up.", -1)
            return Move (xd, yd, "Defending immediate threat", 0)
        elif vo == 7.0:
            return Move(xo, yo, "Win-in-2", 1)
        elif vd == 7.0:
            options = self.defense_options(xd, yd)
            l = list(zip(options, np.zeros(len(options))))
            sampler = StochasticMaxSampler(l, len(options))
            xd, yd = sampler.draw()
            return Move(xd, yd, "Defending Win-in-2", 0)

        elif vo == 6.9:
            return Move(xo, yo, "Soft-win-in-2", 0)
        elif vd == 6.9:
            return Move(xd, yd, "Defending Soft-win-in-2", 0)
        else:
            return None
        
    def defense_options(self, xd, yd):
        """
        return a list of all options that could remedy the critical state
        """
        color = self.board.current_color
        options=[]        
        rc = gt.b2m((xd, yd),self.board.N)
        for direction in ['e', 'ne', 'n', 'nw']:
            step = np.array(gt.dirs()[direction][1])
            for w in [-4,-3,-2,-1,1,2,3,4]:
                r, c = rc+w*step
                if r >= 0 and r < self.board.N and c >= 0 and c < self.board.N:
                    x,y = gt.m2b((r,c), self.board.N)
                    if (x,y) not in self.board.stones:
                            self.board.set(x,y)
                            self.board.compute_scores(color)
                            s=self.board.get_score(color, xd,yd)
                            self.board.undo()
                            self.board.compute_scores(color)
                            if s < 7.0 and self.board.get_score(color, x, y) == 7.0:
                                options.append((x,y))
        options.append((xd,yd))
        return options

                                
    def suggest(self, style=None, bias=1.0, topn=10):
        if style == None:
            style = self.style
        critical = self.most_critical_pos()
        if critical is not None:
            return critical
        else:
            sampler = self.suggest_from_best_value(topn, style, bias)
            r_c = sampler.draw()
            x, y = gt.m2b(r_c, self.board.N)
            return Move(x, y, "Style: %s" % style, 0)


    def suggest_counter(self, style=None, bias=1.0, topn=10):
        if style == None:
            style = self.style
        critical = self.most_critical_pos()
        if critical is not None:
            return critical
        else:
            sampler = self.suggest_from_score(topn, style, bias)
            r_c = sampler.draw()
            x, y = gt.m2b(r_c, self.board.N)
            return Move(x, y, "Style: %s" % style, 0)


        
    def suggest_from_score(self, n, style, bias):
        """
        return a sampler for the top n choices of the given style with a bias > 1.0
        towards the larger scores
        """
        from operator import itemgetter

        clean_scores = self.board.get_clean_scores()

        viewpoint = self.board.current_color
        w_o, w_d = 0.5, 0.5 # relative weights

        if style == 0:  # offensive scores
            scores = clean_scores[1-viewpoint]
            
        elif style == 1: # defensive scores
            scores = clean_scores[viewpoint] 

        elif style == 2: # weighted sum of both scores
            scores = (w_o * clean_scores[1-viewpoint] + 
                      w_d * clean_scores[viewpoint])

        scores = np.ndenumerate(scores)
        return StochasticMaxSampler(scores, n, bias)
            

    def suggest_from_best_value(self, n, style, bias, nscores=10):

        sampler = self.suggest_from_score(max(n, nscores), style, bias)

        scores = []
        for choice in sampler.choices:
            move = gt.m2b(choice[1], 15)
            self.board.set(*move)
            for color in [0,1]:
                self.board.compute_scores(color)

            counter = self.suggest_counter(style=2, bias=1.0, topn=3)# strongest defense assumed
            self.board.set(counter.x, counter.y)
            for color in [0,1]:
                self.board.compute_scores(color)
            value = self.board.get_value()
            self.board.undo().undo()
            for color in [0,1]:
                self.board.compute_scores(color)
            scores.append((choice[1], value))
            
        return StochasticMaxSampler(scores, n, bias)
            