from operator import itemgetter
import numpy as np
from copy import deepcopy
from GomokuTools import GomokuTools as gt

MUST_DEFEND=-1
CAN_ATTACK=1

class Move:
    def __init__(self, x, y, comment, status, off_def=0):
        self.status=status # 0: ongoing, -1: giving up, 1: declaring victory
        self.x = x
        self.y = y
        self.comment = comment
        self.off_def = off_def # +1 = can attack, -1 = must defend
        
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
        
SURE_WIN = 1
IMMEDIATE_WIN = 2
SURE_LOSS = -1
ONGOING = 0
    
class HeuristicGomokuPolicy:
    def __init__(self, board, style, bias, topn, threat_search):
        """
        Params:
        style: 0=aggressive, 1=defensive, 2=mixed
        bias: bias towards the higher values when choosing from a distribution. Low bias tries more.
        topn: number of top moves to include in distribution
        threat_search: The ThreatSearch instance
        """
        self.board = board
        self.style = style # 0=aggressive, 1=defensive, 2=mixed
        self.bias = bias
        self.topn = topn
        self.ts = threat_search
        
    
    def pos_and_scores(self, index, viewpoint):
        "index: the index of the scored position in a flattened array"
        mpos = np.divmod(index, self.board.N)
        bpos = gt.m2b(mpos, self.board.N)
        return (bpos[0], bpos[1], 
            self.board.scores[viewpoint][mpos[0]][mpos[1]])
    
    def most_critical_pos(self, consider_threat_sequences=True):
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
            return Move(xo, yo, "Immediate win", IMMEDIATE_WIN, CAN_ATTACK)
        elif vd > 7.0:
            if sorted(clean_scores[viewpoint].reshape(self.board.N*self.board.N))[-2] > 7.0:
                return Move(0,0,"Two or more immediate threats. Giving up.", SURE_LOSS, MUST_DEFEND)
            return Move (xd, yd, "Defending immediate threat", ONGOING, MUST_DEFEND)
        elif vo == 7.0:
            return Move(xo, yo, "Win-in-2", SURE_WIN, CAN_ATTACK)
        elif vd == 7.0:
            options = self.defense_options(xd, yd)
            l = list(zip(options, np.zeros(len(options))))
            sampler = StochasticMaxSampler(l, len(options))
            xd, yd = sampler.draw()
            return Move(xd, yd, "Defending Win-in-2", ONGOING, MUST_DEFEND)

        elif vo == 6.9:
            return Move(xo, yo, "Soft-win-in-2", ONGOING, CAN_ATTACK)
        elif vd == 6.9:
            return Move(xd, yd, "Defending Soft-win-in-2", ONGOING, MUST_DEFEND)
        
        elif consider_threat_sequences:
            # I might have a winning threat sequence...
            moves, won = self.ts.is_tseq_won(self.board)
            if won:
                x, y = moves[0]
                #print(moves)
                return Move(x, y, "Pursuing winning threat sequence", ONGOING, CAN_ATTACK)

            # I might need to defend a threat sequence...
            moves = self.ts.is_tseq_threat(self.board)
            if moves:
                #print(moves)
                x, y = moves[0]
                return Move(x, y, "Defending lurking threat sequence", ONGOING, MUST_DEFEND)
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
                            clean_scores = self.board.get_clean_scores()
                            s=clean_scores[color][r][c]
                            self.board.undo()
                            self.board.compute_scores(color)
                            clean_scores = self.board.get_clean_scores()
                            s1=clean_scores[color][r][c]
                            if s < 7.0 and s1 == 7.0:
                                options.append((x,y))
        options.append((xd,yd))
        return options

                                
    
                                
    def suggest(self, style=None, bias=None, topn=None):
        style = style or self.style
        bias = bias or self.bias
        topn = topn or self.topn
            
        critical = self.most_critical_pos()
        if critical is not None:
            return critical
        else:
            sampler = self.suggest_from_best_value(topn, style, bias)
            r_c = sampler.draw()
            x, y = gt.m2b(r_c, self.board.N)
            return Move(x, y, "Style: %s" % style, 0)


    def suggest_naive(self, style=None, bias=1.0, topn=10):
        """
        Non-deterministic! Samples from the highest naive scores
        """
        if style == None:
            style = self.style
        critical = self.most_critical_pos(consider_threat_sequences=False)
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
            move = gt.m2b(choice[1], self.board.N)
            self.board.set(*move)
            for color in [0,1]:
                self.board.compute_scores(color)

            counter = self.suggest_naive(style=2, bias=1.0, topn=3)# naive-strongest defense assumed
            if counter.status == -1:
                print("Opponent gave up")
                print(str(counter))
                scores = [(choice[1], np.float32(6.95))] # 6.95 is a 'marker'. 
                self.board.undo()
                break
            
            self.board.set(counter.x, counter.y)
            
            for color in [0,1]:
                self.board.compute_scores(color)
            value = self.board.get_value()
            
            self.board.undo().undo()
            for color in [0,1]:
                self.board.compute_scores(color)
            
            scores.append((choice[1], value))
            
        return StochasticMaxSampler(scores, n, bias)
            
        
        

def least_significant_move(board):
    scores = board.get_clean_scores(tag=1) # tag occupied positions non-zero
    least_score = scores[0] + scores[1]
    index = np.argmin(least_score)
    r, c = np.divmod(index,board.N)

    pos = gt.m2b((r,c), board.N)
    return pos




class ThreatSearch():
    
    def __init__(self, max_depth, max_width):
        self.max_depth = max_depth
        self.max_width = max_width
    
    def is_threat(self, policy, x, y):
        policy.board.set(x,y)
        mcp = policy.most_critical_pos(consider_threat_sequences=False)
        policy.board.undo()
        return mcp    
    
    def is_tseq_won(self, board, max_depth=None, max_width=None):
        """if winnable by a threat sequence, returns that sequence as a list of moves.
        Otherwise returns an empty list"""
        
        max_depth = max_depth or self.max_depth
        max_width = max_width or self.max_width
        
        board = deepcopy(board)
        
        # Need a new policy on the copy.
        policy = HeuristicGomokuPolicy(board=board, style=0, bias=1.0, topn=5, threat_search=self)

        return self._is_tseq_won(board, policy, max_depth, max_width, [])

    
    def _is_tseq_won(self, board, policy, max_depth, max_width, moves):

        if max_depth < 1:
            return moves, False

        #print(moves)

        crit = policy.most_critical_pos(consider_threat_sequences=False) 
        #print("critical:" + str(crit)) 
        if crit and crit.off_def == -1: # must defend, threat sequence is over
            #print(moves)
            #print("critical:" + str(crit)) 
            return moves, False

        sampler = policy.suggest_from_score(max_width, 0, 2.0)
        for c in sampler.choices:
            #print("checking move: " + str(c))
            x,y = gt.m2b((c[1][0],c[1][1]), board.N)

            if self.is_threat(policy, x,y):
                board.set(x,y)
                moves.append((x,y))
                defense0 = policy.suggest()

                if defense0.status == -1: # The opponent gave up
                    return moves, True     
                else: 
                    #print(board.stones)
                    #print("defense:" + str(defense0))
                    #print(policy.defense_options(defense0.x, defense0.y))

                    branches = []
                    for defense in policy.defense_options(defense0.x, defense0.y):
                        # A single successful defense would make this branch useless

                        p = deepcopy(policy)
                        b = p.board
                        m = deepcopy(moves)
                        b.set(defense[0], defense[1])
                        m.append((defense[0], defense[1]))
                        branches.append(self._is_tseq_won(b, p, max_depth-1, max_width, m))

                    won = np.all([br[1] for br in branches])

                    if not won:
                        board.undo()
                        moves = moves[:-1]
                    else:
                        # all branches are successful. Return any.
                        return branches[0]

        return moves, False
    
    def is_tseq_threat(self, board, max_depth=None, max_width=None):
        
        max_depth = max_depth or self.max_depth
        max_width = max_width or self.max_width        

        board = deepcopy(board)
        x,y = least_significant_move(board)
        board.set(x,y)
        moves, won = self.is_tseq_won(board, max_depth, max_width)
        board.undo()
        if won:
            return moves
        else:
            return []