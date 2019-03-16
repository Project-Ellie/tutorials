import numpy as np
import pandas as pd
from GomokuBoard import GomokuBoard
#from LineScores import LineScoresHelper
#from HeuristicScore import HeuristicScore, HeuristicScore2
from GomokuTools2 import NH9x9, Heuristics, LineScoresHelper

class FastGomokuBoard(GomokuBoard):
    
    def __init__(self, size=15, disp_width=6, stones=[]):
        """
        Fast variant of GomokuBoard. Uses vectorized computations and precomputed scores.

        Args:
            size: The number of rows and columns
            disp_width: the visual width for pyplot
            stones: A list of (int, int) pairs representing subsequent legal moves - black first.
            lsh: LineScoresHelper instance with pre-computed scores.
            
        """
        self.lsh = LineScoresHelper()
        self.size = size
        self.current_color=0

        self.board=[ # impacts of all stones on the board
            np.zeros([size,size], dtype=np.int32), # black
            np.zeros([size,size], dtype=np.int32), # white
            np.zeros([size,size], dtype=np.int32)] # borders
        self.impact9x9=[
            [ 
                0x1 << c if r == 4 and c<4 
                else 0x1 << (c-1) if c>4 and r==4

                else 0x100 << c if c == 8-r and c<4
                else 0x100 << (c-1) if c == 8-r and c>4

                else 0x10000 << 8-r-1 if r<4 and c == 4  
                else 0x10000 << 8-r if r>4 and c==4

                else 0x1000000 << 8-c-1 if c == r and c<4
                else 0x1000000 << 8-c if c == r and c>4

                else 0
                 for c in range(9) 
            ] for r in range(9) 
        ]
        
        # for each position, what impact would a stone have from here.
        self.impacts = [[self.impact_from(r,c) 
                        for c in range(size)] 
                       for r in range(size)]        
        
        self.compute_edges()
        
        GomokuBoard.__init__(self,size, disp_width, stones)

        #self.init_constants()
        #self.set_all(stones)

        

    def impact_from(self, r,c):
        """
        Construct a complete nxn impact representation of a stone at row=r, col=c
        """
        src=np.hstack([
            np.zeros((self.size+8,c),dtype=np.int),
            np.vstack([
                np.zeros((r,9), dtype=np.int32), 
                self.impact9x9, 
                np.zeros((self.size-1-r,9), dtype=np.int32)
            ]),
            np.zeros((self.size+8,self.size-1-c),dtype=np.int)
        ])
        return (src[4:-4].T[4:-4].T).copy()

    
    def impact_from_edges(self, r,c):
        """
        Construct a complete nxn impact representation of an edge stone
        That means stones with any of r=-1 or r=size or c=-1 or c=size.
        These edge stone naturally model the "defensive" impact of the edges.
        """
        src=np.hstack([
            np.zeros((self.size+10,c+1),dtype=np.int),
            np.vstack([
                np.zeros((r+1,9), dtype=np.int32), 
                self.impact9x9, 
                np.zeros((self.size-1-r+1,9), dtype=np.int32)
            ]),
            np.zeros((self.size+10,self.size-1-c+1),dtype=np.int)
        ])
        return (src[5:-5].T[5:-5].T).copy()

    
    def compute_edges(self):
        borders_sn=[self.impact_from_edges(r,c) 
                    for r in [-1, self.size] 
                    for c in range(-1, self.size+1)]
        borders_ew=[self.impact_from_edges(r,c) 
                    for c in [-1, self.size] 
                    for r in range(-1, self.size+1)]

        for impact in borders_sn + borders_ew:
            self.board[2] |= impact        
    
    
    def m2b(self, m):
        """matrix index to board position"""
        r, c = m
        return np.array([self.size-r, c+1])

    def b2m(self, p):
        """board position to matrix index"""
        x, y = p
        return np.array([self.size-y, x-1])

    def next(self):
        self.current_color = 1 - self.current_color
        return 1 - self.current_color
    
    
    def color_index(self, c):
        return 0 if c=='b' else 1
    
    
    def comp_ns(self, color, x, y, action):
        r, c = self.b2m((x,y))
        if action == 'r':
            self.board[color] |= self.impacts[r][c]
        elif action == 'u':
            self.board[color] ^= self.impacts[r][c]
        
    def compute_soft_values(self, x, y):
        raise NotImplementedError()
        
    def getn9x9(self, x,y, viewpoint=0):
        return NH9x9(*self.getnh(x,y))
        
    def getnh(self, x,y):
        r, c = self.b2m((x,y))        
        return (self.board[0][r][c], 
                self.board[1][r][c],
                self.board[2][r][c]) 
        
    def get_counts_and_scores(self, c, x, y):
        b, w = self.getnh(x,y)
        #friend, foe = (b, w) if c == 0 else (w, b)
        
            # boundaries are represented by imaginary defensive stones
        mask = self.boundary_mask(x, y)
        if c==0:
            w |= mask
        else:
            b |= mask

        
        lines = [
            [(nh & (0xFF << 8*direction)) >> 8*direction for nh in (b, w)] 
            for direction in range(4)]

        cscores_and_scores = [self.lsh.lookup_score(line, c) for line in lines]
        return [[cas[i] for cas in cscores_and_scores] for i in range(2)]

    #def get_scores(self, c, x, y):
    #    cns_o = self.get_counts_and_scores(c, x, y)
    #    tso = self.lsh.heuristics.euclidean_sum(cns_o[1])
    #    cns_d = self.get_counts_and_scores(1-c, x, y)
    #    tsd = self.lsh.heuristics.euclidean_sum(cns_d[1])
    #    return tso, tsd
    
    def boundary_mask(self, x, y):
        mask = 0
        edges = self.all_edges(x,y)
        for d in range(4):
            l, r = edges[d]
            rhs = 0 if r is None else (0x01 << (8 * d + 7 - r)) 
            lhs = 0 if l is None else (0x01 << (8 * d + 7 - l))
            mask |= rhs | lhs
        return mask
        
    @staticmethod        
    def from_csv(filename, size=19, disp_width=10):
        stones = pd.read_csv(filename, header=None).values.tolist()
        return FastGomokuBoard( size, disp_width, stones=stones)

    def collect_scores(self):
        h2 = self.lsh.heuristics
        scores = []
        for x in range(1, self.size+1):
            for y in range(1, self.size+1):
                if (x,y) not in self.stones[:self.cursor+1]:
                    all_edges = self.all_edges(x,y)
                    nh = self.getn9x9(x, y)
                        
                    b_score, w_score = h2.classify_nh(nh, score_for=0, all_edges=all_edges)

                    h, l, c = b_score
                    soft_black=16*(16*(16*(6-c)+h)+l)+8
                    h, l, c = w_score
                    soft_white=16*(16*(16*(6-c)+h)+l)
                        
                    scores.append(['b', (x,y), b_score, soft_black])
                    scores.append(['w', (x,y), w_score, soft_white])
        return [score for score in scores if score [2][0] > 0]

    
    def top(self, n=0):
        n == self.size * self.size if n==0 else n
        from operator import itemgetter
        scores=self.collect_scores()
                        
        topn=sorted(scores, key=itemgetter(3))[-n:]
        topn.reverse()
        return topn
