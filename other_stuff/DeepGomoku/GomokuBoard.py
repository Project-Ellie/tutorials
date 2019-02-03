import numpy as np
import matplotlib.pyplot as plt
from GomokuTools import GomokuTools, N_9x9
from HeuristicScore import HeuristicScore

class GomokuBoard:
    def __init__(self, size, disp_width, stones=[], heuristics=HeuristicScore()):
        self.bias = 1.3
        self.size=size
        self.side=disp_width
        self.stones=[]
        self.cursor = -1
        self.heuristics = heuristics
        self.next_party='b'
        self.stats =  {'b': [], 'w': []}
        self.ns = [[N_9x9() for i in range(self.size)] for j in range(self.size)]
        #c = 'w'
        self.set_all(stones)
        self.color_scheme = [ # visualize the offensive/defensive score
            ['#F0F0F0', '#FFC0C0', '#FF9090', '#FF6060', '#FF0000'],
            ['#A0FFA0', '#E8D088', '#FFA080', '#F86040', '#F01808'],
            ['#00FF00', '#B0D818', '#EFB060', '#F07040', '#E03010'],
            ['#00CF00', '#80B014', '#C0A048', '#E08050', '#D04820'],
            ['#00A000', '#307810', '#607020', '#907828', '#C06030']
        ]
     
    
    def color_for(self, offensive, defensive):
        o = (offensive - self.bias) * 5 / (5-offensive)
        d = (defensive - self.bias) * 5 / (5-defensive)
        o = max(0, min(4, o))
        d = max(0, min(4, d))
        return self.color_scheme[int(o)][int(d)]
    
    def display(self, score='current', stones=None):
        stones = stones or self.stones
        side=self.side
        size=self.size
        fig, axis = plt.subplots(figsize=(side, side))
        axis.set_xlim([0, size+1])
        axis.set_ylim([0, size+1])
        plt.xticks(range(1,size+1),['A','B','C','D','E','F','G','H',
                         'I','K','L','M','N','O','P', 'Q', 'R', 'S', 'T', 'U'][:size+1])
        plt.yticks(range(1,size+1))
        axis.set_facecolor('#8080FF')
        xlines = [[ [1, size], [y,y], '#E0E0E0'] for y in range(1, size+1)]
        ylines = [[ [x,x], [1, size], '#E0E0E0'] for x in range(1, size+1)]
        ylines = np.reshape(xlines + ylines, [-1])
        axis.plot(*ylines)
        self.display_helpers(axis)
        if self.cursor >= 0:
            self.display_stones(stones, axis)
        if score:
            self.display_score(axis, score)

    def display_helpers(self, axis):
        if self.size==15:
            axis.scatter([4, 4, 12, 12, 8], [4, 12, 12, 4, 8], s=self.side**2, c='#E0E0E0')
        elif self.size==20:
            axis.scatter([6, 6, 15, 15], [6, 15, 15, 6], s=self.side**2, c='#E0E0E0')

            
    def display_cursor(self):
        x,y = self.stones[self.cursor]
        box = np.array(
            [[-0.6,  0.6,  0.6, -0.6, -0.6],
            [-0.6, -0.6,  0.6,  0.6, -0.6]])
        box = box + [[x], [y]]
        plt.plot(*box, color='w', zorder=30)
            
            
    def display_stones(self, stones, axis):
        colors=['white', 'black']
        for i in range(1, self.cursor + 2):
            x,y = self.stones[i-1][0:2]
            stc = colors[i % 2]
            fgc = colors[1 - i % 2]
            axis.scatter([x],[y], c=stc, s=self.stones_size(), zorder=10);
            self.display_cursor()
            plt.text(x, y, i, color=fgc, fontsize=12, zorder=20,
                     horizontalalignment='center', verticalalignment='center');

            
    def stones_size(self):
        return 150 / self.size * self.side**2
        

    def display_score(self, axis, score):
        for x in range(1, self.size+1):
            for y in range(1, self.size+1):

                c = self.next_party if score == 'current' else score
                tso, tsd = self.get_scores(c, x, y)
                
                if (tsd > self.bias or tso > self.bias): #and (x,y) not in self.stones:
                    c = self.color_for(offensive=tso, defensive=tsd)
                    axis.scatter([x],[y], color=c, s=2*self.side**2, zorder=5)
        
        
    def set_all(self, stones):
        for stone in stones:
            self.set(*stone)
        
        
    def ctoggle(self):
        """
        toggle current color
        """
        self.next_party = 'b' if self.next_party == 'w' else 'w'
        return self.next_party
        
            
    def set(self, x,y):
        """
        x,y: 1-based indices of the board
        """
        if self.cursor != len(self.stones)-1:
            raise(ValueError("Cursor not at end position."))
        if (x,y) in self.stones:
            raise(ValueError("Not a valid move. Position is occupied."))
        if not self._is_valid((self.size-y, x-1)):
            raise(ValueError("Not a valid move. Beyond board boundary."))
        self.stones.append((x,y))
        
        c = self.next_party
        c_next = self.ctoggle()
        self.cursor = len(self.stones)-1
        self.comp_ns(c, x, y, 'r')
        self.add_stats(c_next)
        
        return self
            
    def undo(self):
        if self.cursor != len(self.stones)-1:
            raise(ValueError("Cursor not at end position."))

        c = self.ctoggle()
        stone = self.stones[-1]
        self.stones = self.stones[:-1]
        self.cursor = len(self.stones)-1
        self.comp_ns(c, *stone, action='u')
        self.stats[c] = self.stats[c][:-1]
        return self
            
    def fwd(self, n=1):
        if ( n > 1 ):
            self.fwd()
            self.fwd(n-1)
            return self
        if self.cursor < len(self.stones)-1:
            self.cursor += 1
            c = self.next_party
            self.ctoggle()
            self.comp_ns(c, *self.stones[self.cursor], action='r')
        return self
            
    def bwd(self, n=1):
        if ( n > 1 ):
            self.bwd()
            self.bwd(n-1)
            return self
            
        if self.cursor >= 0:
            stone = self.stones[self.cursor]
            self.cursor -= 1
            c = self.ctoggle()
            self.comp_ns(c, *stone, action='u')
        return self
            
            
    def getn9x9(self, x,y):
        """
        x,y: 1-based indices of the board
        """
        return self.ns[self.size-y][x-1]

    def comp_ns(self, color, x, y, action='r'):
        """
        compute neighbourhoods for the given move in board coordinates
        x,y: 1-based indices of the board
        """
        rc = GomokuTools.b2m((x,y), size=self.size)  # row, col
        for dd in GomokuTools.dirs().items():
            step = np.array(dd[1][1])
            for d in range(1,5):
                
                rc_n = rc + d * step
                
                if self._is_valid(rc_n):
                    n_ = self.ns[rc_n[0]][rc_n[1]]
                    if action == 'r':
                        n_.register(color, GomokuTools.opposite(dd[0]), d)
                    else:
                        n_.unregister(color, GomokuTools.opposite(dd[0]), d)
        
    
    def _is_valid(self, index):
        """
        checks the array indexes (not the board coordinates!)
        """
        return index[0] >= 0 and index[0] < self.size and index[1] >= 0 and index[1] < self.size
    
 
    def get_scores(self, c, x, y):
        h = self.heuristics
        n = self.getn9x9(x,y)                                
        fof = 0 if c=='b' else 1
        tso = h.total_score(n.as_bits(), fof=fof)
        tsd = h.total_score(n.as_bits(), fof=1-fof)
        return tso, tsd               
           
    
    def calc_stats(self, c):
        N = len(self.stones)
        scores = [self.get_scores(c, x, y) 
                  for x in range(1, self.size+1)
                  for y in range(1, self.size+1)]
        stats = { 
            'avg_o': sum([s[0] for s in scores]) / N,
            'gsum_o': 0, #np.sqrt(sum([s[0]**2 for s in scores])),
            'max_o': max([s[0] for s in scores]),
            'avg_d': sum([s[1] for s in scores]) / N,
            'gsum_d': 0,# np.sqrt(sum([s[1]**2 for s in scores])),
            'max_d': max([s[1] for s in scores]),
        }
        return stats
    
    
    def add_stats(self, c):
        self.stats[c].append(self.calc_stats(c))