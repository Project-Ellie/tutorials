import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GomokuTools2 import GomokuTools, NH9x9, Heuristics
#from HeuristicScore import HeuristicScore

class GomokuBoard:
    def __init__(self, size, disp_width, stones=[], stats=False):
        self.size=size
        self.side=disp_width
        self.stones=[]
        self.heuristics = Heuristics()
        self.current_color = 0 
        self.ns = [[NH9x9() for i in range(self.size)] for j in range(self.size)]
        self.init_constants()
        self.set_all(stones, stats)
        
    def init_constants(self):
        self.bias = 1.3
        self.cursor = -1
        self.stats =  [[],[]]
        self.color_scheme = [ # visualize the offensive/defensive score
            ['#F0F0F0', '#FFC0C0', '#FF9090', '#FF6060', '#FF0000'],
            ['#A0FFA0', '#E8D088', '#FFA080', '#F86040', '#F01808'],
            ['#00FF00', '#B0D818', '#EFB060', '#F07040', '#E03010'],
            ['#00CF00', '#80B014', '#C0A048', '#E08050', '#D04820'],
            ['#00A000', '#307810', '#607020', '#907828', '#C06030']
        ]
        
    
    def color_for(self, offensive, defensive):
        o = (offensive - self.bias) * 5 / max((5-offensive),0.01)
        d = (defensive - self.bias) * 5 / max((5-defensive),0.01)
        o = max(0, min(4, o))
        d = max(0, min(4, d))
        return self.color_scheme[int(o)][int(d)]
    
    def display(self, score='current'):
        side=self.side
        size=self.size
        fig, axis = plt.subplots(figsize=(side, side))
        axis.set_xlim([0, size+1])
        axis.set_ylim([0, size+1])
        plt.xticks(range(1,size+1),['A','B','C','D','E','F','G','H',
                         'I','J','K','L','M','N','O','P', 'Q', 'R', 'S', 'T', 'U'][:size+1])
        plt.yticks(range(1,size+1))
        axis.set_facecolor('#8080FF')
        xlines = [[ [1, size], [y,y], '#E0E0E0'] for y in range(1, size+1)]
        ylines = [[ [x,x], [1, size], '#E0E0E0'] for x in range(1, size+1)]
        ylines = np.reshape(xlines + ylines, [-1])
        axis.plot(*ylines)
        self.display_helpers(axis)
        if self.cursor >= 0:
            self.display_stones(self.stones, axis)

        if score is not None:
            if score=='current':
                score=self.current_color
            self.display_score(axis, score)

    def display_helpers(self, axis):
        if self.size==15:
            axis.scatter([4, 4, 12, 12, 8], [4, 12, 12, 4, 8], s=self.side**2, c='#E0E0E0')
        elif self.size==19:
            axis.scatter([4, 4, 4, 10, 16, 16, 16, 10, 10], [4, 10, 16, 16, 16, 10, 4, 4, 10], s=self.side**2, c='#E0E0E0')
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

                
    def display_best(self, axis):        
        top1 = self.top(1)
        otop = top1[0][0][0]
        dtop = top1[1][0][0]
        odtop = top1[2][0][0]
        ocircle = plt.Circle(otop, .4, color='#00ff00', lw=2, fill=False)
        dcircle = plt.Circle(dtop, .4, color='#ff0000', lw=2, fill=False)
        odcircle = plt.Circle(odtop, .4, color='#ffff00', lw=2, fill=False)
        axis.add_artist(ocircle)
        axis.add_artist(dcircle)
        axis.add_artist(odcircle)

        
    def stones_size(self):
        return 120 / self.size * self.side**2
        

    def display_score(self, axis, score):  
        for x in range(1, self.size+1):
            for y in range(1, self.size+1):
                all_edges = self.all_edges(x,y)
                if score is None:
                    score = self.current_color
                nh = self.getn9x9(x,y)

                classification = self.heuristics.classify_nh(nh, all_edges, score_for=score)
                
                color = self.heuristics.threat_color(*classification)
                if color is not None:
                    axis.scatter([x],[y], color=color, s=2*self.side**2, zorder=5)
        
        #self.display_best(axis)
        
        
    def display_score_old(self, axis, score):  
        for x in range(1, self.size+1):
            for y in range(1, self.size+1):

                c = self.current_color if score == -1 else score

                tso, tsd = self.get_scores(c, x, y)

                if (tsd > self.bias or tso > self.bias): #and (x,y) not in self.stones:
                    
                    c = self.color_for(offensive=tso, defensive=tsd)
                    axis.scatter([x],[y], color=c, s=2*self.side**2, zorder=5)
        
        #self.display_best(axis)
        
        
        
    def set_all(self, stones, stats=False):
        for stone in stones:
            self.set(*stone, stats=stats)
        
        
    def ctoggle(self):
        """
        toggle current color
        """
        self.current_color = 1 - self.current_color
        return self.current_color
        
    def maybe_convert(self,x):
        if type(x) != str:
            return x
        return ord(x)-64        
            
            
    def set(self, x,y, stats=False):
        """
        x,y: 1-based indices of the board, x may be an uppercase letter
        """
        x = self.maybe_convert(x)

        if self.cursor != len(self.stones)-1:
            raise(ValueError("Cursor not at end position."))
        if (x,y) in self.stones:
            raise(ValueError("Position (%s, %s) is occupied." % (x,y)))
        if not self._is_valid((self.size-y, x-1)):
            raise(ValueError("Not a valid move. Beyond board boundary."))
        
        self.stones.append((x,y))
        
        c = self.current_color
        c_next = self.ctoggle()

        self.cursor = len(self.stones)-1
        
        self.comp_ns(c, x, y, 'r')
        if stats:
            self.add_stats(c_next)
        
        return self
            
    def undo(self, stats=False):
        if self.cursor != len(self.stones)-1:
            raise(ValueError("Cursor not at end position."))

        c = self.ctoggle()
        stone = self.stones[-1]
        self.stones = self.stones[:-1]
        self.cursor = len(self.stones)-1
        self.comp_ns(c, *stone, action='u')
        if stats:
            self.stats[c] = self.stats[c][:-1]
        return self
            
    def fwd(self, n=1):
        if ( n > 1 ):
            self.fwd()
            self.fwd(n-1)
            return self
        if self.cursor < len(self.stones)-1:
            self.cursor += 1
            c = self.current_color
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
                    odir = (dd[1][0] + 4) % 8 # the opposite direction
                    if action == 'r':
                        n_.register(color, odir, d)
                    else:
                        n_.unregister(color, odir, d)
        
    
    def _is_valid(self, index):
        """
        checks the array indexes (not the board coordinates!)
        """
        return index[0] >= 0 and index[0] < self.size and index[1] >= 0 and index[1] < self.size
    
 
    def get_scores(self, c, x, y):
        """
        Compute the scores from c's point of view
        Returns:
        A pair (offensive score, defensive score)
        """
        all_edges = self.all_edges(x,y)

        h = self.heuristics
        n = self.getn9x9(x,y)                                
        tso = h.total_score(n.as_bits(), c, all_edges=all_edges)
        tsd = h.total_score(n.as_bits(), c=1-c, all_edges=all_edges)
        return tso, tsd               

    def top(self, n):
        from operator import itemgetter
        o_scores=[]
        d_scores=[]
        od_scores=[]
        for x in range(1, self.size+1):
                for y in range(1, self.size+1):
                    if (x,y) not in self.stones[:self.cursor+1]:
                        score = self.get_scores(c=self.current_color, x=x, y=y)
                        o_scores.append([(x,y), score[0]])
                        d_scores.append([(x,y), score[1]])
                        od_scores.append([(x,y), score[0] + score[1]])
                        
        otopn=sorted(o_scores, key=itemgetter(1))[-n:]
        dtopn=sorted(d_scores, key=itemgetter(1))[-n:]    
        odtopn=sorted(od_scores, key=itemgetter(1))[-n:]    
        otopn.reverse()
        dtopn.reverse()
        odtopn.reverse()
        return otopn, dtopn, odtopn
    
    
    def edges(self, p):
        N = self.size
        return (4-p, None) if p<5 else (None, N+4-p) if p > N-4 else (None,None)
    
    def emax(self, x1, x2):
        return x1 if x2 is None else x2 if x1 is None else max(x1, x2)

    def emin(self, x1, x2):
        return x1 if x2 is None else x2 if x1 is None else min(x1, x2)    
    
    def ne_edges(self, x, y):
        e_x = self.edges(x)
        e_y = self.edges(y)
        return self.emax(e_x[0], e_y[0]), self.emin(e_x[1], e_y[1])    

    def nw_edges(self, x,y):
        e_x = self.edges(self.size+1-x)
        e_y = self.edges(y)
        return self.emax(e_x[0], e_y[0]), self.emin(e_x[1], e_y[1])    

    def all_edges(self, x, y):
        return [
            self.edges(x), 
            self.ne_edges(x,y), 
            self.edges(y), 
            self.nw_edges(x,y)]    
    
    
    def calc_stats(self, c):
        raise ValueError("Not implemented")
    
    def add_stats(self, c):
        raise ValueError("Not implemented")
        
    
    def save(self, filename):
        df = pd.DataFrame(self.stones)
        df.to_csv(filename, header=None, index=None)
        
