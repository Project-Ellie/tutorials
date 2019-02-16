import numpy as np


class GomokuTools:
    
    @staticmethod
    def dirs():
        return {
            'e' : (0, [0, 1]),
            'ne': (1, [-1, 1]),
            'n' : (2, [-1, 0]),
            'nw': (3, [-1, -1]),
            'w' : (4, [0, -1]),
            'sw': (5, [1, -1]),
            's' : (6, [1, 0]),
            'se': (7, [1, 1])}
    
    @staticmethod
    def as_bit_array(n):
        """
        Returns an array of int 0 or 1 
        """
        assert(n >= 0 and n <= 255)
        return [np.sign(n & (1<<i)) for i in range(7, -1, -1)]

    @staticmethod
    def line_for_xo(xo_string):
        """
        return a 2x8 int array representing the 'x..o..' xo_string 
        """
        return [[1 if (ch=='x' and c==0) 
                 or (ch=='o' and c==1) 
                 else 0 for ch in xo_string] for c in [0,1]]




class NH9x9:
    
    """
    9-by-9 neighbourhood of an empty Gomoku position. Provides 8 bytes (2 int32) representing 
    what is visibile from that field in a particular direction as input for a valuation function.
    Example: the six stones seen from '*' in south-east/north-west:

    - - - - - - - - x
    - - - - - - - x - 
    - - - - - - - - - 
    - - - - - x - - -
    - - - - * - - - - 
    - - - - - - - - - 
    - - o - - - - - - 
    - o - - - - - - -
    o - - - - - - - - 
    
    will be represented by the following bytes encoded as two int32

    Black:
     e : 0 0 0 0   0 0 0 0
    ne : 0 0 0 0   1 0 1 1
     n : 0 0 0 0   0 0 0 0
    nw : 0 0 0 0   0 0 0 0

    White:
     e : 0 0 0 0   0 0 0 0
    ne : 1 1 1 0   0 0 0 0
     n : 0 0 0 0   0 0 0 0
    nw : 0 0 0 0   0 0 0 0
    
    """
    def __init__(self, b=0, w=0):
        self.visible = [b, w]
        
    def set(self, b, w):
        self.visible = [b, w]

    def get_line(self, direction, color=0):
        """
        Return two arrays of 8 integers representing black and white stones on a line
        of length 9. The middle position is not represented in the array
        Args:
            direction: either one of 'e', 'ne', 'n', 'nw' or their integer representations
        """
        d = GomokuTools.dirs()[direction][0] if type(direction) == str else direction
        return [ GomokuTools.as_bit_array((self.visible[c] & (0xFF << (8 * d))) >> (8 * d)) 
                for c in [color, 1-color]]

    
    def as_bits(self):
        return [self.get_line(d) for d in range(4)]

    
    def setline_xo (self, direction, string_xo):
        """
        Set stones in direction using a string with 'x' and 'o' for black and white
        Example: setline_xo(3, 'x..oo.xx') sets the given stones in nw direction
        """
        TWO_N = np.array([128, 64, 32, 16, 8, 4, 2, 1])
        line = GomokuTools.line_for_xo(string_xo)        
        b = self.visible[0]
        w = self.visible[1]
        mask = (~(0xFF << 8 * direction)) & 0xFFFFFFFF
        b &= mask
        w &= mask
        new_bytes=[np.sum(TWO_N * line[i]) << (8*direction) for i in [0,1]]
        b |= new_bytes[0]
        w |= new_bytes[1]
        self.visible = [b, w]
        return self

    
    def __repr__(self):
        field = [[' ' for i in range(9)] for j in range(9)]
        field[4][4]='*'
        for h in list(GomokuTools.dirs().items())[:4]:               
            step = GomokuTools.dirs()[h[0]][1]
            pos0 = np.array([4,4]) - 4 * np.array(step)
            bits = self.get_line(h[0])
            for x in range(8):
                row, col = pos0 + (x + x//4) *np.array(step)
                field[row][col]='x' if bits[0][x] == 1 else 'o' if bits[1][x] == 1 else ' '                
        return "\n".join([('|' + ' '.join(field[r]) + '|') for r in range(9)])

    
    
    
    
class Heuristics:
    
    def __init__(self):

        # map cscore to threat value
        self.c2t={
            (1,1): 1,
            (1,2): 2,
            (2,1): 3,
            (2,2): 4,
            (3,1): 5,
            (3,2): 6,
            (4,1): 8,
            (4,2): 9
        }
        
        self.color_scheme = [ # visualize the offensive/defensive score
            ['#F0F0F0', '#FFC0C0', '#FF9090', '#FF6060', '#FF0000'],
            ['#A0FFA0', '#E8D088', '#FFA080', '#F86040', '#F01808'],
            ['#00FF00', '#B0D818', '#EFB060', '#F07040', '#E03010'],
            ['#00CF00', '#80B014', '#C0A048', '#E08050', '#D04820'],
            ['#00A000', '#307810', '#607020', '#907828', '#C06030']
        ]


        
    def criticality(self, h, l):
        if h == 9: 
            return ('lost', 1)
        elif h == 8:
            return ('move or lose in 1', 2)
        elif h == 7: 
            return ('move or lose in 2', 3)
        elif (h, l) in [(5,5), (5,4), (6,5), (6,4), (6,6)]:
            return ('move or lose in 2', 4)
        elif (h, l) == (4,4):
            return ('move or lose in 3', 5)
        else:
            return ('defendable', 6)
            
    
    def classify(self, b, w, edges=(None, None)):
        """
        Computes a criticality score for the neighbourhood represented by the two int32 
        b for black and w for white stones
        
        Returns:
            A criticality score: that's two triples of ints, one for black and the other for white.
            The triple consists of the largest and the second-larges single-line treats, and the 
            total criticality, a number between 1 and 6, 1 for immediate loss and 6 for defendable.
        """
        return self.classify_nh(NH9x9(b, w), edges=edges)    
            
    def classify_nh(self, nh, edges=(None, None)):
        res = []
        for color in [0, 1]:
            classes=[self.classify_line(nh.get_line(direction, color), edges) for direction in range(4)]
            
            l, h = sorted(classes)[-2:]
            c = self.criticality(h, l)
            res.append((h, l, c[1]))
        return res
    
                        
    def classify_line(self, line, edges=(None, None)):
        cscore = self.cscore(line=line, cap=2, edges=edges)
        return 0 if cscore[0] == 0 else self.c2t[cscore]
    
                        
    def f_range(self, line, c=0, edges=(None, None)):
        """
        The largest adversary-free range within a given line
        
        Args:
            line: 8x2 integer array that represents the stones
            c:    0 to look at black, 1 to consider white
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

    
    def cscore(self, line, c=0, edges=(None, None), cap=2):
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
        c_ = min(c_,cap)
        return (m, c_)

    
    def color_for_triple(self, h, l, c):
        """
        """
        if c <= 2:
            return 4
        elif c <= 5:
            return 6-c
        elif h == 4:
            return 0
        else:
            return None
    
    def threat_color(self, offensive, defensive):
        """
        return appropriate color for given pair of threat triples
        """
        o, d = [self.color_for_triple(*triple) for triple in [offensive, defensive]]
        if o is None and d is None:
            return None
        o, d = 0 if o is None else o, 0 if d is None else d
        return self.color_scheme[int(o)][int(d)]
        


