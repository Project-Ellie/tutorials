import numpy as np


class GomokuTools:

    
    @staticmethod    
    def str_base(number, base, width=8):
        def _str_base(number,base):
            (d, m) = divmod(number, base)
            if d > 0:
                return _str_base(d, base) + str(m)
            return str(m)
        s = _str_base(number, base)
        return '0'*(width-len(s))+s

    
    @staticmethod    
    def base2_to_xo(number):
        return GomokuTools.str_base(number, 3).replace('2', 'o').replace('1', 'x').replace('0', '.')    

    
    @staticmethod    
    def mask(offensive, defensive):
        n = defensive
        l = n & 0xF0
        l = (l | l<<1 | l<<2 | l<<3) & 0xF0

        r = n & 0x0F
        r = (r | r>>1 | r>>2 | r>>3) & 0x0F

        mask=(~(l | r)) & 0xFF
        free_stones=mask & offensive

        return free_stones, mask


    @staticmethod    
    def num_offensive(o, d):
        s, l, offset = GomokuTools.mask2(o, d)
        m2o_bits = GomokuTools.as_bit_array(s)[:l]
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
            max_count += 1
        if m2o_bits[-1] == 0:
            max_count += 1

        # Criticality correction for the fatal double-open 3
        if max_count == 8:
            max_count=13
        return max_count        
    
    
    @staticmethod    
    def mask2(offensive, defensive):
        n = defensive
        l = n & 0xF0
        l = (l | l<<1 | l<<2 | l<<3) & 0xF0

        r = n & 0x0F
        r = (r | r>>1 | r>>2 | r>>3) & 0x0F

        mask=(~(l | r))
        free_stones=mask & offensive

        free_length=np.sum([(mask>>i)&1 for i in range(8)], axis=0)
        l_offset = np.sum([(l>>i)&1 for i in range(8)], axis=0)
        #free_length = (free_length > 5) * 5 + (free_length <= 5) * free_length
        return free_stones << l_offset, free_length, l_offset    

    
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
    def int_for_dir(dirstr):
        return GomokuTools.dirs()[dirstr][0]
    
    @staticmethod
    def as_bit(direction, distance):
        return (1<<(distance+3) if direction//4 else 1<<(4-distance))# << (8*(direction%4))
        
    @staticmethod
    def m2b(m, size):
        """matrix index to board position"""
        r, c = m
        return np.array([c+1, size-r])

    @staticmethod
    def b2m(p, size):
        """board position to matrix index"""
        x, y = p
        return np.array([size-y, x-1])
        
    
        
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
        powers=np.array([2**i for i in range(7, -1, -1)])
        return [sum([1 if (ch=='x' and c==0) 
                 or (ch=='o' and c==1) 
                 else 0 for ch in xo_string] * powers) for c in [0,1]]



class NH9x9:
    
    """
    9-by-9 neighbourhood of an empty Gomoku position. Provides 12 8 bit integers representing 
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
    def __init__(self, black=[0,0,0,0], white=[0,0,0,0], edges=[0,0,0,0]):
        for n in [black, white, edges]:
            for i in range(4):
                assert(n[i] >= 0 and n[i] < 256)
        self.b = black
        self.w = white
        self.e = edges
        
        
    def register(self, color, direction, distance):
        assert(color==0 or color==1)
        assert(direction>=0 and direction<8)
        assert(distance>=1 and distance <=4)
        
        if color==0:
            self.b[direction%4] |= GomokuTools.as_bit(direction, distance)
        else:
            self.w[direction%4] |= GomokuTools.as_bit(direction, distance)
        return self
        
    def unregister(self, color, direction, distance):
        assert(color==0 or color==1)
        assert(direction>=0 and direction<8)
        assert(distance>=1 and distance <=4)
        
        bit = GomokuTools.as_bit(direction, distance)
        if color==0:
            self.b[direction] &= (0xFF ^ bit)
        else:
            self.w[direction] &= (0xFF ^ bit)
        return self

    
    def get_line(self, direction):
        """
        Return two arrays of 8 integers representing black and white stones on a line
        of length 9. The middle position is not represented in the array
        Args:
            direction: either one of 'e', 'ne', 'n', 'nw' or their integer representations
        """
        assert(direction>=0 and direction <= 3)
        
        return [
            GomokuTools.as_bit_array(self.b[direction]),
            GomokuTools.as_bit_array(self.w[direction]),
            GomokuTools.as_bit_array(self.e[direction])]
    

    def as_bits(self):
        return [self.get_line(d) for d in range(4)]

    
    def __repr__(self):
        dirs=[[0, 1], [-1, 1], [-1, 0], [-1, -1]]
        field = [[' ' for i in range(9)] for j in range(9)]
        field[4][4]='*'
        for h in range(4):               
            step = dirs[h]
            pos0 = np.array([4,4]) - 4 * np.array(step)
            bits = self.get_line(h)
            for x in range(8):
                row, col = pos0 + (x + x//4) * np.array(step)
                field[row][col]='x' if bits[0][x] == 1 \
                    else 'o' if bits[1][x] == 1 \
                    else '+' if bits[2][x] == 1 \
                    else ' '                
        return "\n".join([('|' + ' '.join(field[r]) + '|') for r in range(9)])

    
    
    
    
class Heuristics:
    
    def __init__(self):

        self.color_scheme = [ # visualize the offensive/defensive score
            ['#F0F0F0', '#FFC0C0', '#FF9090', '#FF6060', '#FF0000'],
            ['#A0FFA0', '#E8D088', '#FFA080', '#F86040', '#F01808'],
            ['#00FF00', '#B0D818', '#EFB060', '#F07040', '#E03010'],
            ['#00CF00', '#80B014', '#C0A048', '#E08050', '#D04820'],
            ['#00A000', '#307810', '#607020', '#907828', '#C06030']
        ]
        # map cscore to threat value
        self.c2t={
            (1,1): 1,
            (1,2): 2,
            (2,1): 3,
            (2,2): 4,
            (3,1): 5,
            (3,2): 7,
            (4,1): 8,
            (4,2): 9
        }
        
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
        gt = GomokuTools()
        self._all_scores = np.zeros(256*256, dtype=int)
        self._all_counts = np.zeros(256*256, dtype=int)
        self._all_scores_and_more = [0 for _ in range(256*256)]
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

    
    def criticality(self, h, l):
        if h == 9: 
            return ('lost', 1)
        elif h == 8:
            return ('move or lose in 1', 2)
        elif h == 7: 
            return ('move or lose in 2', 3)
        elif (h, l) in [(5,5), (5,4)]:
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
            
        
        
    def classify_nh(self, nh, all_edges=None, score_for=0):
        if all_edges==None:
            all_edges=[(None, None) for i in range(4)]
        res = []
        for color in [score_for, 1-score_for]:
            classes=[self.classify_line(nh.get_line(direction, color), all_edges[direction]) 
                     for direction in range(4)]
            
            l, h = sorted(classes)[-2:]
            c = self.criticality(h, l)
            res.append((h, l, c[1]))
        return res
    
    
    
    def soft_values(self, nh, viewpoint=0):
        classification = self.classify_nh(nh)
        values=[]
        for color in [0,1]:
            h, l, c = classification[color]
            values.append(16*(16*(16*(6-c)+h)+l)+8*(viewpoint==color))
        return values
    
    
    
    def classify_line(self, line, edges=(None, None)):
        cscore = self.cscore(line=line, cap=2, edges=edges)
        return 0 if cscore[0] == 0 else self.c2t[cscore]

    
    def describe(self, classification):
        descriptions = {
            1: "double open 4 - done.",
            2: "single open 4 - must move",
            3: "double open 3 - move or lose in 2",
            4: "threat to double open 3 - move or lose in 3",
            5: "2 double open 2s - move or lose in 3",
            6: "defendable"        
        }
        return descriptions[classification[2]] 
    
    def describe_both(self, classifications):
        return [ ("White: " if c==1 else "Black: ") + describe(classifications[c]) for c in [0,1]]
    
                        
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
        elif h == 6:
            return 1
        elif h >= 4:
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
        
        
        
        
class Reasoner():
    def __init__(self, topn, my_color):
        self.topn = topn
        self.my_color = my_color
        self.other_color = 'b' if my_color=='w' else 'w'
        
    def list_by_level(self, level_or_higher, color):
        return [s for s in self.topn if s[2][0] >= level_or_higher and s[0] == color]    
    
    def list_by_criticality(self, criticalities, color):
        return [s for s in self.topn if s[2][2] in criticalities and s[0] == color]    
    
    def i_can_win_now(self):
        options = self.list_by_level(8, self.my_color)
        return len(options) > 0, options
    
    def i_will_lose(self):
        return not self.i_can_win_now()[0] and self.two_fours_against_me()
    
    def two_fours_against_me(self):
        return (len(self.list_by_level(9, self.other_color)) > 0 or 
                len(self.list_by_level(8, self.other_color)) > 1)
        
    def is_winning_attack(self):
        options = self.list_by_criticality([5, 4, 3, 2], self.my_color)
        return options != [], options
    
    def is_urgent_defense(self):
        options = self.list_by_criticality([5, 4, 3, 2], self.other_color)
        return options != [], options
        
    def suggest(self):
        i_can, move = self.i_can_win_now()
        if i_can:
            return move
        if self.i_will_lose():
            return "Giving up"

        is_winning, move = self.is_winning_attack()
        if is_winning:
            return move
        
        is_urgent, move = self.is_urgent_defense()
        if is_urgent:
            return move
        
        return "Treesearch"
            
                    
        
        
