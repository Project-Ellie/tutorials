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
    
    def m2b(m, size=15):
        """matrix index to board position"""
        r, c = m
        return np.array([size-r, c+1])

    def b2m(p, size=15):
        """board position to matrix index"""
        x, y = p
        return np.array([size-y, x-1])
        
    
    @staticmethod
    def opposite(d):
        """
        Returns 'e' for 'w', 'sw' for 'ne' and so on.
        """
        return [i[0] for i in GomokuTools.dirs().items() if (GomokuTools.dirs()[d][0] + 4) % 8 == i[1][0]][0]

    @staticmethod    
    def as_bits(byte):
        return [np.sign(byte & (1<<i)) for i in range(7, -1, -1)]

    @staticmethod
    def as_bitarray(code):
        # l3 is the nw-line. point 2,2 on that line sees the stone at a distance of 2
        l=code
        l0=(l & 0xFF)               # east
        l1=(l & 0xFF00) >> 8        # north east
        l2=(l & 0xFF0000) >> 16     # north
        l3=(l & 0xFF000000) >> 24   # north west
        return {
            'e': GomokuTools.as_bits(l0),
            'ne': GomokuTools.as_bits(l1),
            'n': GomokuTools.as_bits(l2),
            'nw': GomokuTools.as_bits(l3)
        }


    
    
class N_9x9:
    """
    9-by-9 neighbourhood of an empty field. Provides 8 bytes representing what is 
    visibile in a particular direction as input for a valuation function.
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
    
    will be represented by the following bytes:
    
    b,  e : 0 0 0 0   0 0 0 0
    w,  e : 0 0 0 0   0 0 0 0
    b, ne : 0 0 0 0   1 0 1 1
    w, ne : 1 1 1 0   0 0 0 0
    b,  n : 0 0 0 0   0 0 0 0
    w,  n : 0 0 0 0   0 0 0 0
    b, nw : 0 0 0 0   0 0 0 0
    w, nw : 0 0 0 0   0 0 0 0
    
    """

    
    def __init__(self, ba=bytearray(8)):
        self.ba = ba.copy()

        
    @staticmethod
    def xy(color, direction, distance):
        """
        Position of the bit representing the stone described with the parameters
        
        Args:
            color:     'b' or 'w'
            direction: any of 'e', 'ne',...
            distance:  distance from the center: any of 1, 2, 3, 4
        
        Returns:
            x: the position of the representing bit within the byte
            y: the position of the byte representing color and direction, 
               counting right to left (bit 0 is right-most)
        """

        h_ = GomokuTools.dirs()[direction][0]
        c_ = color
        y=c_+2*(h_%4)
        x=4 - (h_//4) + (2*(h_//4)-1)*distance
        return x, y

    
    def register(self, color, direction, distance):
        """
        Register a stone at the given position
        
        Args:
            color:     0 for black, 1 for white
            direction: any of 'e', 'ne',...
            distance:  distance from the center: any of 1, 2, 3, 4

        Returns this object
        """
        
        x,y=N_9x9.xy(color, direction, distance)
        self.ba[y] |= (1 << x)
        return self
    
    def unregister(self, c, h, d):
        x,y=N_9x9.xy(c,h,d)
        self.ba[y] ^= (1 << x)
        return self
    
    def setline(self, h, array12):
        for i in range(8):
            s = array12[i]
            c = 1 if s==2 else 0 if s==1 else None
            if c is not None:
                h_ = h if i//4 else GomokuTools.opposite(h)
                d = i-3 if i//4 else 4-i
                self.register(c, h_, d)
        return self
    
    def bits_in_line(self, h):
        _, y = N_9x9.xy(0, h, 1)
        h_ = GomokuTools.dirs()[h][0]
        r = range(8) if h_//4 else range(7, -1, -1)
        return [ 
            [np.sign(self.ba[y]   & (1<<i)) for i in r],
            [np.sign(self.ba[y+1] & (1<<i)) for i in r]]

    def as_hex(self):
        return self.ba.hex()
    
    def as_bits(self):
        return [self.bits_in_line(h) for h in ['e', 'ne', 'n', 'nw']]
    
    def __repr__(self):
        field = [[' ' for i in range(9)] for j in range(9)]
        field[4][4]='*'
        for h in list(GomokuTools.dirs().items())[:4]:               
            step = GomokuTools.dirs()[h[0]][1]
            pos0 = np.array([4,4]) - 4 * np.array(step)
            bits = self.bits_in_line(h[0])
            for x in range(8):
                row, col = pos0 + (x + x//4) *np.array(step)
                field[row][col]='x' if bits[0][x] == 1 else 'o' if bits[1][x] == 1 else ' '                
        return "\n".join([('|' + ' '.join(field[r]) + '|') for r in range(9)])

    
