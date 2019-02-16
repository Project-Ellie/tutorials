class NaiveOpponent:
    """
    A naive opponent that simply looks at line scores
    """
    def move(self, board, width=(1,1), depth=0):
        topn = board.top(1)
        #if depth == 0:
        o,d,od=topn[0], topn[1], topn[2]
        to, td, tod = o[0], d[0], od[0]
        if to[1] >= 2.99:
            choice = to[0]
            #print("defensive: (%s, %s)" % choice)
        elif td[1] >= 2.99:
            choice = td[0]
            #print("offensive: (%s, %s)" % choice)
        else:
            choice = tod[0]
            #print("combined: (%s, %s)" % choice)
        return choice

class FwdLookingPolicy:
    
    def __init__(self, board, kappa=3, opponent=NaiveOpponent()):
        self.board = board
        self.kappa = kappa
        self.opponent_model = opponent
        
        
        
    def future_value(self, width, depth):
        if depth == 0:
            top1 = self.board.top(1)
            to = top1[0][0][1]
            td = top1[1][0][1]
            return to**self.kappa - td**self.kappa

        path = []
        wo, wd = width
        ntop = max(wo, wd)
        topn = self.board.top(ntop)

        aggressive_moves = [i[0] for i in topn[0][:wo] ]    
        defensive_moves = [i[0] for i in topn[1][:wo] ]    
        moves = aggressive_moves + defensive_moves

        #print(moves)
        options=[]
        for p in moves:
            self.board.set(*p)
            opponent_move = self.opponent_model.move(self.board, width, depth - 1)
            self.board.set(*opponent_move)

            # Don't consider any moves that reveal open-4 after opponent's move
            top1 = self.board.top(1)            
            top2 = self.board.top(2)            
            if top1[1][0][1] <= 4.3 and sum([item[1] for item in self.board.top(2)[1] ]) < 7.8: 
                fv = self.future_value(width, depth-2)
                if fv:
                    options.append((p, opponent_move, fv))
            
            self.board.undo()
            self.board.undo()


        best = [o for o in options if o[2] == max([o[2] for o in options])]
        best = best[0] if best else ((-1, -1), (-1, -1), -999)
        #if depth <= 2:
        #    print("Depth %s, best move: %s" % (depth, best))
        #print(options)
        #print("--------------")
        return best
    
    
    
