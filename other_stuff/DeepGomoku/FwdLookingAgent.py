from FwdLookingPolicy import FwdLookingPolicy

class FwdLookingAgent:
    def __init__(self, board, depth, width=(2,0)):
        self.board = board
        self.depth = depth
        self.width = width
        self.policy = FwdLookingPolicy(board)
        
    def suggest(self):        
        threshold = self.board.lsh.heuristics.tactical_threshold()
        top1 = self.board.top2(1)

        #print("Checking:", threshold, top1[1][0][1])
        if not self.board.stones:
            return {'reason': 'default center', 
                    'move': (self.board.size//2+1, self.board.size//2+1)}

        
        if top1[0][0][1] > 3.99: # that's a four, accounting for rounding errors
            return {'reason': 'immediate win', 'move': top1[0][0][0]}
        elif top1[1][0][1] > 3.99: 
            return {'reason': 'immediate loss prevention', 'move': top1[1][0][0]}

        elif top1[1][0][1] > threshold:
            return {'reason': 'defensive MUST move', 'move': top1[1][0][0]}
        
        else:
            sequence = self.policy.future_value(self.width, self.depth)
            if type(sequence) == str:
                    return {'reason': "Resignation", 'move': None}

            return {'reason': "Tree: " + str(sequence[1:]), 'move': sequence[0]}
        
    def move(self):
        suggestion = self.suggest()
        if suggestion['move'] != (-1, -1):
            print(suggestion)
            self.board.set(*suggestion['move'])
        else:
            print("Failed to find a move. Giving up.")
            