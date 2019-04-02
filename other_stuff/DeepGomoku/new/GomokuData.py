import numpy as np
from GomokuTools import GomokuTools as gt
from GomokuBoard import GomokuBoard
from HeuristicPolicy import HeuristicGomokuPolicy
from QFunction import heuristic_QF

BLACK=0
WHITE=1
EDGES=2
STYLE_MIXED=2

def record_game(policy, max_n = 40):
    board = policy.board
    n = 0
    board.compute_all_scores()
    move = policy.suggest()
    while move.status == 0 and n < max_n:
        board.set(move.x,move.y)
        board.compute_all_scores()
        move = policy.suggest()
        n+=1
    return board


def variants_for(board):
    """
    Create a tensor 8x2xNxN to represent the 8 equivalent boards 
    that can be created from the stones by reflection and rotation.
    """
    stones = board.stones.copy()
    N = board.N
    array=np.zeros([8,2,N,N], dtype=float)
    color = np.arange(len(stones)) % 2
    for l, pos in list(zip(color, stones)):
        r, c = gt.b2m(pos, 15)
        array[0][l][r][c] = 1.0
        array[6][l][c][r] = 1.0

        array[1][l][c][N-r] = 1.0
        array[4][l][N-r][c] = 1.0

        array[3][l][N-c][r] = 1.0
        array[7][l][r][N-c] = 1.0

        array[2][l][N-r][N-c] = 1.0
        array[5][l][N-c][N-r] = 1.0

    return array


def transform(stones, N, quarters, reflect=False):
    """
    return stones' coordinates after rotation and reflection
    """
    coords = [gt.b2m(stone, N) for stone in stones]
    if quarters == 0:
        res = coords

    if quarters == 1:
        res = [(N-c-1, r) for r,c in coords]

    if quarters == 2:
        res = [(N-r-1, N-c-1) for r,c in coords]

    if quarters == 3:
        res = [(c, N-r-1) for r,c in coords]

    if reflect:
        res = [(r, N-c-1) for r,c in res]
        
    stones = [gt.m2b(coord, N) for coord in res]
    return stones


def create_sample(stones, N, viewpoint):
    
    sample = np.zeros([2, N, N], dtype=np.uint8)

    current = 0
    for move in stones:
        r,c=gt.b2m(move,N)
        sample[current][r][c]=1
        current = 1-current

        
    offensive = np.hstack([
        np.zeros([N+2,1], dtype=np.uint8), 
        np.vstack([np.zeros(N, dtype=np.uint8), 
                   sample[viewpoint], 
                   np.zeros(N, dtype=np.uint8)]),
        np.zeros([N+2,1], dtype=np.uint8)
    ])
    
    defensive = np.hstack([
        np.ones([N+2,1], dtype=np.uint8), 
        np.vstack([np.ones(N, dtype=np.uint8), 
                   sample[1-viewpoint], 
                   np.ones(N, dtype=np.uint8)]),
        np.ones([N+2,1], dtype=np.uint8)
    ])
    both = np.array([offensive, defensive])
    return np.rollaxis(both, 0, 3)


def wrap_sample(array, value):
    N = np.shape(array)[0]
    return np.hstack([
        np.zeros([N+2,1], dtype=np.float32) + value, 
        np.vstack([np.zeros(N, dtype=np.float32) + value, 
                   array, 
                   np.zeros(N, dtype=np.float32) + value]),
        np.zeros([N+2,1], dtype=np.float32) + value
    ])


def create_samples_and_qvalues(board, heuristics):
    """
    create 8 equivalent samples and qvalues from the given board
    """
    all_stones_t = [transform(board.stones.copy(), board.N, rot, ref) 
                    for rot in range(4)
                    for ref in [False, True]]

    samples = []
    qvalues = []
    for stones_t in all_stones_t:
        sample = create_sample(stones_t, board.N, 1-board.current_color)
        board = GomokuBoard(heuristics=heuristics, stones=stones_t)
        policy = HeuristicGomokuPolicy(board, STYLE_MIXED)
        qvalue, default_value = heuristic_QF(board, policy)
        qvalue = wrap_sample(qvalue, default_value)
        samples.append(sample)
        qvalues.append(qvalue)

    return np.array(samples), np.reshape(qvalues, [8, board.N+2, board.N+2, 1])


def data_from_game(board, policy, heuristics):    
    """
    Careful: This function rolls back the board
    """
    s,v = create_samples_and_qvalues(board, heuristics)
    while board.cursor > 6:
        board.undo()
        s1, v1 = create_samples_and_qvalues(board, heuristics)
        s = np.concatenate((s,s1))
        v = np.concatenate((v,v1))
    return s,v