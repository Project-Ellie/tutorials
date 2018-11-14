SIGNATURE_INT_COLUMNS = ['DEP_DOW', 'DEP_T']
SIGNATURE_FLOAT_COLUMNS = ['DEP_LAT', 'DEP_LON', 'DEP_DELAY', 'MEAN_TEMP_DEP', 
                 'MEAN_VIS_DEP', 'WND_SPD_DEP', 'ARR_LAT', 
                 'ARR_LON', 'ARR_DELAY', 'MEAN_TEMP_ARR', 'MEAN_VIS_ARR', 'WND_SPD_ARR']

TRAINING_COLUMNS=[
    'DEP_DOW', 'DEP_HOD', 
    
    'DEP_LAT', 'DEP_LON', 
    'MEAN_TEMP_DEP', 'MEAN_VIS_DEP', 'WND_SPD_DEP', 
    'DEP_DELAY', 

    'ARR_LAT', 'ARR_LON', 'ARR_DELAY',
    'MEAN_TEMP_ARR', 'MEAN_VIS_ARR', 'WND_SPD_ARR',  

    'DIFF_LAT', 'DIFF_LON', 'DISTANCE']

TRAINING_DEFAULTS = [
    [0], [0], 

    [0.], [0.], 
    [0.], [0.], [0.], 
    [0.], 

    [0.], [0.], [0.],
    [0.], [0.], [0.], 
    
    [0.], [0.], [0.]]

# Need strictly increasing column names for tf.data csv decoder
from operator import itemgetter
C_D = zip(TRAINING_COLUMNS, TRAINING_DEFAULTS)
C_D.sort(key=itemgetter(0))


ORDERED_TRAINING_COLUMNS = [item[0] for item in C_D]
ORDERED_TRAINING_DEFAULTS = [item[1] for item in C_D]

LABEL_COLUMN = 'ARR_DELAY'
