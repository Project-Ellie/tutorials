import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as beam_impl
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.beam.tft_beam_io import transform_fn_io


PROJECT='going-tfx'
BUCKET='going-tfx'
DATASET='examples'


def directories(env, stage): 
    if env == 'gs':
        form = "gs://{}/{}/{}".format(BUCKET, '{}', {})
    elif env == 'local':
        form = "/tmp/atl_june/{}/{}"
    else: 
        raise Exception("Environment {} not supported".format(env))
        
    def _dir(usage):
        return form.format(stage, usage)
    return {usage: _dir(usage) for usage in ['tmp', 'data', 'metadata']}

#
# Columns in the database
#
STRING_COLUMNS = ['DATE', 'AIRLINE', 'DEP', 'DEP_W', 'ARR', 'ARR_W']
INT_COLUMNS = ['YEAR', 'MONTH', 'DAY', 'DEP_DOW', 'DEP_T', 'ARR_T']
FLOAT_COLUMNS = ['DEP_LAT', 'DEP_LON', 'DEP_DELAY', 'MEAN_TEMP_DEP', 
                 'MEAN_VIS_DEP', 'WND_SPD_DEP', 'ARR_DELAY', 'ARR_LAT', 
                 'ARR_LON', 'MEAN_TEMP_ARR', 'MEAN_VIS_ARR', 'WND_SPD_ARR']

SIGNATURE_COLUMNS = STRING_COLUMNS + INT_COLUMNS + FLOAT_COLUMNS
#
# Schema of the signature stage data for tftransform
#
SIGNATURE_SCHEMA = {}


for t, cols in [(tf.string, STRING_COLUMNS), (tf.float32, FLOAT_COLUMNS), (tf.int64, INT_COLUMNS)]:
    SIGNATURE_SCHEMA.update({
        col : dataset_schema.ColumnSchema(t, [], dataset_schema.FixedColumnRepresentation())
                   for col in cols})
SIGNATURE_METADATA = dataset_metadata.DatasetMetadata(dataset_schema.Schema(SIGNATURE_SCHEMA))


#
# Signature placeholders. Will be used for the serving_input_fn
# 
spec = SIGNATURE_METADATA.schema.as_feature_spec()
SIGNATURE_PLACEHOLDERS={
    key: tf.placeholder(name=key, shape=[None], dtype=spec[key].dtype) for key in spec.keys()
}

#
# These are the dense features that will go into model training 
# Note that the feature_column operations, such as integerizing,
# bucketizing, crossing, and embedding are still to be applied.
#

TRAINING_COLUMNS=[
    'YEAR', 'MONTH', 'DEP_DOW', 'DEP_HOD', 'AIRLINE', 
    
    'DEP', 'DEP_LAT', 'DEP_LON', 
    'MEAN_TEMP_DEP', 'MEAN_VIS_DEP', 'WND_SPD_DEP', 
    'DEP_T', 'DEP_DELAY', 

    'ARR', 'ARR_LAT', 'ARR_LON', 
    'MEAN_TEMP_ARR', 'MEAN_VIS_ARR', 'WND_SPD_ARR', 
    'ARR_T', 'ARR_DELAY', 

    'DIFF_LAT', 'DIFF_LON', 'DISTANCE']

TRAINING_DEFAULTS = [
    [0], [0], [0], [0], ['none'], 

    ['none'], [0.], [0.], 
    [0.], [0.], [0.], 
    [0], [0.],

    ['none'], [0.], [0.], 
    [0.], [0.], [0.], 
    [0], [0.0],
    
    [0.], [0.], [0.]]

# Need strictly increasing column names for tf.data csv decoder
from operator import itemgetter
C_D = zip(TRAINING_COLUMNS, TRAINING_DEFAULTS)
C_D.sort(key=itemgetter(0))


ORDERED_TRAINING_COLUMNS = [item[0] for item in C_D]
ORDERED_TRAINING_DEFAULTS = [item[1] for item in C_D]