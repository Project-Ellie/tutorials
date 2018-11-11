import math

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as beam_impl
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.beam.tft_beam_io import transform_fn_io

from tools import tf_haversine


PROJECT='going-tfx'
BUCKET='going-tfx'
DATASET='examples'

LOCAL_BASE_DIR='/tmp/atl_june'


def directories(env, stage): 
    if env == 'gs':
        form = "gs://{}/{}/{}".format(BUCKET, '{}', {})
    elif env == 'local':
        form = LOCAL_BASE_DIR + "/{}/{}"
    else: 
        raise Exception("Environment {} not supported".format(env))
        
    def _dir(usage):
        return form.format(stage, usage)
    return {usage: _dir(usage) for usage in ['tmp', 'data', 'metadata']}

#
# Selected Columns in the database
#
SIGNATURE_INT_COLUMNS = ['DEP_DOW', 'DEP_T']
SIGNATURE_FLOAT_COLUMNS = ['DEP_LAT', 'DEP_LON', 'DEP_DELAY', 'MEAN_TEMP_DEP', 
                 'MEAN_VIS_DEP', 'WND_SPD_DEP', 'ARR_LAT', 
                 'ARR_LON', 'ARR_DELAY', 'MEAN_TEMP_ARR', 'MEAN_VIS_ARR', 'WND_SPD_ARR']

SIGNATURE_COLUMNS = SIGNATURE_INT_COLUMNS + SIGNATURE_FLOAT_COLUMNS
#
# Schema of the signature stage data for tftransform
#
SIGNATURE_SCHEMA = {}


for t, cols in [(tf.float32, SIGNATURE_FLOAT_COLUMNS), (tf.int64, SIGNATURE_INT_COLUMNS)]:
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


def add_engineered(row):
    dep_lat = row['DEP_LAT']
    dep_lon = row['DEP_LON']
    arr_lat = row['ARR_LAT']
    arr_lon = row['ARR_LON']

    row['DEP_HOD'] = row['DEP_T'] // 100
    row.pop('DEP_T')  # no longer needed

    row['DIFF_LAT'] = arr_lat - dep_lat
    row['DIFF_LON'] = arr_lon - dep_lon
    row['DISTANCE'] = tf_haversine(arr_lat, arr_lon, dep_lat, dep_lon)
    return row

def scale_floats(row):
    for c in ['MEAN_TEMP_DEP', 'MEAN_VIS_DEP', 'WND_SPD_DEP', 'MEAN_TEMP_ARR', 'MEAN_VIS_ARR', 'WND_SPD_ARR', 'DEP_DELAY',
             'DIFF_LAT', 'DIFF_LON', 'DISTANCE']:
        row[c] = tft.scale_to_0_1(row[c])
    return row

def pre_processor(row):
    row = row.copy()
    row = add_engineered(row)
    row = scale_floats(row)
    return row