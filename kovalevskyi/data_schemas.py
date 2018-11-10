import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as beam_impl
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.beam.tft_beam_io import transform_fn_io

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
    key: tf.placeholder(shape=[None], dtype=spec[key].dtype) for key in spec.keys()
}

#
# These are the dense features that will go into model training 
# Note that the feature_column operations, such as integerizing,
# bucketizing, crossing, and embedding are still to be applied.
#

TRAINING_COLUMNS=[
    'YEAR', 'MONTH', 'DEP_DOW', 'AIRLINE', 
    
    'DEP', 'DEP_LAT', 'DEP_LON', 
    'MEAN_TEMP_DEP', 'MEAN_VIS_DEP', 'WND_SPD_DEP', 
    'DEP_T', 'DEP_DELAY', 

    'ARR', 'ARR_LAT', 'ARR_LON', 
    'MEAN_TEMP_ARR', 'MEAN_VIS_ARR', 'WND_SPD_ARR', 
    'ARR_T', 'ARR_DELAY', 

    'DIFF_LAT', 'DIFF_LON', 'DISTANCE']
