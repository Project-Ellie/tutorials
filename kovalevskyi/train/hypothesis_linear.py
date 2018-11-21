def hypothesis_linear(features, feature_columns, options):
    
    import tensorflow as tf

    all_feature_columns = feature_columns['wide'] + feature_columns['deep']
    
    input_layer = tf.feature_column.input_layer( 
        features, feature_columns=all_feature_columns)

    out = tf.layers.dense(input_layer, 1, activation=None)

    return out
