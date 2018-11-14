def make_hypothesis(input_layer, options):
    
    import tensorflow as tf
        
    out = tf.layers.dense(input_layer, 1, activation=None)

    return out
