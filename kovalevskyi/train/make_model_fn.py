def make_model_fn(feature_columns, options):
    
    import tensorflow as tf
    from train.make_hypothesis import make_hypothesis
    
    def _model_fn(features, labels, mode):

        input_layer = tf.feature_column.input_layer( 
            features, feature_columns=feature_columns)

        #############################################################
        # This single line is the actual model
        #############################################################
        out = make_hypothesis(input_layer, options)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=out)


        labels = tf.expand_dims(labels, -1)
        loss = tf.losses.mean_squared_error(labels, out)

        if mode == tf.estimator.ModeKeys.EVAL:    
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss = loss,
                #eval_metric_ops={'my_metric': }
            )

        else:
            optimizer = tf.train.GradientDescentOptimizer(options['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

            return tf.estimator.EstimatorSpec(  
                mode,
                loss = loss,
                train_op = train_op)
        
    return _model_fn
