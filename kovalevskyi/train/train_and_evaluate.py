def train_and_evaluate(options, distribute=False):

    import tensorflow as tf
    from train.make_model_fn import make_model_fn
    from train.make_tft_serving_input_fn import make_tft_serving_input_fn
    from train.make_input_fn import make_input_fn
    from train.create_feature_columns import create_feature_columns
    
    feature_columns = create_feature_columns()
    
    if distribute:
        strategy=tf.contrib.distribute.MirroredStrategy()    
        config = tf.estimator.RunConfig(model_dir=options['model_dir'], train_distribute=strategy)
    else:
        config = tf.estimator.RunConfig(model_dir=options['model_dir'])
        

    model_fn = make_model_fn(feature_columns, options)

    estimator = tf.estimator.Estimator(
            config=config,
            model_fn=model_fn)

    exporter = tf.estimator.LatestExporter('exporter', 
                                           make_tft_serving_input_fn(options['metadata_dir']))

    train_input_fn = make_input_fn(
        options['train_data_pattern'], shuffle_buffer_size=80000, 
        batch_size=options['train_batch_size'], distribute=distribute)

    eval_input_fn = make_input_fn(
        options['eval_data_pattern'], 
        batch_size=options['eval_batch_size'])  

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=options['max_train_steps'])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps = options['eval_batch_size'], exporters=exporter)
    
    tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)
