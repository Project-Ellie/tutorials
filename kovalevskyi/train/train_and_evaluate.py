def train_and_evaluate(options):

    import tensorflow as tf
    from train.make_model_fn import make_model_fn
    from train.make_tft_serving_input_fn import make_tft_serving_input_fn
    from train.create_feature_columns import create_feature_columns

    #from train.make_input_fn import make_input_fn
    from train.make_tfr_input_fn import make_tfr_input_fn
    from train.make_hypotheses import make_hypotheses

    hypothesis = make_hypotheses()[options['hypothesis']]
    
    feature_columns = create_feature_columns()
    
    strategy = tf.contrib.distribute.MirroredStrategy() if options['distribute'] else None
    config = tf.estimator.RunConfig(model_dir=options['model_dir'], 
                                    train_distribute=strategy, 
                                    save_checkpoints_steps=options['save_checkpoints_steps'],
                                    log_step_count_steps=options['log_step_count_steps'])
        

    model_fn = make_model_fn(feature_columns, options, hypothesis )

    estimator = tf.estimator.Estimator(
            config=config,
            model_fn=model_fn)

    exporter = tf.estimator.LatestExporter('exporter', 
                                           make_tft_serving_input_fn(options['metadata_dir']))

    train_input_fn = make_tfr_input_fn(
        options['train_data_pattern'], shuffle_buffer_size=80000, 
        batch_size=options['train_batch_size'], distribute=options['distribute'],
        prefetch_buffer_size=options['prefetch_buffer_size'])

    eval_input_fn = make_tfr_input_fn(
        options['eval_data_pattern'], 
        batch_size=options['eval_batch_size'])  

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=options['max_train_steps'])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, exporters=exporter,
        steps = options['eval_steps'],
        throttle_secs=options['throttle_secs'],
        start_delay_secs=0)
    
    tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)
