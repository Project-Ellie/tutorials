def make_tfr_input_fn(filename_pattern, batch_size, shuffle_buffer_size=10000, distribute=False,
                     reader_num_threads=16, parser_num_threads=16, sloppy_ordering=True,
                      prefetch_buffer_size=1024):
    
    import tensorflow as tf
    from train.model_config import LABEL_COLUMN
    from train.model_config import TRAINING_METADATA

    feature_spec = TRAINING_METADATA.schema.as_feature_spec()

    def _input_fn():
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=filename_pattern,
            batch_size=batch_size,
            features=feature_spec,
            shuffle_buffer_size=shuffle_buffer_size,
            prefetch_buffer_size=prefetch_buffer_size,
            reader_num_threads=reader_num_threads,
            parser_num_threads=parser_num_threads,
            sloppy_ordering=sloppy_ordering,
            label_key=LABEL_COLUMN)

        if distribute:
            return dataset 
        else:
            return dataset.make_one_shot_iterator().get_next()
    return _input_fn
