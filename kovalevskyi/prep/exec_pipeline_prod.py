def exec_pipeline_prod (options, train_dir, eval_dir, test_dir, 
                        metadata_dir, tmp_dir,
                        fractions, sample_rate, prefix,
                        runner='DirectRunner'):
    
    import os
    import tensorflow_transform as tft
    import tensorflow_transform.beam.impl as beam_impl
    import apache_beam as beam
    from tensorflow_transform.tf_metadata import dataset_metadata
    from tensorflow_transform.tf_metadata import dataset_schema
    from tensorflow_transform.beam.tft_beam_io import transform_fn_io
    
    from train.model_config import (SIGNATURE_COLUMNS, TRAINING_COLUMNS,
        SIGNATURE_METADATA)
    from prep.pre_process import pre_process
    from prep.sample_queries import sample_queries

    
    with beam.Pipeline(runner, options=options) as p:
        with beam_impl.Context(temp_dir=tmp_dir):
            
            # Process training data and obtain transform_fn
            #
            queries = sample_queries(SIGNATURE_COLUMNS, fractions, sample_rate)

            signature_data = (p | "ReadFromBigQuery_train"  
                              >> beam.io.Read(beam.io.BigQuerySource(
                                  query=queries['train'], use_standard_sql=True)))
            signature_dataset = (signature_data, SIGNATURE_METADATA)
            
            tds, transform_fn = (signature_dataset | "AnalyzeAndTransform" 
                        >> beam_impl.AnalyzeAndTransformDataset(pre_process))
            t_data, t_metadata = tds

            train_prefix = os.path.join(train_dir, prefix)
            encoder = tft.coders.ExampleProtoCoder(t_metadata.schema)

            _ = (t_data
                 | 'EncodeTFRecord_train' >> beam.Map(encoder.encode)
                 | 'WriteTFRecord_train' >> beam.io.WriteToTFRecord(train_prefix))
        
        
            #  Process evaluation data with the obtained transform_fn
            #
            signature_data = (p | "ReadFromBigQuery_eval"  
                              >> beam.io.Read(beam.io.BigQuerySource(
                                  query=queries['eval'], use_standard_sql=True))) 
            signature_dataset = (signature_data, SIGNATURE_METADATA)

            t_dataset = ((signature_dataset, transform_fn) 
                         | "TransformEval" >> beam_impl.TransformDataset())
            t_data, _ = t_dataset
            eval_prefix = os.path.join(eval_dir, prefix)
            _ = (t_data
                 | 'EncodeTFRecord_eval' >> beam.Map(encoder.encode)
                 | 'WriteTFRecord_eval' >> beam.io.WriteToTFRecord(eval_prefix))
        
            
            #  Also process test data with the obtained transform_fn
            #
            signature_data = (p | "ReadFromBigQuery_test"  
                              >> beam.io.Read(beam.io.BigQuerySource(
                                  query=queries['test'], use_standard_sql=True)))
            signature_dataset = (signature_data, SIGNATURE_METADATA)

            t_dataset = ((signature_dataset, transform_fn) 
                         | "TransformTest" >> beam_impl.TransformDataset())
            t_data, _ = t_dataset
            test_prefix = os.path.join(test_dir, prefix)
            _ = (t_data
                 | 'EncodeTFRecord_test' >> beam.Map(encoder.encode)
                 | 'WriteTFRecord_test' >> beam.io.WriteToTFRecord(test_prefix))
        
            
            # save transforma function to disk for use at serving time
            #
            transform_fn | 'WriteTransformFn' >> transform_fn_io.WriteTransformFn(metadata_dir)
