# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example implementation of code to run on the Cloud ML service.
"""

import traceback
import argparse
import json
import os
import tensorflow as tf

from train.train_tools import join_paths
from train.train_and_evaluate import train_and_evaluate

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--base_dir',
        help = 'base_dir from which to interpret all other relative paths',
        required = True
    )
    parser.add_argument(
        '--metadata_dir',
        help = 'base_dir from which to interpret all other relative paths',
        required = True
    )
    
    parser.add_argument(
        '--train_data_pattern',
        help = 'GCS path glob pattern to training data',
        required = True
    )
    parser.add_argument(
        '--train_batch_size',
        help = 'Batch size for training steps',
        type = int,
        default = 512
    )
    parser.add_argument(
        '--max_train_steps',
        help = 'Max num steps to run the training job up to',
        type = int,
        default = 5000
    )

    parser.add_argument(
        '--eval_data_pattern',
        help = 'GCS patch glob pattern to evaluation data',
        required = True
    )
    
    parser.add_argument(
        '--file_format',
        help='File format. One of "csv", "tfr"',
        type = str,
        required = True
    )
    
    parser.add_argument(
        '--eval_batch_size',
        help = 'Batch size for evaluation steps',
        type = int,
        default = 512
    )
    parser.add_argument(
        '--eval_steps',
        help = 'Number of steps to run evalution for at each checkpoint',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--eval_throttle',
        help = 'number of seconds between two evaluations',
        default = 10,
        type = int
    )
    # Training arguments
    parser.add_argument(
        '--learning_rate',
        help = 'How long to wait before running first evaluation',
        default = 1e-4,
        type = float
    )
    parser.add_argument(
        '--nbuckets',
        help = 'Number of buckets into which to discretize lats and lons',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--hidden_units',
        help = 'Hidden layer sizes to use for DNN feature columns -- provide space-separated layers',
        type = str,
        default = "128 32 4"
    )
    parser.add_argument(
        '--model_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    
    
    parser.add_argument(
        '--distribute',
        help = 'boolean. Whether or not to distribute between GPUs',
        type = bool
    )
    parser.add_argument(
        '--prefetch_buffer_size',
        help = 'how many batches to pre-fetch into GPU memory',
        type = int,
        default = 1024
    )
    
    parser.add_argument(
        '--throttle_secs',
        help="evaluate every after this number of seconds, given a checkpoint is available.",
        type = int,
        default=30
    )
    parser.add_argument(
        '--log_step_count_steps',
        help="log current training loss every this number of steps",
        type = int,
        default = 1000
    )
    parser.add_argument(
        '--hypothesis',
        help="name of the hypothesis. Examine 'hypotheses.py' for a list of available names.",
        type = str,
        required=True
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        help="number of steps to go until another check_point is created.",
        type = int,
        default = 2000
    )        
    
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'junk'
    )


    parser.add_argument(
        '--parser_num_threads',
        help="Number of Parser threads to use",
        type = int,
        default=16
    )    
    parser.add_argument(
        '--reader_num_threads',
        help="Number of io reader threads to use",
        type = int,
        default=16
    )
    parser.add_argument(
        '--shuffle_buffer_size',
        help="Shuffle buffer size",
        type = int,
        default=10000
    )
    parser.add_argument(
        '--sloppy_ordering',
        help="Whether to allow non-deterministic ordering. Boosts performance",
        type = bool,
        default=True
    )
    

    def not_now():
        # Eval arguments
        parser.add_argument(
            '--eval_delay_secs',
            help = 'How long to wait before running first evaluation',
            default = 10,
            type = int
        )
        parser.add_argument(
            '--min_eval_frequency',
            help = 'Minimum number of training steps between evaluations',
            default = 1,
            type = int
        )
        parser.add_argument(
            '--format',
            help = 'Is the input data format csv or tfrecord?',
            default = 'csv'
        )

    args = parser.parse_args()
    arguments = args.__dict__

    # Unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    arguments['model_dir'] = os.path.join(
        arguments['model_dir'],
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    ) 

    if arguments['distribute']:
        print("###############################################################")
        print("       Running distributed!")
        print("###############################################################")
    # Run the training job:
    try:
        train_and_evaluate(join_paths(arguments))
    except:
        traceback.print_exc()