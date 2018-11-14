def join_paths(args):
    """
    A little helper that allows us to specify directories with the help of a particular base path
    """
    import os
    def combine_path(basedir, other):
        if other.startswith("/") or other.startswith("gs://"):
            return other
        else:
            return os.path.join(basedir, other)
    for key in ['metadata_dir', 'model_dir', 'eval_data_pattern', 'train_data_pattern']:
        args[key] = combine_path(args['base_dir'], args[key])
    return args