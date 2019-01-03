    if [ "" = "$WORKSPACE" ]
then
	echo "WORKSPACE variable not set"
	echo "WORKSPACE must point to the directory containing the cloned git repo"
	exit -1
fi 

SRC_DIR="${WORKSPACE}/tutorials/kovalevskyi/"

if [ ! -d "$SRC_DIR" ]
then
	echo "$SRC_DIR doesn't exist"
	echo "Did you clone Project-Ellie/tutorials into $WORKSPACE?"
	exit -1
fi

if [ ! -d ~/py2 ] 
then
	virtualenv --python=python2.7 ~/py2
	source ~/py2/bin/activate
	pip install google_compute_engine
	if [ "$(which nvidia-smi)" = "" ]
	then
		echo "nvidia-smi not found. Installing tensorflow without GPU support."
		pip install tensorflow
	else
		echo "nvidia-smi found. Installing tensorflow with GPU support."
		gsutil cp gs://going-tfx/transfer/tensorflow-1.12.0rc2.748435b.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl ~/
		pip install ~/tensorflow-1.12.0rc2.748435b.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl
	fi
	pip install --upgrade -r ${SRC_DIR}/requirements.txt
	pip install jupyterlab
	pip install jupyter-tensorboard
else
	source ~/py2/bin/activate
fi
PYTHONHASHSEED=0 jupyter-lab --config=${SRC_DIR}/jupyter_notebook_config.py
