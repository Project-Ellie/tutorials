if [ ! -d ~/py2 ] 
then
	virtualenv --python=python2.7 ~/py2
	source ~/py2/bin/activate
	pip install google_compute_engine
	gsutil cp gs://going-tfx/transfer/tensorflow-1.12.0rc2.748435b.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl ~/
	pip install ~/tensorflow-1.12.0rc2.748435b.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl
	pip install --upgrade -r ./requirements.txt
	pip install jupyterlab
else
	source ~/py2/bin/activate
fi
jupyter-lab --config=./jupyter_notebook_config.py
