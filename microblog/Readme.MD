### Creating a secure web app on GKE

We begin our journey by creating a VM instance on the compute engine. That's so easy that my only advice is:

*Make sure to select "Allow full access to all Cloud APIs"*

Then ```ssh``` to it using the button named "SSH" next to your instance. From now on, I assume the instance name is "microblog-dev".

```
$ cd
$ mkdir workspace
$ cd workspace
$ export WORKSPACE=$(pwd)
$ sudo apt-get install python3-pip
$ sudo apt-get install virtualenv
$ virtualenv venv3 --python=python3
$ source venv3/bin/activate
$ pip install jupyterlab
$ sudo apt-get install git
$ git clone https://github.com/Project-Ellie/tutorials
$ jupyter-lab --config=tutorials/jupyter_notebook_config.py

```


On a cloud shell
```
$ gcloud compute ssh microblog-dev --zone=us-central1-a -- -L 8081:localhost:8081
```
Open a web preview on Port 8081 and enjoy the warm welcome of your fresh jupyter lab environment. Navigate to tutorials/microblog and open Tutorial.ipynb. Continue from there. 

Have fun!
