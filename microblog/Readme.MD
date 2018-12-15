## Creating a secure web app on GKE

If you have a linux OS somewhere with git, gcloud, kubectl, and jupypter, then you can just clone this repo and start from Tutorial.ipynb, otherwise you may follow the following steps to set up a virtual machine on GCE that will serve as your workbench.

#### Preparation


We begin our journey by creating a VM instance on the compute engine. I'm using the Debian OS that comes up as default. 
*Make sure to select "Allow full access to all Cloud APIs"*

Then ```ssh``` to it using the button named "SSH" next to your instance. From now on, I assume the instance name is "microblog-dev".

```
# Install jupyter and git and clone this repo
cd
mkdir workspace
cd workspace
sudo apt-get install -y python3-pip
sudo apt-get install -y virtualenv
virtualenv venv3 --python=python3
source venv3/bin/activate
pip install jupyterlab
sudo apt-get install -y git
git clone https://github.com/Project-Ellie/tutorials

# Cloud SDK (gcloud and friends)
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y google-cloud-sdk
sudo apt-get install -y kubectl

# docker
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg2 software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce

# start the jupyter server
jupyter-lab --config=tutorials/jupyter_notebook_config.py
```

Open a [cloud shell](https://cloud.google.com/shell/docs/quickstart).

On that shell, create an ssh tunnel to your jupyter lab server
```
$ gcloud compute ssh microblog-dev --zone=us-central1-a -- -L 8081:localhost:8081
```
Open a web preview on Port 8081 and enjoy the warm welcome of your fresh jupyter lab environment. Navigate to tutorials/microblog and open Tutorial.ipynb. Continue from there. 

Have fun!
