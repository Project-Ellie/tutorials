#! /bin/bash
echo == creating cluster...
gcloud container clusters create microblog-cluster --num-nodes=2
echo done.
echo 

echo == getting credentials...
gcloud container clusters get-credentials microblog-cluster --zone us-central1-a --project going-tfx
echo done.

echo == creating tiller service account...
kubectl create serviceaccount -n kube-system tiller
echo done.
echo 

echo == creating cluster role binding...
kubectl create clusterrolebinding tiller-binding --clusterrole=cluster-admin --serviceaccount kube-system:tiller
echo done.
echo 

echo == initializing helm...
helm init --service-account tiller
echo done.
echo 

echo == updating helm repo...
helm repo update
echo done.
echo 

echo == now waiting for tiller pods to become available...
while [ $(kubectl get pods --namespace kube-system | grep tiller | awk '{print $3}') != "Running" ] 
do
    sleep 10
done    

echo == The pod is there, but we give it another 60 seconds...
sleep 60
echo ok. 
echo

echo == installing cert-manager...
helm install --name cert-manager --version v0.3.2 --namespace kube-system stable/cert-manager
echo done.
echo 

echo == installing cluster issuer...
kubectl apply -f k8s/clusterissuer.yaml
echo done.
echo

echo == showing cert-manager deployment...
kubectl get deployment -n kube-system cert-manager-cert-manager
echo done.
echo 

