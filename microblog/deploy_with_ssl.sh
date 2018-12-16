
echo == creating the app...
kubectl run microblog --image=us.gcr.io/going-tfx/microblog:v1 --port=5000
echo done.
echo

echo == creating service for the app...
kubectl apply -f k8s/service.yaml
echo done.
echo

echo == creating http01 challenge endpoint...
kubectl apply -f k8s/ingress-tls.yaml
echo done.

echo now waiting for ACME http01 challenge endpoint to become available...
while [ "$(curl http://microblog.fnf.ninja 2>&1 | grep Wolfie | cut -d, -f2 | cut -d! -f1)" != ' Wolfie' ]
do     
    sleep 10
done

echo == Endpoint is ready.
curl http://microblog.fnf.ninja 2>&1 | grep Wolfie 
echo

echo == creating certificate...
kubectl apply -f k8s/certificate.yaml
echo done.
echo 

while [ true ]
do
    kubectl describe -f k8s/certificate.yaml | tail -4
    sleep 30
done
