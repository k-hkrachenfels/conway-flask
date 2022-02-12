# Notes

## How to start

When you run this locally you need to have `ffmpeg` be installed 
```
python -m flask run
```
It is therefore recommended to run this by starting the docker container which brings all the dependencies you need.
Call it with the address exposed by the webserver.

## Build and run
You can build the docker image in the following way:
```
docker build -t khkrachenfels/game-of-life-conway-webserver .
```
And run it with
```
docker run -p5000:5000 khkrachenfels/game-of-life-conway-webserver:latest
```

## Deploying into kubernetes cluster
You can deploy it to dockerhub with
```
docker push khkrachenfels/game-of-life-conway-webserver
```
And start it on your kubernetes cluster (e.g. google cloud) with
(this assumes that kubectl runs on your computer and you have set up a cluter in the cloud)
```
kubectl apply -f my-deployment.yaml
kubectl apply -f my-service.yaml
```

Lookup the IP of your service and you can start it from the web (port 80)

## Example
![image](conway.png)


