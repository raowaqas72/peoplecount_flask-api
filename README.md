


## commands to run

Install the dependencies and devDependencies and start the server.

```sh
git clone git@github.com:raowaqas72/peoplecount_flask-api.git
cd peoplecount_flask-api
bash ./setup.sh
pip3 install -r requirements.txt
python3 -m flask --app detect.py run
```
```sh
docker pull raowaqas72/peoplecount:api
docker run -p 5000:5000 raowaqas72/peoplecount:api
```
we can run following docker command in order to run everytime automatically on reboot 
```
docker run -d -p 5000:5000 --restart unless-stopped --net host raowaqas72/peoplecount:api

```
we can run the kubernetes cluster using following command to run peoplecount api on kiubernetes

```
kubectl apply -f peoplecount-deployment.yml 

```


<img src="https://github.com/raowaqas72/peoplecount_flask-api/blob/main/object_detector.jpg?raw=true" alt="Alt text" title="object detection ">
<img src="https://github.com/raowaqas72/peoplecount_flask-api/blob/main/Screenshot%20from%202023-01-19%2016-17-31.png" alt="Alt text" title="object detection ">
