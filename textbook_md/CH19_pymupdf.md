### **CHAPTER 19** 

# **Training and Deploying TensorFlow Models at Scale** 

Once you have a beautiful model that makes amazing predictions, what do you do with it? Well, you need to put it in production! This could be as simple as running the model on a batch of data, and perhaps writing a script that runs this model every night. However, it is often much more involved. Various parts of your infrastructure may need to use this model on live data, in which case you will probably want to wrap your model in a web service: this way, any part of your infrastructure can query the model at any time using a simple REST API (or some other protocol), as we discussed in Chapter 2. But as time passes, you’ll need to regularly retrain your model on fresh data and push the updated version to production. You must handle model versioning, gracefully transition from one model to the next, possibly roll back to the previous model in case of problems, and perhaps run multiple different models in parallel to perform _A/B experiments_ .<sup>1</sup> If your product becomes successful, your service may start to get a large number of of queries per second (QPS), and it must scale up to support the load. A great solution to scale up your service, as you will see in this chapter, is to use TF Serving, either on your own hardware infrastructure or via a cloud service such as Google Vertex AI.<sup>2</sup> It will take care of efficiently serving your model, handle graceful model transitions, and more. If you use the cloud platform you will also get many extra features, such as powerful monitoring tools. 

> 1 An A/B experiment consists in testing two different versions of your product on different subsets of users in order to check which version works best and get other insights. 

> 2 Google AI Platform (formerly known as Google ML Engine) and Google AutoML merged in 2021 to form Google Vertex AI. 

**721** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

Moreover, if you have a lot of training data and compute-intensive models, then training time may be prohibitively long. If your product needs to adapt to changes quickly, then a long training time can be a showstopper (e.g., think of a news recommendation system promoting news from last week). Perhaps even more impor‐ tantly, a long training time will prevent you from experimenting with new ideas. In machine learning (as in many other fields), it is hard to know in advance which ideas will work, so you should try out as many as possible, as fast as possible. One way to speed up training is to use hardware accelerators such as GPUs or TPUs. To go even faster, you can train a model across multiple machines, each equipped with multiple hardware accelerators. TensorFlow’s simple yet powerful distribution strategies API makes this easy, as you will see. 

In this chapter we will look at how to deploy models, first using TF Serving, then using Vertex AI. We will also take a quick look at deploying models to mobile apps, embedded devices, and web apps. Then we will discuss how to speed up com‐ putations using GPUs and how to train models across multiple devices and servers using the distribution strategies API. Lastly, we will explore how to train models and fine-tune their hyperparameters at scale using Vertex AI. That’s a lot of topics to discuss, so let’s dive in! 

## **Serving a TensorFlow Model** 

Once you have trained a TensorFlow model, you can easily use it in any Python code: if it’s a Keras model, just call its `predict()` method! But as your infrastructure grows, there comes a point where it is preferable to wrap your model in a small service whose sole role is to make predictions and have the rest of the infrastructure query it (e.g., via a REST or gRPC API).<sup>3</sup> This decouples your model from the rest of the infrastructure, making it possible to easily switch model versions or scale the service up as needed (independently from the rest of your infrastructure), perform A/B experiments, and ensure that all your software components rely on the same model versions. It also simplifies testing and development, and more. You could create your own microservice using any technology you want (e.g., using the Flask library), but why reinvent the wheel when you can just use TF Serving? 

#### **Using TensorFlow Serving** 

TF Serving is a very efficient, battle-tested model server, written in C++. It can sustain a high load, serve multiple versions of your models and watch a model repository to automatically deploy the latest versions, and more (see Figure 19-1). 

> 3 A REST (or RESTful) API is an API that uses standard HTTP verbs, such as GET, POST, PUT, and DELETE, and uses JSON inputs and outputs. The gRPC protocol is more complex but more efficient; data is exchanged using protocol buffers (see Chapter 13). 

###### **722 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



<!-- Start of picture text -->
inputs TF Serving<br>Predictions Model B v1<br>G Autodeploy<br>| Allversions model<br><!-- End of picture text -->



<!-- Start of picture text -->
|<br><!-- End of picture text -->

| 

```
$ saved_model_clishow--dir0001/my_mnist_model--tag_setserve\
--signature_defserving_default
The given SavedModel SignatureDef contains the following input(s):
  inputs['flatten_input'] tensor_info:
      dtype: DT_UINT8
      shape: (-1, 28, 28)
      name: serving_default_flatten_input:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['dense_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 10)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict
```

Note that the function’s input is named `"flatten_input"` , and the output is named `"dense_1"` . These correspond to the Keras model’s input and output layer names. You can also see the type and shape of the input and output data. Looks good! 

Now that you have a SavedModel, the next step is to install TF Serving. 

###### **Installing and starting TensorFlow Serving** 

There are many ways to install TF Serving: using the system’s package manager, using a Docker image,<sup>4</sup> installing from source, and more. Since Colab runs on Ubuntu, we can use Ubuntu’s `apt` package manager like this: 

```
url="https://storage.googleapis.com/tensorflow-serving-apt"
src="stable tensorflow-model-server tensorflow-model-server-universal"
!echo'deb {url} {src}'>/etc/apt/sources.list.d/tensorflow-serving.list
!curl'{url}/tensorflow-serving.release.pub.gpg'|apt-keyadd-
!aptupdate-q&&apt-getinstall-ytensorflow-model-server
%pip install -q -U tensorflow-serving-api
```

This code starts by adding TensorFlow’s package repository to Ubuntu’s list of package sources. Then it downloads TensorFlow’s public GPG key and adds it to the package manager’s key list so it can verify TensorFlow’s package signatures. Next, it uses `apt` to install the `tensorflow-model-server` package. Lastly, it installs the `tensorflow-serving-api` library, which we will need to communicate with the server. 

> 4 If you are not familiar with Docker, it allows you to easily download a set of applications packaged in a _Docker image_ (including all their dependencies and usually some good default configuration) and then run them on your system using a _Docker engine_ . When you run an image, the engine creates a _Docker container_ that keeps the applications well isolated from your own system—but you can give it some limited access if you want. It is similar to a virtual machine, but much faster and lighter, as the container relies directly on the host’s kernel. This means that the image does not need to include or run its own kernel. 

**Serving a TensorFlow Model | 725** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

Now we want to start the server. The command will require the absolute path of the base model directory (i.e., the path to `my_mnist_model` , not `0001` ), so let’s save that to the `MODEL_DIR` environment variable: 

```
importos
```

```
os.environ["MODEL_DIR"] =str(model_path.parent.absolute())
```

We can then start the server: 

```
%%bash --bg
tensorflow_model_server \
     --port=8500 \
     --rest_api_port=8501 \
     --model_name=my_mnist_model \
     --model_base_path="${MODEL_DIR}" >my_server.log 2>&1
```

In Jupyter or Colab, the `%%bash --bg` magic command executes the cell as a bash script, running it in the background. The `>my_server.log 2>&1` part redirects the standard output and standard error to the _my_server.log_ file. And that’s it! TF Serving is now running in the background, and its logs are saved to _my_server.log_ . It loaded our MNIST model (version 1), and it is now waiting for gRPC and REST requests, respectively, on ports 8500 and 8501. 

##### **Running TF Serving in a Docker Container** 

If you are running the notebook on your own machine and you have installed Docker, you can run `docker pull tensorflow/serving` in a terminal to download the TF Serving image. The TensorFlow team highly recommends this installation method because it is simple, it will not mess with your system, and it offers high performance.<sup>5</sup> To start the server inside a Docker container, you can run the following command in a terminal: 

```
$ dockerrun-it--rm-v"/path/to/my_mnist_model:/models/my_mnist_model"\
-p8500:8500-p8501:8501-eMODEL_NAME=my_mnist_modeltensorflow/serving
```

Here is what all these command-line options mean: 

```
-it
```

Makes the container interactive (so you can press Ctrl-C to stop it) and displays the server’s output. 

```
--rm
```

Deletes the container when you stop it: no need to clutter your machine with interrupted containers. However, it does not delete the image. 

- 5 There are also GPU images available, and other installation options. For more details, please check out the official installation instructions. 

###### **726 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

- `-v "/path/to/my_mnist_model:/models/my_mnist_model"` Makes the host’s _my_mnist_model_ directory available to the container at the path _/models/mnist_model_ . You must replace _/path/to/my_mnist_model_ with the absolute path of this directory. On Windows, remember to use `\` instead of `/` in the host path, but not in the container path (since the container runs on Linux). 

- `-p 8500:8500` 

Makes the Docker engine forward the host’s TCP port 8500 to the container’s TCP port 8500. By default, TF Serving uses this port to serve the gRPC API. 

- `-p 8501:8501` 

Forwards the host’s TCP port 8501 to the container’s TCP port 8501. The Docker image is configured to use this port by default to serve the REST API. 

- `-e MODEL_NAME=my_mnist_model` 

Sets the container’s `MODEL_NAME` environment variable, so TF Serving knows which model to serve. By default, it will look for models in the _/models_ directory, and it will automatically serve the latest version it finds. 

###### `tensorflow/serving` 

This is the name of the image to run. 

Now that the server is up and running, let’s query it, first using the REST API, then the gRPC API. 

###### **Querying TF Serving through the REST API** 

Let’s start by creating the query. It must contain the name of the function signature you want to call, and of course the input data. Since the request must use the JSON format, we have to convert the input images from a NumPy array to a Python list: 

```
importjson
```

```
X_new=X_test[:3]  # pretend we have 3 new digit images to classify
request_json=json.dumps({
"signature_name": "serving_default",
"instances": X_new.tolist(),
})
```

Note that the JSON format is 100% text-based. The request string looks like this: 

```
>>> request_json
'{"signature_name": "serving_default", "instances": [[[0, 0, 0, 0, ... ]]]}'
```

Now let’s send this request to TF Serving via an HTTP POST request. This can be done using the `requests` library (it is not part of Python’s standard library, but it is preinstalled on Colab): 

**Serving a TensorFlow Model | 727** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

```
importrequests
```

```
server_url="http://localhost:8501/v1/models/my_mnist_model:predict"
response=requests.post(server_url, data=request_json)
response.raise_for_status()  # raise an exception in case of error
response=response.json()
```

If all goes well, the response should be a dictionary containing a single `"predictions"` key. The corresponding value is the list of predictions. This list is a Python list, so let’s convert it to a NumPy array and round the floats it contains to the second decimal: 

```
>>> importnumpyasnp
>>> y_proba=np.array(response["predictions"])
>>> y_proba.round(2)
array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  ],
       [0.  , 0.  , 0.99, 0.01, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.97, 0.01, 0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.  ]])
```

Hurray, we have the predictions! The model is close to 100% confident that the first image is a 7, 99% confident that the second image is a 2, and 97% confident that the third image is a 1. That’s correct. 

The REST API is nice and simple, and it works well when the input and output data are not too large. Moreover, just about any client application can make REST queries without additional dependencies, whereas other protocols are not always so readily available. However, it is based on JSON, which is text-based and fairly verbose. For example, we had to convert the NumPy array to a Python list, and every float ended up represented as a string. This is very inefficient, both in terms of serialization/deserialization time—we have to convert all the floats to strings and back—and in terms of payload size: many floats end up being represented using over 15 characters, which translates to over 120 bits for 32-bit floats! This will result in high latency and bandwidth usage when transferring large NumPy arrays.<sup>6</sup> So, let’s see how to use gRPC instead. 



When transferring large amounts of data, or when latency is important, it is much better to use the gRPC API, if the client supports it, as it uses a compact binary format and an efficient communication protocol based on HTTP/2 framing. 

- 6 To be fair, this can be mitigated by serializing the data first and encoding it to Base64 before creating the REST request. Moreover, REST requests can be compressed using gzip, which reduces the payload size significantly. 

###### **728 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

###### **Querying TF Serving through the gRPC API** 

The gRPC API expects a serialized `PredictRequest` protocol buffer as input, and it outputs a serialized `PredictResponse` protocol buffer. These protobufs are part of the `tensorflow-serving-api` library, which we installed earlier. First, let’s create the request: 

```
fromtensorflow_serving.apis.predict_pb2importPredictRequest
```

```
request=PredictRequest()
request.model_spec.name=model_name
request.model_spec.signature_name="serving_default"
input_name=model.input_names[0]  # == "flatten_input"
request.inputs[input_name].CopyFrom(tf.make_tensor_proto(X_new))
```

This code creates a `PredictRequest` protocol buffer and fills in the required fields, including the model name (defined earlier), the signature name of the function we want to call, and finally the input data, in the form of a `Tensor` protocol buffer. The `tf.make_tensor_proto()` function creates a `Tensor` protocol buffer based on the given tensor or NumPy array, in this case `X_new` . 

Next, we’ll send the request to the server and get its response. For this, we will need the `grpcio` library, which is preinstalled in Colab: 

```
importgrpc
fromtensorflow_serving.apisimportprediction_service_pb2_grpc
```

```
channel=grpc.insecure_channel('localhost:8500')
predict_service=prediction_service_pb2_grpc.PredictionServiceStub(channel)
response=predict_service.Predict(request, timeout=10.0)
```

The code is quite straightforward: after the imports, we create a gRPC communica‐ tion channel to _localhost_ on TCP port 8500, then we create a gRPC service over this channel and use it to send a request, with a 10-second timeout. Note that the call is synchronous: it will block until it receives the response or when the timeout period expires. In this example the channel is insecure (no encryption, no authentication), but gRPC and TF Serving also support secure channels over SSL/TLS. 

Next, let’s convert the `PredictResponse` protocol buffer to a tensor: 

```
output_name=model.output_names[0]  # == "dense_1"
outputs_proto=response.outputs[output_name]
y_proba=tf.make_ndarray(outputs_proto)
```

If you run this code and print `y_proba.round(2)` , you will get the exact same estima‐ ted class probabilities as earlier. And that’s all there is to it: in just a few lines of code, you can now access your TensorFlow model remotely, using either REST or gRPC. 

**Serving a TensorFlow Model | 729** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



<!-- Start of picture text -->
|<br><!-- End of picture text -->

| 





<!-- Start of picture text -->
TF Serving<br>| Load P<br>TF Serving<br><!-- End of picture text -->



<!-- Start of picture text -->
FE _stortyour treet wih $300 in creat Dont worry you wort be charged if yourun out of ret, Lea more bismiss<br>= Google Cloud Platform Selecta project ~ Q Search Products, resources, docs (/) v aon:<br># Home ><br>Get Started with<br>@ Recent - Google Cloud Platform<br>t= View all products 90-day, $300 free trial to get you started<br>PINNED /i,| Always free products to keep you going [ |<br>TRY FOR FREE 4]<br>© tAMand admin ><br>& Billing<br>API APIs and services > Top products<br>W_ Marketplace° im =ooo g »<br>{2 Compute Engine > Compute Engine Cloud Storage Cloud SQL Cloud Run<br>== Cloud Storage > machinesScalable, high-performance virtual Aeffectivepowerful, objectsimplestorageand  servicecost- APostgreSQLfully managed and SQLMySQL, Server Fullyfor deployingmanaged andcompute scalingplatform<br>database service containerised applications quickly<br>HH svec network > and securely<br>©__ Kubernetes Engine > a| 8<br><!-- End of picture text -->

**3.** If you have used GCP before and your free trial has expired, then the services you will use in this chapter will cost you some money. It shouldn’t be too much, especially if you remember to turn off the services when you don’t need them anymore. Make sure you understand and agree to the pricing conditions before you run any service. I hereby decline any responsibility if services end up costing more than you expected! Also make sure your billing account is active. To check, open the ☰ navigation menu at the top left and click Billing, then make sure you have set up a payment method and that the billing account is active. 

**4.** Every resource in GCP belongs to a _project_ . This includes all the virtual machines you may use, the files you store, and the training jobs you run. When you create an account, GCP automatically creates a project for you, called “My First Project”. If you want, you can change its display name by going to the project settings: in the ☰ navigation menu, select “IAM and admin → Settings”, change the project’s display name, and click SAVE. Note that the project also has a unique ID and number. You can choose the project ID when you create a project, but you cannot change it later. The project number is automatically generated and cannot be changed. If you want to create a new project, click the project name at the top of the page, then click NEW PROJECT and enter the project name. You can also click EDIT to set the project ID. Make sure billing is active for this new project so that service fees can be billed (to your free credits, if any). 



Always set an alarm to remind yourself to turn services off when you know you will only need them for a few hours, or else you might leave them running for days or months, incurring potentially significant costs. 

**5.** Now that you have a GCP account and a project, and billing is activated, you must activate the APIs you need. In the ☰ navigation menu, select “APIs and services”, and make sure the Cloud Storage API is enabled. If needed, click + ENABLE APIS AND SERVICES, find Cloud Storage, and enable it. Also enable the Vertex AI API. 

You could continue to do everything via the GCP console, but I recommend using Python instead: this way you can write scripts to automate just about anything you want with GCP, and it’s often more convenient than clicking your way through menus and forms, especially for common tasks. 

**Serving a TensorFlow Model | 733** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

##### **Google Cloud CLI and Shell** 

Google Cloud’s command-line interface (CLI) includes the `gcloud` command, which lets you control almost everything in GCP, and `gsutil` , which lets you interact with Google Cloud Storage. This CLI is preinstalled in Colab: all you need to do is authenticate using `google.auth.authenticate_user()` , and you’re good to go. For example, `!gcloud config list` will display the configuration. 

GCP also offers a preconfigured shell environment called the Google Cloud Shell, which you can use directly in your web browser; it runs on a free Linux VM (Debian) with the Google Cloud SDK already preinstalled and configured for you, so there’s no need to authenticate. The Cloud Shell is available anywhere in GCP: just click the Activate Cloud Shell icon at the top right of the page (see Figure 19-4). 



_Figure 19-4. Activating the Google Cloud Shell_ 

If you prefer to install the CLI on your machine, then after installation you need to initialize it by running `gcloud init` : follow the instructions to log in to GCP and grant access to your GCP resources, then select the default GCP project you want to use (if you have more than one) and the default region where you want your jobs to run. 

The first thing you need to do before you can use any GCP service is to authenticate. The simplest solution when using Colab is to execute the following code: 

```
fromgoogle.colabimportauth
```

```
auth.authenticate_user()
```

The authentication process is based on _OAuth 2.0_ : a pop-up window will ask you to confirm that you want the Colab notebook to access your Google credentials. If you accept, you must select the same Google account you used for GCP. Then you will be asked to confirm that you agree to give Colab full access to all your data on Google Drive and in GCP. If you allow access, only the current notebook will have access, and only until the Colab runtime expires. Obviously, you should only accept this if you trust the code in the notebook. 

**734 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



If you are _not_ working with the official notebooks from _https:// github.com/ageron/handson-ml3_ , then you should be extra careful: if the notebook’s author is mischievous, they could include code to do whatever they want with your data. 

##### **Authentication and Authorization on GCP** 

In general, using OAuth 2.0 authentication is only recommended when an application must access the user’s personal data or resources from another application, on the user’s behalf. For example, some applications allow the user to save data to their Google Drive, but for that the application first needs the user to authenticate with Google and allow access to Google Drive. In general, the application will only ask for the level of access it needs; it won’t be an unlimited access: for example, the application will only request access to Google Drive, not Gmail or any other Google service. Moreover, the authorization usually expires after a while, and it can always be revoked. 

When an application needs to access a service on GCP on its own behalf, not on behalf of the user, then it should generally use a _service account_ . For example, if you build a website that needs to send prediction requests to a Vertex AI endpoint, then the website will be accessing the service on its own behalf. There’s no data or resource that it needs to access in the user’s Google account. In fact, many users of the website will not even have a Google account. For this scenario, you first need to create a service account. Select “IAM and admin → Service accounts” in the GCP console’s ☰ navigation menu (or use the search box), then click + CREATE SERVICE ACCOUNT, fill in the first page of the form (service account name, ID, description), and click CREATE AND CONTINUE. Next, you must give this account some access rights. Select the “Vertex AI user” role: this will allow the service account to make predictions and use other Vertex AI services, but nothing else. Click CONTINUE. You can now optionally grant some users access to the service account: this is useful when your GCP user account is part of an organization and you wish to authorize other users in the organization to deploy applications that will be based on this service account, or to manage the service account itself. Next, click DONE. 

Once you have created a service account, your application must authenticate as that service account. There are several ways to do that. If your application is hosted on GCP—for example, if you are coding a website hosted on Google Compute Engine—then the simplest and safest solution is to attach the service account to the GCP resource that hosts your website, such as a VM instance or a Google App Engine service. This can be done when creating the GCP resource, by selecting the service account in the “Identity and API access” section. Some resources, such as VM instances, also let you attach the service account after the VM instance is created: you must stop it and edit its settings. In any case, once a service account is attached to a VM instance, or any other GCP resource running your code, GCP’s client libraries 

**Serving a TensorFlow Model | 735** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

(discussed shortly) will automatically authenticate as the chosen service account, with no extra step needed. 

If your application is hosted using Kubernetes, then you should use Google’s Work‐ load Identity service to map the right service account to each Kubernetes service account. If your application is not hosted on GCP—for example, if you are just running the Jupyter notebook on your own machine—then you can either use the Workload Identity Federation service (that’s the safest but hardest option), or just generate an access key for your service account, save it to a JSON file, and point the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to it so your client applica‐ tion can access it. You can manage access keys by clicking the service account you just created, and then opening the KEYS tab. Make sure to keep the key file secret: it’s like a password for the service account. 

For more details on setting up authentication and authorization so your application can access GCP services, check out the documentation. 

Now let’s create a Google Cloud Storage bucket to store our SavedModels (a GCS _bucket_ is a container for your data). For this we will use the `google-cloud-storage` library, which is preinstalled in Colab. We first create a `Client` object, which will serve as the interface with GCS, then we use it to create the bucket: 

```
fromgoogle.cloudimportstorage
```

```
project_id="my_project"# change this to your project ID
bucket_name="my_bucket"# change this to a unique bucket name
location="us-central1"
```

```
storage_client=storage.Client(project=project_id)
bucket=storage_client.create_bucket(bucket_name, location=location)
```



If you want to reuse an existing bucket, replace the last line with `bucket = storage_client.bucket(bucket_name)` . Make sure `location` is set to the bucket’s region. 

GCS uses a single worldwide namespace for buckets, so simple names like “machinelearning” will most likely not be available. Make sure the bucket name conforms to DNS naming conventions, as it may be used in DNS records. Moreover, bucket names are public, so do not put anything private in the name. It is common to use your domain name, your company name, or your project ID as a prefix to ensure uniqueness, or simply use a random number as part of the name. 

You can change the region if you want, but be sure to choose one that supports GPUs. Also, you may want to consider the fact that prices vary greatly between regions, 

###### **736 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

some regions produce much more CO₂ than others, some regions do not support all services, and using a single-region bucket improves performance. See Google Cloud’s list of regions and Vertex AI’s documentation on locations for more details. If you are unsure, it might be best to stick with `"us-central1"` . 

Next, let’s upload the _my_mnist_model_ directory to the new bucket. Files in GCS are called _blobs_ (or _objects_ ), and under the hood they are all just placed in the bucket without any directory structure. Blob names can be arbitrary Unicode strings, and they can even contain forward slashes ( `/` ). The GCP console and other tools use these slashes to give the illusion that there are directories. So, when we upload the _my_mnist_model_ directory, we only care about the files, not the directories: 

```
defupload_directory(bucket, dirpath):
dirpath=Path(dirpath)
forfilepathindirpath.glob("**/*"):
iffilepath.is_file():
blob=bucket.blob(filepath.relative_to(dirpath.parent).as_posix())
blob.upload_from_filename(filepath)
```

```
upload_directory(bucket, "my_mnist_model")
```

This function works fine now, but it would be very slow if there were many files to upload. It’s not too hard to speed it up tremendously by multithreading it (see the notebook for an implementation). Alternatively, if you have the Google Cloud CLI, then you can use following command instead: 

```
!gsutil-mcp-rmy_mnist_modelgs://{bucket_name}/
```

Next, let’s tell Vertex AI about our MNIST model. To communicate with Vertex AI, we can use the `google-cloud-aiplatform` library (it still uses the old AI Platform name instead of Vertex AI). It’s not preinstalled in Colab, so we need to install it. After that, we can import the library and initialize it—just to specify some default values for the project ID and the location—then we can create a new Vertex AI model: we specify a display name, the GCS path to our model (in this case the version 0001), and the URL of the Docker container we want Vertex AI to use to run this model. If you visit that URL and navigate up one level, you will find other containers you can use. This one supports TensorFlow 2.8 with a GPU: 

```
fromgoogle.cloudimportaiplatform
```

```
server_image="gcr.io/cloud-aiplatform/prediction/tf2-gpu.2-8:latest"
```

```
aiplatform.init(project=project_id, location=location)
mnist_model=aiplatform.Model.upload(
display_name="mnist",
artifact_uri=f"gs://{bucket_name}/my_mnist_model/0001",
serving_container_image_uri=server_image,
```

```
)
```

**Serving a TensorFlow Model | 737** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

Now let’s deploy this model so we can query it via a gRPC or REST API to make pre‐ dictions. For this we first need to create an _endpoint_ . This is what client applications connect to when they want to access a service. Then we need to deploy our model to this endpoint: 

```
endpoint=aiplatform.Endpoint.create(display_name="mnist-endpoint")
```

```
endpoint.deploy(
mnist_model,
min_replica_count=1,
max_replica_count=5,
machine_type="n1-standard-4",
accelerator_type="NVIDIA_TESLA_K80",
accelerator_count=1
)
```

This code may take a few minutes to run, because Vertex AI needs to set up a virtual machine. In this example, we use a fairly basic machine of type `n1-standard-4` (see _https://homl.info/machinetypes_ for other types). We also use a basic GPU of type `NVIDIA_TESLA_K80` (see _https://homl.info/accelerators_ for other types). If you selected another region than `"us-central1"` , then you may need to change the machine type or the accelerator type to values that are supported in that region (e.g., not all regions have Nvidia Tesla K80 GPUs). 



Google Cloud Platform enforces various GPU quotas, both world‐ wide and per region: you cannot create thousands of GPU nodes without prior authorization from Google. To check your quotas, open “IAM and admin → Quotas” in the GCP console. If some quotas are too low (e.g., if you need more GPUs in a particular region), you can ask for them to be increased; it often takes about 48 hours. 

Vertex AI will initially spawn the minimum number of compute nodes (just one in this case), and whenever the number of queries per second becomes too high, it will spawn more nodes (up to the maximum number you defined, five in this case) and will load-balance the queries between them. If the QPS rate goes down for a while, Vertex AI will stop the extra compute nodes automatically. The cost is therefore directly linked to the load, as well as the machine and accelerator types you selected and the amount of data you store on GCS. This pricing model is great for occasional users and for services with important usage spikes. It’s also ideal for startups: the price remains low until the startup actually starts up. 

**738 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

Congratulations, you have deployed your first model to the cloud! Now let’s query this prediction service: 

```
response=endpoint.predict(instances=X_new.tolist())
```

We first need to convert the images we want to classify to a Python list, as we did earlier when we sent requests to TF Serving using the REST API. The response object contains the predictions, represented as a Python list of lists of floats. Let’s round them to two decimal places and convert them to a NumPy array: 

```
>>> importnumpyasnp
>>> np.round(response.predictions, 2)
array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  ],
       [0.  , 0.  , 0.99, 0.01, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.97, 0.01, 0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.  ]])
```

Yes! We get the exact same predictions as earlier. We now have a nice prediction service running on the cloud that we can query from anywhere securely, and which can automatically scale up or down depending on the number of QPS. When you are done using the endpoint, don’t forget to delete it, to avoid paying for nothing: 

```
endpoint.undeploy_all()  # undeploy all models from the endpoint
endpoint.delete()
```

Now let’s see how to run a job on Vertex AI to make predictions on a potentially very large batch of data. 

#### **Running Batch Prediction Jobs on Vertex AI** 

If we have a large number of predictions to make, then instead of calling our predic‐ tion service repeatedly, we can ask Vertex AI to run a prediction job for us. This does not require an endpoint, only a model. For example, let’s run a prediction job on the first 100 images of the test set, using our MNIST model. For this, we first need to prepare the batch and upload it to GCS. One way to do this is to create a file containing one instance per line, each formatted as a JSON value—this format is called _JSON Lines_ —then pass this file to Vertex AI. So let’s create a JSON Lines file in a new directory, then upload this directory to GCS: 

```
batch_path=Path("my_mnist_batch")
batch_path.mkdir(exist_ok=True)
withopen(batch_path/"my_mnist_batch.jsonl", "w") asjsonl_file:
forimageinX_test[:100].tolist():
jsonl_file.write(json.dumps(image))
jsonl_file.write("\n")
```

```
upload_directory(bucket, batch_path)
```

**Serving a TensorFlow Model | 739** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

Now we’re ready to launch the prediction job, specifying the job’s name, the type and number of machines and accelerators to use, the GCS path to the JSON Lines file we just created, and the path to the GCS directory where Vertex AI will save the model’s predictions: 

```
batch_prediction_job=mnist_model.batch_predict(
job_display_name="my_batch_prediction_job",
machine_type="n1-standard-4",
starting_replica_count=1,
max_replica_count=5,
accelerator_type="NVIDIA_TESLA_K80",
accelerator_count=1,
gcs_source=[f"gs://{bucket_name}/{batch_path.name}/my_mnist_batch.jsonl"],
gcs_destination_prefix=f"gs://{bucket_name}/my_mnist_predictions/",
sync=True# set to False if you don't want to wait for completion
)
```



For large batches, you can split the inputs into multiple JSON Lines files and list them all via the `gcs_source` argument. 

This will take a few minutes, mostly to spawn the compute nodes on Vertex AI. Once this command completes, the predictions will be available in a set of files named something like _prediction.results-00001-of-00002_ . These files use the JSON Lines format by default, and each value is a dictionary containing an instance and its corresponding prediction (i.e., 10 probabilities). The instances are listed in the same order as the inputs. The job also outputs _prediction-errors*_ files, which can be useful for debugging if something goes wrong. We can iterate through all these output files using `batch_prediction_job.iter_outputs()` , so let’s go through all the predictions and store them in a `y_probas` array: 

```
y_probas= []
forblobinbatch_prediction_job.iter_outputs():
if"prediction.results"inblob.name:
forlineinblob.download_as_text().splitlines():
y_proba=json.loads(line)["prediction"]
y_probas.append(y_proba)
```

Now let’s see how good these predictions are: 

```
>>> y_pred=np.argmax(y_probas, axis=1)
>>> accuracy=np.sum(y_pred==y_test[:100]) /100
0.98
```

Nice, 98% accuracy! 

###### **740 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

The JSON Lines format is the default, but when dealing with large instances such as images, it is too verbose. Luckily, the `batch_predict()` method accepts an `instances_format` argument that lets you choose another format if you want. It defaults to `"jsonl"` , but you can change it to `"csv"` , `"tf-record"` , `"tf-recordgzip"` , `"bigquery"` , or `"file-list"` . If you set it to `"file-list"` , then the `gcs_source` argument should point to a text file containing one input filepath per line; for instance, pointing to PNG image files. Vertex AI will read these files as binary, encode them using Base64, and pass the resulting byte strings to the model. This means that you must add a preprocessing layer in your model to parse the Base64 strings, using `tf.io.decode_base64()` . If the files are images, you must then parse the result using a function like `tf.io.decode_image()` or `tf.io.decode_png()` , as discussed in Chapter 13. 

When you’re finished using the model, you can delete it if you want, by running `mnist_model.delete()` . You can also delete the directories you created in your GCS bucket, optionally the bucket itself (if it’s empty), and the batch prediction job: 

```
forprefixin ["my_mnist_model/", "my_mnist_batch/", "my_mnist_predictions/"]:
blobs=bucket.list_blobs(prefix=prefix)
forblobinblobs:
blob.delete()
```

```
bucket.delete()  # if the bucket is empty
batch_prediction_job.delete()
```

You now know how to deploy a model to Vertex AI, create a prediction service, and run batch prediction jobs. But what if you want to deploy your model to a mobile app instead? Or to an embedded device, such as a heating control system, a fitness tracker, or a self-driving car? 

## **Deploying a Model to a Mobile or Embedded Device** 

Machine learning models are not limited to running on big centralized servers with multiple GPUs: they can run closer to the source of data (this is called _edge comput‐ ing_ ), for example in the user’s mobile device or in an embedded device. There are many benefits to decentralizing the computations and moving them toward the edge: it allows the device to be smart even when it’s not connected to the internet, it reduces latency by not having to send data to a remote server and reduces the load on the servers, and it may improve privacy, since the user’s data can stay on the device. 

However, deploying models to the edge has its downsides too. The device’s computing resources are generally tiny compared to a beefy multi-GPU server. A large model may not fit in the device, it may use too much RAM and CPU, and it may take too long to download. As a result, the application may become unresponsive, and the device may heat up and quickly run out of battery. To avoid all this, you need to make 

**Deploying a Model to a Mobile or Embedded Device | 741** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

a lightweight and efficient model, without sacrificing too much of its accuracy. The TFLite library provides several tools<sup>7</sup> to help you deploy your models to the edge, with three main objectives: 

- Reduce the model size, to shorten download time and reduce RAM usage. 

- Reduce the amount of computations needed for each prediction, to reduce latency, battery usage, and heating. 

- Adapt the model to device-specific constraints. 

To reduce the model size, TFLite’s model converter can take a SavedModel and compress it to a much lighter format based on FlatBuffers. This is an efficient crossplatform serialization library (a bit like protocol buffers) initially created by Google for gaming. It is designed so you can load FlatBuffers straight to RAM without any preprocessing: this reduces the loading time and memory footprint. Once the model is loaded into a mobile or embedded device, the TFLite interpreter will execute it to make predictions. Here is how you can convert a SavedModel to a FlatBuffer and save it to a file: _.tflite_ 

```
converter=tf.lite.TFLiteConverter.from_saved_model(str(model_path))
tflite_model=converter.convert()
withopen("my_converted_savedmodel.tflite", "wb") asf:
f.write(tflite_model)
```



You can also save a Keras model directly to a FlatBuffer using `tf.lite.TFLiteConverter.from_keras_model(model)` . 

The converter also optimizes the model, both to shrink it and to reduce its latency. It prunes all the operations that are not needed to make predictions (such as training operations), and it optimizes computations whenever possible; for example, 3 × _a_ + 4 ×_ a_ + 5 × _a_ will be converted to 12 × _a_ . Addtionally, it tries to fuse operations whenever possible. For example, if possible, batch normalization layers end up folded into the previous layer’s addition and multiplication operations. To get a good idea of how much TFLite can optimize a model, download one of the pretrained TFLite models, such as _Inception_V1_quant_ (click _tflite&pb_ ), unzip the archive, then open the excellent Netron graph visualization tool and upload the _.pb_ file to view the original model. It’s a big, complex graph, right? Next, open the optimized _.tflite_ model and marvel at its beauty! 

> 7 Also check out TensorFlow’s Graph Transform Tool for modifying and optimizing computational graphs. 

**742 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



<!-- Start of picture text -->
-15 0.0 +0.8 +15<br>otRe oo oo eo oot Quantized<br>ee HEOONOOM EE a. weighs<br><!-- End of picture text -->

The most effective way to reduce latency and power consumption is to also quantize the activations so that the computations can be done entirely with integers, without the need for any floating-point operations. Even when using the same bit-width (e.g., 32-bit integers instead of 32-bit floats), integer computations use less CPU cycles, consume less energy, and produce less heat. And if you also reduce the bit-width (e.g., down to 8-bit integers), you can get huge speedups. Moreover, some neural network accelerator devices—such as Google’s Edge TPU—can only process integers, so full quantization of both weights and activations is compulsory. This can be done post-training; it requires a calibration step to find the maximum absolute value of the activations, so you need to provide a representative sample of training data to TFLite (it does not need to be huge), and it will process the data through the model and measure the activation statistics required for quantization. This step is typically fast. 

The main problem with quantization is that it loses a bit of accuracy: it is similar to adding noise to the weights and activations. If the accuracy drop is too severe, then you may need to use _quantization-aware training_ . This means adding fake quantiza‐ tion operations to the model so it can learn to ignore the quantization noise during training; the final weights will then be more robust to quantization. Moreover, the calibration step can be taken care of automatically during training, which simplifies the whole process. 

I have explained the core concepts of TFLite, but going all the way to coding a mobile or embedded application would require a dedicated book. Fortunately, some exist: if you want to learn more about building TensorFlow applications for mobile and embedded devices, check out the O’Reilly books _TinyML: Machine Learning with TensorFlow on Arduino and Ultra-Low Power Micro-Controllers_ , by Pete Warden (former lead of the TFLite team) and Daniel Situnayake and _AI and Machine Learning for On-Device Development_ , by Laurence Moroney. 

Now what if you want to use your model in a website, running directly in the user’s browser? 

## **Running a Model in a Web Page** 

Running your machine learning model on the client side, in the user’s browser, rather than on the server side can be useful in many scenarios, such as: 

- When your web application is often used in situations where the user’s connec‐ tivity is intermittent or slow (e.g., a website for hikers), so running the model directly on the client side is the only way to make your website reliable. 

- When you need the model’s responses to be as fast as possible (e.g., for an online game). Removing the need to query the server to make predictions will definitely reduce the latency and make the website much more responsive. 

###### **744 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

- When your web service makes predictions based on some private user data, and you want to protect the user’s privacy by making the predictions on the client side so that the private data never has to leave the user’s machine. 

For all these scenarios, you can use the TensorFlow.js (TFJS) JavaScript library. This library can load a TFLite model and make predictions directly in the user’s browser. For example, the following JavaScript module imports the TFJS library, downloads a pretrained MobileNet model, and uses this model to classify an image and log the predictions. You can play with the code at _https://homl.info/tfjscode_ , using Glitch.com, a website that lets you build web apps in your browser for free; click the PREVIEW button in the lower-right corner of the page to see the code in action: 

```
import"https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest";
import"https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@1.0.0";
```

```
constimage=document.getElementById("image");
mobilenet.load().then(model=> {
model.classify(image).then(predictions=> {
for (vari=0; i<predictions.length; i++) {
letclassName=predictions[i].className
letproba= (predictions[i].probability*100).toFixed(1)
console.log(className+" : "+proba+"%");
        }
    });
});
```

It’s even possible to turn this website into a _progressive web app_ (PWA): this is a website that respects a number of criteria<sup>8</sup> that allow it to be viewed in any browser, and even installed as a standalone app on a mobile device. For example, try visiting _https://homl.info/tfjswpa_ on a mobile device: most modern browsers will ask you whether you would like to add TFJS Demo to your home screen. If you accept, you will see a new icon in your list of applications. Clicking this icon will load the TFJS Demo website inside its own window, just like a regular mobile app. A PWA can even be configured to work offline, by using a _service worker_ : this is a JavaScript module that runs in its own separate thread in the browser and intercepts network requests, allowing it to cache resources so the PWA can run faster, or even entirely offline. It can also deliver push messages, run tasks in the background, and more. PWAs allow you to manage a single code base for the web and for mobile devices. They also make it easier to ensure that all users run the same version of your application. You can play with this TFJS Demo’s PWA code on Glitch.com at _https://homl.info/wpacode_ . 

> 8 For example, a PWA must include icons of various sizes for different mobile devices, it must be served via HTTPS, it must include a manifest file containing metadata such as the name of the app and the background color. 

**Running a Model in a Web Page | 745** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



Check out many more demos of machine learning models running in your browser at _https://tensorflow.org/js/demos_ . 

TFJS also supports training a model directly in your web browser! And it’s actually pretty fast. If your computer has a GPU card, then TFJS can generally use it, even if it’s not an Nvidia card. Indeed, TFJS will use WebGL when it’s available, and since modern web browsers generally support a wide range of GPU cards, TFJS actually supports more GPU cards than regular TensorFlow (which only supports Nvidia cards). 

Training a model in a user’s web browser can be especially useful to guarantee that this user’s data remains private. A model can be trained centrally, and then fine-tuned locally, in the browser, based on that user’s data. If you’re interested in this topic, check out _federated learning_ . 

Once again, doing justice to this topic would require a whole book. If you want to learn more about TensorFlow.js, check out the O’reilly books _Practical Deep Learning for Cloud, Mobile, and Edge_ , by Anirudh Koul et al., or _Learning TensorFlow.js_ , by Gant Laborde. 

Now that you’ve seen how to deploy TensorFlow models to TF Serving, or to the cloud with Vertex AI, or to mobile and embedded devices using TFLite, or to a web browser using TFJS, let’s discuss how to use GPUs to speed up computations. 

## **Using GPUs to Speed Up Computations** 

In Chapter 11 we looked at several techniques that can considerably speed up train‐ ing: better weight initialization, sophisticated optimizers, and so on. But even with all of these techniques, training a large neural network on a single machine with a single CPU can take hours, days, or even weeks, depending on the task. Thanks to GPUs, this training time can be reduced down to minutes or hours. Not only does this save an enormous amount of time, but it also means that you can experiment with various models much more easily, and frequently retrain your models on fresh data. 

In the previous chapters, we used GPU-enabled runtimes on Google Colab. All you have to do for this is select “Change runtime type” from the Runtime menu, and choose the GPU accelerator type; TensorFlow automatically detects the GPU and uses it to speed up computations, and the code is exactly the same as without a GPU. Then, in this chapter you saw how to deploy your models to Vertex AI on multiple GPU-enabled compute nodes: it’s just a matter of selecting the right GPU-enabled Docker image when creating the Vertex AI model, and selecting the desired GPU type when calling `endpoint.deploy()` . But what if you want to buy your own GPU? 

**746 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



<!-- Start of picture text -->
aT<br>ao OS<br>| CPU<br>wil<br><!-- End of picture text -->



<!-- Start of picture text -->
TTT<br><!-- End of picture text -->

```
$ nvidia-smi
Sun Apr 10 04:52:10 2022
```

|`+-----------------------------------------------------------------------------+`<br>`| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |`<br>`|-------------------------------+----------------------+----------------------+`|
|---|
|`| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |`|
|`| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |`<br>`|                               |                      |               MIG M. |`<br>`|===============================+======================+======================|`<br>|
|`|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |`|
|`| N/A   34C    P8     9W /  70W |      3MiB / 15109MiB |      0%      Default |`<br>`|                               |                      |                  N/A |`<br>`+-------------------------------+----------------------+----------------------+`<br>`+-----------------------------------------------------------------------------+`<br>`| Processes:                                                                  |`<br>|
|`|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |`|
|`|        ID   ID                                                   Usage      |`|
|`|=============================================================================|`<br>`|  No running processes found                                                 |`<br>`+-----------------------------------------------------------------------------+`|



To check that TensorFlow actually sees your GPU, run the following commands and make sure the result is not empty: 

```
>>> physical_gpus=tf.config.list_physical_devices("GPU")
>>> physical_gpus
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

#### **Managing the GPU RAM** 

By default TensorFlow automatically grabs almost all the RAM in all available GPUs the first time you run a computation. It does this to limit GPU RAM fragmentation. This means that if you try to start a second TensorFlow program (or any program that requires the GPU), it will quickly run out of RAM. This does not happen as often as you might think, as you will most often have a single TensorFlow program running on a machine: usually a training script, a TF Serving node, or a Jupyter notebook. If you need to run multiple programs for some reason (e.g., to train two different models in parallel on the same machine), then you will need to split the GPU RAM between these processes more evenly. 

**Using GPUs to Speed Up Computations | 749** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



<!-- Start of picture text -->
/gpu:0 |/gpu:1 /gpu:1 |/gpu:0<br>agegea<br>ceased || [Scued| | [Sewell [See<br><!-- End of picture text -->



<!-- Start of picture text -->
ae<br>eel<br><!-- End of picture text -->

```
>>> logical_gpus=tf.config.list_logical_devices("GPU")
>>> logical_gpus
[LogicalDevice(name='/device:GPU:0', device_type='GPU'),
 LogicalDevice(name='/device:GPU:1', device_type='GPU')]
```

Now let’s see how TensorFlow decides which devices it should use to place variables and execute operations. 

#### **Placing Operations and Variables on Devices** 

Keras and tf.data generally do a good job of placing operations and variables where they belong, but you can also place operations and variables manually on each device, if you want more control: 

- You generally want to place the data preprocessing operations on the CPU, and place the neural network operations on the GPUs. 

- GPUs usually have a fairly limited communication bandwidth, so it is important to avoid unnecessary data transfers into and out of the GPUs. 

- Adding more CPU RAM to a machine is simple and fairly cheap, so there’s usually plenty of it, whereas the GPU RAM is baked into the GPU: it is an expensive and thus limited resource, so if a variable is not needed in the next few training steps, it should probably be placed on the CPU (e.g., datasets generally belong on the CPU). 

By default, all variables and all operations will be placed on the first GPU (the one named `"/gpu:0"` ), except for variables and operations that don’t have a GPU kernel:<sup>10</sup> these are placed on the CPU (always named `"/cpu:0"` ). A tensor or variable’s `device` attribute tells you which device it was placed on:<sup>11</sup> 

```
>>> a=tf.Variable([1., 2., 3.])  # float32 variable goes to the GPU
>>> a.device
'/job:localhost/replica:0/task:0/device:GPU:0'
>>> b=tf.Variable([1, 2, 3])  # int32 variable goes to the CPU
>>> b.device
'/job:localhost/replica:0/task:0/device:CPU:0'
```

You can safely ignore the prefix `/job:localhost/replica:0/task:0` for now; we will discuss jobs, replicas, and tasks later in this chapter. As you can see, the first variable was placed on GPU #0, which is the default device. However, the second variable was 

> 10 As we saw in Chapter 12, a kernel is an operation’s implementation for a specific data type and device type. For example, there is a GPU kernel for the `float32 tf.matmul()` operation, but there is no GPU kernel for `int32 tf.matmul()` , only a CPU kernel. 

> 11 You can also use `tf.debugging.set_log_device_placement(True)` to log all device placements. 

###### **752 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



<!-- Start of picture text -->
|<br><!-- End of picture text -->

| 



<!-- Start of picture text -->
'<br>tee eee \ !'<br>"\<br>1'<br>" Inter-o !<br>l1<br>| =ntra-op HHi! cuDNN '''<br>rT SYV '<br>°o O fa<br>Till<br><!-- End of picture text -->

intra-op thread pool and runs the operation across many GPU threads in parallel. Suppose C finishes first. The dependency counters of D and E are decremented and they reach 0, so both operations are pushed to GPU #0’s evaluation queue, and they are executed sequentially. Note that C only gets evaluated once, even though both D and E depend on it. Suppose B finishes next. Then F’s dependency counter is decremented from 4 to 3, and since that’s not 0, it does not run yet. Once A, D, and E are finished, then F’s dependency counter reaches 0, and it is pushed to the CPU’s evaluation queue and evaluated. Finally, TensorFlow returns the requested outputs. 

An extra bit of magic that TensorFlow performs is when the TF function modifies a stateful resource, such as a variable: it ensures that the order of execution matches the order in the code, even if there is no explicit dependency between the statements. For example, if your TF function contains `v.assign_add(1)` followed by `v.assign(v * 2)` , TensorFlow will ensure that these operations are executed in that order. 



You can control the number of threads in the inter-op thread pool by calling `tf.config.threading.set_inter_op_parallel ism_threads()` . To set the number of intra-op threads, use `tf.config.threading.set_intra_op_parallelism_threads()` . This is useful if you do not want TensorFlow to use all the CPU cores or if you want it to be single-threaded.<sup>12</sup> 

With that, you have all you need to run any operation on any device, and exploit the power of your GPUs! Here are some of the things you could do: 

- You could train several models in parallel, each on its own GPU: just write a training script for each model and run them in parallel, setting `CUDA_DEVICE_ORDER` and `CUDA_VISIBLE_DEVICES` so that each script only sees a single GPU device. This is great for hyperparameter tuning, as you can train in parallel multiple models with different hyperparameters. If you have a single machine with two GPUs, and it takes one hour to train one model on one GPU, then training two models in parallel, each on its own dedicated GPU, will take just one hour. Simple! 

- You could train a model on a single GPU and perform all the preprocessing in parallel on the CPU, using the dataset’s `prefetch()` method<sup>13</sup> to prepare the next few batches in advance so that they are ready when the GPU needs them (see Chapter 13). 

> 12 This can be useful if you want to guarantee perfect reproducibility, as I explain in this video, based on TF 1. 

> 13 At the time of writing, it only prefetches the data to the CPU RAM, but use `tf.data.experimental.prefetch _to_device()` to make it prefetch the data and push it to the device of your choice so that the GPU does not waste time waiting for the data to be transferred. 

**Using GPUs to Speed Up Computations | 755** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

- If your model takes two images as input and processes them using two CNNs before joining their outputs,<sup>14</sup> then it will probably run much faster if you place each CNN on a different GPU. 

- You can create an efficient ensemble: just place a different trained model on each GPU so that you can get all the predictions much faster to produce the ensemble’s final prediction. 

But what if you want to speed up training by using multiple GPUs? 

## **Training Models Across Multiple Devices** 

There are two main approaches to training a single model across multiple devices: _model parallelism_ , where the model is split across the devices, and _data parallelism_ , where the model is replicated across every device, and each replica is trained on a different subset of the data. Let’s look at these two options. 

#### **Model Parallelism** 

So far we have trained each neural network on a single device. What if we want to train a single neural network across multiple devices? This requires chopping the model into separate chunks and running each chunk on a different device. Unfortunately, such model parallelism turns out to be pretty tricky, and its effective‐ ness really depends on the architecture of your neural network. For fully connected networks, there is generally not much to be gained from this approach (see Fig‐ ure 19-11). Intuitively, it may seem that an easy way to split the model is to place each layer on a different device, but this does not work because each layer needs to wait for the output of the previous layer before it can do anything. So perhaps you can slice it vertically—for example, with the left half of each layer on one device, and the right part on another device? This is slightly better, since both halves of each layer can indeed work in parallel, but the problem is that each half of the next layer requires the output of both halves, so there will be a lot of cross-device communication (represented by the dashed arrows). This is likely to completely cancel out the benefit of the parallel computation, since cross-device communication is slow (and even more so when the devices are located on different machines). 

> 14 If the two CNNs are identical, then it is called a _Siamese neural network_ . 

**756 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



<!-- Start of picture text -->
o (3)<br>a |[be0 G0<br>Fully connected One layer per device Vertical split<br>neural network Bad!<br><!-- End of picture text -->



<!-- Start of picture text -->
(O00):(00) [O}<br>= =<br>OOQOOOO0O [OOO OOO]<br>Partially connected Vertical split<br>neural network Fairly good!<br><!-- End of picture text -->



<!-- Start of picture text -->
Outputs Outputs...<br>©); Ge<br>2) =" ~<br>a =Ce LA = LA = W = Ld =Se Ww<br>fh} (ChCbCbCbCha<br>es ee ee ee<br>Deep recurrent Time<br>neural network<br><!-- End of picture text -->



<!-- Start of picture text -->
Update Update Update<br>OOO0O0O OOO00O OOO00O0<br>OOO00O0 OOO000O0 OOO00O<br>A B C — Mini-batches<br><!-- End of picture text -->



<!-- Start of picture text -->
Tal! Ta} Ta <):<br>vv"<br>Update<br>OOO00O OOO000 OOO00O<br>OOO00O OCOO00O OOO00O<br>x y y oo<br>A B C — Mini-batches<br><!-- End of picture text -->

**Synchronous updates.** With _synchronous updates_ , the aggregator waits until all gradi‐ ents are available before it computes the average gradients and passes them to the optimizer, which will update the model parameters. Once a replica has finished computing its gradients, it must wait for the parameters to be updated before it can proceed to the next mini-batch. The downside is that some devices may be slower than others, so the fast devices will have to wait for the slow ones at every step, mak‐ ing the whole process as slow as the slowest device. Moreover, the parameters will be copied to every device almost at the same time (immediately after the gradients are applied), which may saturate the parameter servers’ bandwidth. 



To reduce the waiting time at each step, you could ignore the gradi‐ ents from the slowest few replicas (typically ~10%). For example, you could run 20 replicas, but only aggregate the gradients from the fastest 18 replicas at each step, and just ignore the gradients from the last 2. As soon as the parameters are updated, the first 18 replicas can start working again immediately, without having to wait for the 2 slowest replicas. This setup is generally described as having 18 replicas plus 2 _spare replicas_ .<sup>16</sup> 

**Asynchronous updates.** With asynchronous updates, whenever a replica has finished computing the gradients, the gradients are immediately used to update the model parameters. There is no aggregation (it removes the “mean” step in Figure 19-15) and no synchronization. Replicas work independently of the other replicas. Since there is no waiting for the other replicas, this approach runs more training steps per minute. Moreover, although the parameters still need to be copied to every device at every step, this happens at different times for each replica, so the risk of bandwidth saturation is reduced. 

Data parallelism with asynchronous updates is an attractive choice because of its simplicity, the absence of synchronization delay, and its better use of the bandwidth. However, although it works reasonably well in practice, it is almost surprising that it works at all! Indeed, by the time a replica has finished computing the gradients based on some parameter values, these parameters will have been updated several times by other replicas (on average _N_ – 1 times, if there are _N_ replicas), and there is no guarantee that the computed gradients will still be pointing in the right direction (see Figure 19-16). When gradients are severely out of date, they are called _stale gradients_ : they can slow down convergence, introducing noise and wobble effects 

> 16 This name is slightly confusing because it sounds like some replicas are special, doing nothing. In reality, all replicas are equivalent: they all work hard to be among the fastest at each training step, and the losers vary at every step (unless some devices are really slower than others). However, it does mean that if one or two servers crash, training will continue just fine. 

**Training Models Across Multiple Devices | 761** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



<!-- Start of picture text -->
8, CO cost<br>Gradients are ..but they are<br>computed here... applied here<br>'I<br>'<br>8 Oops, goingup hill!<br>oo” Sh SS \/ 8<br>“ot Sen <4<br>\ I / >s Of ‘Ny \<br>/ Se. JN<br>AT, \<br>v Stale<br>Updates by gradients<br>other replicas 9,<br><!-- End of picture text -->

###### **Bandwidth saturation** 

Whether you use synchronous or asynchronous updates, data parallelism with cen‐ tralized parameters still requires communicating the model parameters from the parameter servers to every replica at the beginning of each training step, and the gradients in the other direction at the end of each training step. Similarly, when using the mirrored strategy, the gradients produced by each GPU will need to be shared with every other GPU. Unfortunately, there often comes a point where adding an extra GPU will not improve performance at all because the time spent moving the data into and out of GPU RAM (and across the network in a distributed setup) will outweigh the speedup obtained by splitting the computation load. At that point, adding more GPUs will just worsen the bandwidth saturation and actually slow down training. 

Saturation is more severe for large dense models, since they have a lot of parameters and gradients to transfer. It is less severe for small models (but the parallelization gain is limited) and for large sparse models, where the gradients are typically mostly zeros and so can be communicated efficiently. Jeff Dean, initiator and lead of the Google Brain project, reported typical speedups of 25–40× when distributing computations across 50 GPUs for dense models, and a 300× speedup for sparser models trained across 500 GPUs. As you can see, sparse models really do scale better. Here are a few concrete examples: 

- Neural machine translation: 6× speedup on 8 GPUs 

- Inception/ImageNet: 32× speedup on 50 GPUs 

- RankBrain: 300× speedup on 500 GPUs 

There is plenty of research going on to alleviate the bandwidth saturation issue, with the goal of allowing training to scale linearly with the number of GPUs available. For example, a 2018 paper<sup>18</sup> by a team of researchers from Carnegie Mellon University, Stanford University, and Microsoft Research proposed a system called _PipeDream_ that managed to reduce network communications by over 90%, making it possible to train large models across many machines. They achieved this using a new technique called _pipeline parallelism_ , which combines model parallelism and data parallelism: the model is chopped into consecutive parts, called _stages_ , each of which is trained on a different machine. This results in an asynchronous pipeline in which all machines work in parallel with very little idle time. During training, each stage alternates one round of forward propagation and one round of backpropagation (see Figure 19-17): it pulls a mini-batch from its input queue, processes it, and sends the outputs to the next stage’s input queue, then it pulls one mini-batch of gradients from its gradient 

> 18 Aaron Harlap et al., “PipeDream: Fast and Efficient Pipeline Parallel DNN Training”, arXiv preprint arXiv:1806.03377 (2018). 

**Training Models Across Multiple Devices | 763** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



<!-- Start of picture text -->
(3) Stage 1 Stage 2 Stage 3<br>queue 4, Forward-p/(6 / Forward Forward (5,<br>Backward [ {4 p-Backward (aterradient Backward<q——ILoss<br><!-- End of picture text -->

very well interconnected servers. You can also try dropping the float precision from 32 bits ( `tf.float32` ) to 16 bits ( `tf.bfloat16` ). This will cut in half the amount of data to transfer, often without much impact on the convergence rate or the model’s performance. Lastly, if you are using centralized parameters, you can shard (split) the parameters across multiple parameter servers: adding more parameter servers will reduce the network load on each server and limit the risk of bandwidth saturation. 

OK, now that we’ve gone through all the theory, let’s actually train a model across multiple GPUs! 

#### **Training at Scale Using the Distribution Strategies API** 

Luckily, TensorFlow comes with a very nice API that takes care of all the complexity of distributing your model across multiple devices and machines: the _distribution strategies API_ . To train a Keras model across all available GPUs (on a single machine, for now) using data parallelism with the mirrored strategy, just create a `Mirrored Strategy` object, call its `scope()` method to get a distribution context, and wrap the creation and compilation of your model inside that context. Then call the model’s `fit()` method normally: 

```
strategy=tf.distribute.MirroredStrategy()
```

```
withstrategy.scope():
model=tf.keras.Sequential([...])  # create a Keras model normally
model.compile([...])  # compile the model normally
```

```
batch_size=100# preferably divisible by the number of replicas
model.fit(X_train, y_train, epochs=10,
```

```
validation_data=(X_valid, y_valid), batch_size=batch_size)
```

Under the hood, Keras is distribution-aware, so in this `MirroredStrategy` context it knows that it must replicate all variables and operations across all available GPU devices. If you look at the model’s weights, they are of type `MirroredVariable` : 

```
>>> type(model.weights[0])
tensorflow.python.distribute.values.MirroredVariable
```

Note that the `fit()` method will automatically split each training batch across all the replicas, so it’s preferable to ensure that the batch size is divisible by the number of replicas (i.e., the number of available GPUs) so that all replicas get batches of the same size. And that’s all! Training will generally be significantly faster than using a single device, and the code change was really minimal. 

**Training Models Across Multiple Devices | 765** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

Once you have finished training your model, you can use it to make predictions efficiently: call the `predict()` method, and it will automatically split the batch across all replicas, making predictions in parallel. Again, the batch size must be divisible by the number of replicas. If you call the model’s `save()` method, it will be saved as a regular model, _not_ as a mirrored model with multiple replicas. So when you load it, it will run like a regular model, on a single device: by default on GPU #0, or on the CPU if there are no GPUs. If you want to load a model and run it on all available devices, you must call `tf.keras.models.load_model()` within a distribution context: 

```
withstrategy.scope():
```

```
model=tf.keras.models.load_model("my_mirrored_model")
```

If you only want to use a subset of all the available GPU devices, you can pass the list to the `MirroredStrategy` ’s constructor: 

```
strategy=tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
```

By default, the `MirroredStrategy` class uses the _NVIDIA Collective Communications Library_ (NCCL) for the AllReduce mean operation, but you can change it by setting the `cross_device_ops` argument to an instance of the `tf.distribute.Hierarchical CopyAllReduce` class, or an instance of the `tf.distribute.ReductionToOneDevice` class. The default NCCL option is based on the `tf.distribute.NcclAllReduce` class, which is usually faster, but this depends on the number and types of GPUs, so you may want to give the alternatives a try.<sup>20</sup> 

If you want to try using data parallelism with centralized parameters, replace the `MirroredStrategy` with the `CentralStorageStrategy` : 

```
strategy=tf.distribute.experimental.CentralStorageStrategy()
```

You can optionally set the `compute_devices` argument to specify the list of devices you want to use as workers—by default it will use all available GPUs—and you can optionally set the `parameter_device` argument to specify the device you want to store the parameters on. By default it will use the CPU, or the GPU if there is just one. 

Now let’s see how to train a model across a cluster of TensorFlow servers! 

#### **Training a Model on a TensorFlow Cluster** 

A _TensorFlow cluster_ is a group of TensorFlow processes running in parallel, usually on different machines, and talking to each other to complete some work—for exam‐ ple, training or executing a neural network model. Each TF process in the cluster is called a _task_ , or a _TF server_ . It has an IP address, a port, and a type (also called its 

> 20 For more details on AllReduce algorithms, read Yuichiro Ueno’s post on the technologies behind deep learning and Sylvain Jeaugey’s post on massively scaling deep learning training with NCCL. 

###### **766 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

_role_ or its _job_ ). The type can be either `"worker"` , `"chief"` , `"ps"` (parameter server), or `"evaluator"` : 

- Each _worker_ performs computations, usually on a machine with one or more GPUs. 

- The _chief_ performs computations as well (it is a worker), but it also handles extra work such as writing TensorBoard logs or saving checkpoints. There is a single chief in a cluster. If no chief is specified explicitly, then by convention the first worker is the chief. 

- A _parameter server_ only keeps track of variable values, and it is usually on a CPUonly machine. This type of task is only used with the `ParameterServerStrategy` . 

- An _evaluator_ obviously takes care of evaluation. This type is not used often, and when it’s used, there’s usually just one evaluator. 

To start a TensorFlow cluster, you must first define its specification. This means defining each task’s IP address, TCP port, and type. For example, the following _cluster specification_ defines a cluster with three tasks (two workers and one parameter server; see Figure 19-18). The cluster spec is a dictionary with one key per job, and the values are lists of task addresses ( _IP_ : _port_ ): 

```
cluster_spec= {
"worker": [
"machine-a.example.com:2222",     # /job:worker/task:0
"machine-b.example.com:2222"# /job:worker/task:1
    ],
"ps": ["machine-a.example.com:2221"]  # /job:ps/task:0
}
```

In general there will be a single task per machine, but as this example shows, you can configure multiple tasks on the same machine if you want. In this case, if they share the same GPUs, make sure the RAM is split appropriately, as discussed earlier. 



By default, every task in the cluster may communicate with every other task, so make sure to configure your firewall to authorize all communications between these machines on these ports (it’s usually simpler if you use the same port on every machine). 

**Training Models Across Multiple Devices | 767** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



<!-- Start of picture text -->
Job “ps” Job “worker”<br>| Task 0 | | Task 0 Task 1 ]<br>tcp:2221 tcp:2222 tcp:2222<br>Machine B<br>SiaLT) ¢zSnr<br><!-- End of picture text -->



```
importtempfile
importtensorflowastf
```

```
strategy=tf.distribute.MultiWorkerMirroredStrategy()  # at the start!
resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver()
print(f"Starting task {resolver.task_type} #{resolver.task_id}")
[...] # load and split the MNIST dataset
```

```
withstrategy.scope():
model=tf.keras.Sequential([...])  # build the Keras model
model.compile([...])  # compile the model
```

```
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10)
```

```
ifresolver.task_id==0:  # the chief saves the model to the right location
model.save("my_mnist_multiworker_model", save_format="tf")
else:
```

```
tmpdir=tempfile.mkdtemp()  # other workers save to a temporary directory
model.save(tmpdir, save_format="tf")
tf.io.gfile.rmtree(tmpdir)  # and we can delete this directory at the end!
```

That’s almost the same code you used earlier, except this time you are using the `MultiWorkerMirroredStrategy` . When you start this script on the first workers, they will remain blocked at the AllReduce step, but training will begin as soon as the last worker starts up, and you will see them all advancing at exactly the same rate since they synchronize at each step. 



When using the `MultiWorkerMirroredStrategy` , it’s important to ensure that all workers do the same thing, including saving model checkpoints or writing TensorBoard logs, even though you will only keep what the chief writes. This is because these operations may need to run the AllReduce operations, so all workers must be in sync. 

There are two AllReduce implementations for this distribution strategy: a ring All‐ Reduce algorithm based on gRPC for the network communications, and NCCL’s implementation. The best algorithm to use depends on the number of workers, the number and types of GPUs, and the network. By default, TensorFlow will apply some heuristics to select the right algorithm for you, but you can force NCCL (or RING) like this: 

```
strategy=tf.distribute.MultiWorkerMirroredStrategy(
communication_options=tf.distribute.experimental.CommunicationOptions(
implementation=tf.distribute.experimental.CollectiveCommunication.NCCL))
```

**Training Models Across Multiple Devices | 769** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

If you prefer to implement asynchronous data parallelism with parameter servers, change the strategy to `ParameterServerStrategy` , add one or more parameter servers, and configure `TF_CONFIG` appropriately for each task. Note that although the workers will work asynchronously, the replicas on each worker will work synchronously. 

Lastly, if you have access to TPUs on Google Cloud—for example, if you use Colab and you set the accelerator type to TPU—then you can create a `TPUStrategy` like this: 

```
resolver=tf.distribute.cluster_resolver.TPUClusterResolver()
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy=tf.distribute.experimental.TPUStrategy(resolver)
```

This needs to be run right after importing TensorFlow. You can then use this strategy normally. 



If you are a researcher, you may be eligible to use TPUs for free; see _https://tensorflow.org/tfrc_ for more details. 

You can now train models across multiple GPUs and multiple servers: give yourself a pat on the back! If you want to train a very large model, however, you will need many GPUs, across many servers, which will require either buying a lot of hardware or managing a lot of cloud virtual machines. In many cases, it’s less hassle and less expensive to use a cloud service that takes care of provisioning and managing all this infrastructure for you, just when you need it. Let’s see how to do that using Vertex AI. 

#### **Running Large Training Jobs on Vertex AI** 

Vertex AI allows you to create custom training jobs with your own training code. In fact, you can use almost the same training code as you would use on your own TF cluster. The main thing you must change is where the chief should save the model, the checkpoints, and the TensorBoard logs. Instead of saving the model to a local directory, the chief must save it to GCS, using the path provided by Vertex AI in the `AIP_MODEL_DIR` environment variable. For the model checkpoints and TensorBoard logs, you should use the paths contained in the `AIP_CHECKPOINT_DIR` and `AIP_TENSORBOARD_LOG_DIR` environment variables, respectively. Of course, you must also make sure that the training data can be accessed from the virtual machines, such as on GCS, or another GCP service like BigQuery, or directly from the web. Lastly, Vertex AI sets the `"chief"` task type explicitly, so you should identify the chief using `resolved.task_type == "chief"` instead of `resolved.task_id == 0` : 

**770 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

```
importos
[...]  # other imports, create MultiWorkerMirroredStrategy, and resolver
ifresolver.task_type=="chief":
model_dir=os.getenv("AIP_MODEL_DIR")  # paths provided by Vertex AI
tensorboard_log_dir=os.getenv("AIP_TENSORBOARD_LOG_DIR")
checkpoint_dir=os.getenv("AIP_CHECKPOINT_DIR")
else:
tmp_dir=Path(tempfile.mkdtemp())  # other workers use temporary dirs
model_dir=tmp_dir/"model"
tensorboard_log_dir=tmp_dir/"logs"
checkpoint_dir=tmp_dir/"ckpt"
callbacks= [tf.keras.callbacks.TensorBoard(tensorboard_log_dir),
tf.keras.callbacks.ModelCheckpoint(checkpoint_dir)]
[...]  # build and  compile using the strategy scope, just like earlier
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10,
callbacks=callbacks)
model.save(model_dir, save_format="tf")
```



If you place the training data on GCS, you can create a `tf.data.TextLineDataset` or `tf.data.TFRecordDataset` to access it: just use the GCS paths as the filenames (e.g., _gs://my_bucket/ data/001.csv_ ). These datasets rely on the `tf.io.gfile` package to access files: it supports both local files and GCS files. 

Now you can create a custom training job on Vertex AI, based on this script. You’ll need to specify the job name, the path to your training script, the Docker image to use for training, the one to use for predictions (after training), any additional Python libraries you may need, and lastly the bucket that Vertex AI should use as a staging directory to store the training script. By default, that’s also where the training script will save the trained model, as well as the TensorBoard logs and model checkpoints (if any). Let’s create the job: 

```
custom_training_job=aiplatform.CustomTrainingJob(
display_name="my_custom_training_job",
script_path="my_vertex_ai_training_task.py",
container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-4:latest",
model_serving_container_image_uri=server_image,
requirements=["gcsfs==2022.3.0"],  # not needed, this is just an example
staging_bucket=f"gs://{bucket_name}/staging"
)
```

**Training Models Across Multiple Devices | 771** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 



<!-- Start of picture text -->
|<br><!-- End of picture text -->

| 

```
importargparse
```

```
parser=argparse.ArgumentParser()
parser.add_argument("--n_hidden", type=int, default=2)
parser.add_argument("--n_neurons", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--optimizer", default="adam")
args=parser.parse_args()
```

The hyperparameter tuning service will call your script multiple times, each time with different hyperparameter values: each run is called a _trial_ , and the set of trials is called a _study_ . Your training script must then use the given hyperparameter values to build and compile a model. You can use a mirrored distribution strategy if you want, in case each trial runs on a multi-GPU machine. Then the script can load the dataset and train the model. For example: 

```
importtensorflowastf
```

```
defbuild_model(args):
withtf.distribute.MirroredStrategy().scope():
model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28, 28], dtype=tf.uint8))
for_inrange(args.n_hidden):
model.add(tf.keras.layers.Dense(args.n_neurons, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
opt=tf.keras.optimizers.get(args.optimizer)
opt.learning_rate=args.learning_rate
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])
returnmodel
[...]  # load the dataset
model=build_model(args)
history=model.fit([...])
```



You can use the `AIP_*` environment variables we mentioned earlier to determine where to save the checkpoints, the TensorBoard logs, and the final model. 

Lastly, the script must report the model’s performance back to Vertex AI’s hyperpara‐ meter tuning service, so it can decide which hyperparameters to try next. For this, you must use the `hypertune` library, which is automatically installed on Vertex AI training VMs: 

**Training Models Across Multiple Devices | 773** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

```
importhypertune
```

```
hypertune=hypertune.HyperTune()
hypertune.report_hyperparameter_tuning_metric(
hyperparameter_metric_tag="accuracy",  # name of the reported metric
metric_value=max(history.history["val_accuracy"]),  # metric value
global_step=model.optimizer.iterations.numpy(),
)
```

Now that your training script is ready, you need to define the type of machine you would like to run it on. For this, you must define a custom job, which Vertex AI will use as a template for each trial: 

```
trial_job=aiplatform.CustomJob.from_local_script(
display_name="my_search_trial_job",
script_path="my_vertex_ai_trial.py",  # path to your training script
container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-4:latest",
staging_bucket=f"gs://{bucket_name}/staging",
accelerator_type="NVIDIA_TESLA_K80",
accelerator_count=2,  # in this example, each trial will have 2 GPUs
)
```

Finally, you’re ready to create and run the hyperparameter tuning job: 

```
fromgoogle.cloud.aiplatformimporthyperparameter_tuningashpt
```

```
hp_job=aiplatform.HyperparameterTuningJob(
display_name="my_hp_search_job",
custom_job=trial_job,
metric_spec={"accuracy": "maximize"},
parameter_spec={
"learning_rate": hpt.DoubleParameterSpec(min=1e-3, max=10, scale="log"),
"n_neurons": hpt.IntegerParameterSpec(min=1, max=300, scale="linear"),
"n_hidden": hpt.IntegerParameterSpec(min=1, max=10, scale="linear"),
"optimizer": hpt.CategoricalParameterSpec(["sgd", "adam"]),
    },
max_trial_count=100,
parallel_trial_count=20,
)
hp_job.run()
```

Here, we tell Vertex AI to maximize the metric named `"accuracy"` : this name must match the name of the metric reported by the training script. We also define the search space, using a log scale for the learning rate and a linear (i.e., uniform) scale for the other hyperparameters. The hyperparameter names must match the command-line arguments of the training script. Then we set the maximum number of trials to 100, and the maximum number of trials running in parallel to 20. If you increase the number of parallel trials to (say) 60, the total search time will be reduced significantly, by a factor of up to 3. But the first 60 trials will be started in parallel, so they will not benefit from the other trials’ feedback. Therefore, you should increase the max number of trials to compensate—for example, up to about 140. 

###### **774 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

This will take quite a while. Once the job is completed, you can fetch the trial results using `hp_job.trials` . Each trial result is represented as a protobuf object, containing the hyperparameter values and the resulting metrics. Let’s find the best trial: 

```
defget_final_metric(trial, metric_id):
formetricintrial.final_measurement.metrics:
ifmetric.metric_id==metric_id:
returnmetric.value
```

```
trials=hp_job.trials
trial_accuracies= [get_final_metric(trial, "accuracy") fortrialintrials]
best_trial=trials[np.argmax(trial_accuracies)]
```

Now let’s look at this trial’s accuracy, and its hyperparameter values: 

```
>>> max(trial_accuracies)
0.977400004863739
>>> best_trial.id
'98'
>>> best_trial.parameters
[parameter_id: "learning_rate" value { number_value: 0.001 },
 parameter_id: "n_hidden" value { number_value: 8.0 },
 parameter_id: "n_neurons" value { number_value: 216.0 },
 parameter_id: "optimizer" value { string_value: "adam" }
]
```

That’s it! Now you can get this trial’s SavedModel, optionally train it a bit more, and deploy it to production. 



Vertex AI also includes an AutoML service, which completely takes care of finding the right model architecture and training it for you. All you need to do is upload your dataset to Vertex AI using a special format that depends on the type of dataset (images, text, tabular, video, etc.), then create an AutoML training job, pointing to the dataset and specifying the maximum number of compute hours you’re willing to spend. See the notebook for an example. 

##### **Hyperparameter Tuning Using Keras Tuner on Vertex AI** 

Instead of using Vertex AI’s hyperparameter tuning service, you can use Keras Tuner (introduced in Chapter 10) and run it on Vertex AI VMs. Keras Tuner provides a sim‐ ple way to scale hyperparameter search by distributing it across multiple machines: it only requires setting three environment variables on each machine, then running your regular Keras Tuner code on each machine. You can use the exact same script on all machines. One of the machines acts as the chief (i.e., the oracle), and the others act as workers. Each worker asks the chief which hyperparameter values to try, then the worker trains the model using these hyperparameter values, and finally it reports the 

**Training Models Across Multiple Devices | 775** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

model’s performance back to the chief, which can then decide which hyperparameter values the worker should try next. 

The three environment variables you need to set on each machine are: 

###### `KERASTUNER_TUNER_ID` 

This is equal to `"chief"` on the chief machine, or a unique identifier on each worker machine, such as `"worker0"` , `"worker1"` , etc. 

###### `KERASTUNER_ORACLE_IP` 

This is the IP address or hostname of the chief machine. The chief itself should generally use `"0.0.0.0"` to listen on every IP address on the machine. 

###### `KERASTUNER_ORACLE_PORT` 

This is the TCP port that the chief will be listening on. 

You can distribute Keras Tuner across any set of machines. If you want to run it on Vertex AI machines, then you can spawn a regular training job, and just modify the training script to set the environment variables properly before using Keras Tuner. See the notebook for an example. 

Now you have all the tools and knowledge you need to create state-of-the-art neural net architectures and train them at scale using various distribution strategies, on your own infrastructure or on the cloud, and then deploy them anywhere. In other words, you now have superpowers: use them well! 

## **Exercises** 

**1.** What does a SavedModel contain? How do you inspect its content? 

**2.** When should you use TF Serving? What are its main features? What are some tools you can use to deploy it? 

**3.** How do you deploy a model across multiple TF Serving instances? 

**4.** When should you use the gRPC API rather than the REST API to query a model served by TF Serving? 

**5.** What are the different ways TFLite reduces a model’s size to make it run on a mobile or embedded device? 

**6.** What is quantization-aware training, and why would you need it? 

**7.** What are model parallelism and data parallelism? Why is the latter generally recommended? 

**8.** When training a model across multiple servers, what distribution strategies can you use? How do you choose which one to use? 

###### **776 | Chapter 19: Training and Deploying TensorFlow Models at Scale** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

**9.** Train a model (any model you like) and deploy it to TF Serving or Google Vertex AI. Write the client code to query it using the REST API or the gRPC API. Update the model and deploy the new version. Your client code will now query the new version. Roll back to the first version. 

**10.** Train any model across multiple GPUs on the same machine using the `Mirrored Strategy` (if you do not have access to GPUs, you can use Google Colab with a GPU runtime and create two logical GPUs). Train the model again using the `CentralStorageStrategy` and compare the training time. 

**11.** Fine-tune a model of your choice on Vertex AI, using either Keras Tuner or Vertex AI’s hyperparameter tuning service. 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

## **Thank You!** 

Before we close the last chapter of this book, I would like to thank you for reading it up to the last paragraph. I truly hope that you had as much fun reading this book as I had writing it, and that it will be useful for your projects, big or small. 

If you find errors, please send feedback. More generally, I would love to know what you think, so please don’t hesitate to contact me via O’Reilly, through the _ageron/ handson-ml3_ GitHub project, or on Twitter at @aureliengeron. 

Going forward, my best advice to you is to practice and practice: try going through all the exercises (if you have not done so already), play with the notebooks, join Kaggle or some other ML community, watch ML courses, read papers, attend conferences, and meet experts. Things move fast, so try to keep up to date. Several YouTube channels regularly present deep learning papers in great detail, in a very approachable way. I particularly recommend the channels by Yannic Kilcher, Letitia Parcalabescu, and Xander Steenbrugge. For fascinating ML discussions and higher-level insights, make sure to check out ML Street Talk, and Lex Fridman’s channel. It also helps tremendously to have a concrete project to work on, whether it is for work or for fun (ideally for both), so if there’s anything you have always dreamed of building, give it a shot! Work incrementally; don’t shoot for the moon right away, but stay focused on your project and build it piece by piece. It will require patience and perseverance, but when you have a walking robot, or a working chatbot, or whatever else you fancy building, it will be immensely rewarding! 

My greatest hope is that this book will inspire you to build a wonderful ML applica‐ tion that will benefit all of us. What will it be? 

###### **_—Aurélien Géron_** 

**Thank You! | 777** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:26. 

