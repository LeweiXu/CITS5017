### **CHAPTER 10 Introduction to Artificial Neural Networks with Keras** 

Birds inspired us to fly, burdock plants inspired Velcro, and nature has inspired countless more inventions. It seems only logical, then, to look at the brain’s architec‐ ture for inspiration on how to build an intelligent machine. This is the logic that sparked _artificial neural networks_ (ANNs), machine learning models inspired by the networks of biological neurons found in our brains. However, although planes were inspired by birds, they don’t have to flap their wings to fly. Similarly, ANNs have gradually become quite different from their biological cousins. Some researchers even argue that we should drop the biological analogy altogether (e.g., by saying “units” rather than “neurons”), lest we restrict our creativity to biologically plausible systems.<sup>1</sup> 

ANNs are at the very core of deep learning. They are versatile, powerful, and scalable, making them ideal to tackle large and highly complex machine learning tasks such as classifying billions of images (e.g., Google Images), powering speech recognition services (e.g., Apple’s Siri), recommending the best videos to watch to hundreds of millions of users every day (e.g., YouTube), or learning to beat the world champion at the game of Go (DeepMind’s AlphaGo). 

The first part of this chapter introduces artificial neural networks, starting with a quick tour of the very first ANN architectures and leading up to multilayer percep‐ trons, which are heavily used today (other architectures will be explored in the next chapters). In the second part, we will look at how to implement neural networks using TensorFlow’s Keras API. This is a beautifully designed and simple high-level 

> 1 You can get the best of both worlds by being open to biological inspirations without being afraid to create biologically unrealistic models, as long as they work well. 

**299** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

API for building, training, evaluating, and running neural networks. But don’t be fooled by its simplicity: it is expressive and flexible enough to let you build a wide variety of neural network architectures. In fact, it will probably be sufficient for most of your use cases. And should you ever need extra flexibility, you can always write custom Keras components using its lower-level API, or even use TensorFlow directly, as you will see in Chapter 12. 

But first, let’s go back in time to see how artificial neural networks came to be! 

#### **From Biological to Artificial Neurons** 

Surprisingly, ANNs have been around for quite a while: they were first introduced back in 1943 by the neurophysiologist Warren McCulloch and the mathematician Walter Pitts. In their landmark paper<sup>2</sup> “A Logical Calculus of Ideas Immanent in Nervous Activity”, McCulloch and Pitts presented a simplified computational model of how biological neurons might work together in animal brains to perform complex computations using _propositional logic_ . This was the first artificial neural network architecture. Since then many other architectures have been invented, as you will see. 

The early successes of ANNs led to the widespread belief that we would soon be conversing with truly intelligent machines. When it became clear in the 1960s that this promise would go unfulfilled (at least for quite a while), funding flew elsewhere, and ANNs entered a long winter. In the early 1980s, new architectures were invented and better training techniques were developed, sparking a revival of interest in _connectionism_ , the study of neural networks. But progress was slow, and by the 1990s other powerful machine learning techniques had been invented, such as support vector machines (see Chapter 5). These techniques seemed to offer better results and stronger theoretical foundations than ANNs, so once again the study of neural networks was put on hold. 

We are now witnessing yet another wave of interest in ANNs. Will this wave die out like the previous ones did? Well, here are a few good reasons to believe that this time is different and that the renewed interest in ANNs will have a much more profound impact on our lives: 

- There is now a huge quantity of data available to train neural networks, and ANNs frequently outperform other ML techniques on very large and complex problems. 

- The tremendous increase in computing power since the 1990s now makes it possible to train large neural networks in a reasonable amount of time. This is 

- 2 Warren S. McCulloch and Walter Pitts, “A Logical Calculus of the Ideas Immanent in Nervous Activity”, _The Bulletin of Mathematical Biology_ 5, no. 4 (1943): 115–113. 

**300 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

in part due to Moore’s law (the number of components in integrated circuits has doubled about every 2 years over the last 50 years), but also thanks to the gaming industry, which has stimulated the production of powerful GPU cards by the millions. Moreover, cloud platforms have made this power accessible to everyone. 

- The training algorithms have been improved. To be fair they are only slightly different from the ones used in the 1990s, but these relatively small tweaks have had a huge positive impact. 

- Some theoretical limitations of ANNs have turned out to be benign in practice. For example, many people thought that ANN training algorithms were doomed because they were likely to get stuck in local optima, but it turns out that this is not a big problem in practice, especially for larger neural networks: the local optima often perform almost as well as the global optimum. 

- ANNs seem to have entered a virtuous circle of funding and progress. Amazing products based on ANNs regularly make the headline news, which pulls more and more attention and funding toward them, resulting in more and more progress and even more amazing products. 

##### **Biological Neurons** 

Before we discuss artificial neurons, let’s take a quick look at a biological neuron (represented in Figure 10-1). It is an unusual-looking cell mostly found in animal brains. It’s composed of a _cell body_ containing the nucleus and most of the cell’s complex components, many branching extensions called _dendrites_ , plus one very long extension called the _axon_ . The axon’s length may be just a few times longer than the cell body, or up to tens of thousands of times longer. Near its extremity the axon splits off into many branches called _telodendria_ , and at the tip of these branches are minus‐ cule structures called _synaptic terminals_ (or simply _synapses_ ), which are connected to the dendrites or cell bodies of other neurons.<sup>3</sup> Biological neurons produce short electrical impulses called _action potentials_ (APs, or just _signals_ ), which travel along the axons and make the synapses release chemical signals called _neurotransmitters_ . When a neuron receives a sufficient amount of these neurotransmitters within a few milliseconds, it fires its own electrical impulses (actually, it depends on the neurotransmitters, as some of them inhibit the neuron from firing). 

> 3 They are not actually attached, just so close that they can very quickly exchange chemical signals. 

**From Biological to Artificial Neurons | 301** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 



<!-- Start of picture text -->
Cell body<br>Axon Telodendria c<br>Nucleus LTS<br>(S aS Axon hillock Synaptic terminals<br>a<br>Endoplasmic 7 ) Golgi apparatus<br>reticulum<br>. Q<br>Mitochondrion Dendrite<br>Dendritic branches<br><!-- End of picture text -->



<!-- Start of picture text -->
PRAa SSll SR¥ ———é eaSSae= oe> wee ee. Sarr.“3ae)oebe SSSeneeieeTye.eeERY.i8SteeoFSe: 5eens=aeaesONBMAee RS) eSP Le eaeNOD = arstt—<br>ee ae ee ~a gee— SS—ae.=  Soeay ete  ae. caeSO eyry  tm. ttWag as SET ee<br>22)PathogNE& 0 ae eereeSe SanSeei Taskne es ae eeeVelSAGhamt LAYSLD NOUSse 6S TSae ;<br>ONLN eeeSaenaaES|Ee eet a AR1s EANN a 9oe a ee<br>bysFlths Pe 2SSSeeSSSSe eSSO SeSSaeeeneenein.FSA aSeaeeeMEW).  Ba Aghe RSYoOn te+ RV O R,SS aaaeS gee Baea<br>(ightYe‘SoliseiOF  SoaESSoeieaseBtSSSSaae—ee SOee——ooa = eeaSeseaee= RReeaeSSEeenereSNSHS SY— Aane NO=aeTheeeeSSOPER . a= SshSESSeeSra a SSRNaaarnesyea iite  Aehy BieesSN Oa es =P tya—= sD—<br><!-- End of picture text -->



<!-- Start of picture text -->
Neurons Connection A=And \<br>4 / | V=0r i}<br>all ‘ Lye 5<br>CA C=AAB C=AvVB C=AA-B<br><!-- End of picture text -->



<!-- Start of picture text -->
Output: hy, »,0d=step(w'x+h)<br>(S75) Step function: step(z)<br>ee Linear function: z=w!x+b<br>@ (Ww) @ Weights<br>x; X> x3; Inputs<br><!-- End of picture text -->



<!-- Start of picture text -->
QO | () |<br><!-- End of picture text -->



<!-- Start of picture text -->
Outputs<br>TLU — ES — Output layer<br>Ki. \<br>Xx X> j Input layer<br>Inputs /<br>_7<br><!-- End of picture text -->

( ) 

In this equation: 

- **Ŷ** is the output matrix. It has one row per instance and one column per output feature. 

- **X** represents the matrix of input features. It also has one row per instance and one column per input feature. 

- The weight matrix **W** contains all the connection weights. It has one row per input and one column per neuron. 

- The bias vector **b** contains all the bias terms: one per neuron. 

- The function ϕ is called the _activation function_ : when the artificial neurons are TLUs, it is a step function (we will discuss other activation functions shortly). 



In mathematics, the sum of a matrix and a vector is undefined. However, in data science, we allow “broadcasting”: adding a vector to a matrix means adding it to every row in the matrix. So, **XW** + **b** first multiplies **X** by **W** —which results in a matrix with one row per instance and one column per output—then adds the vector **b** to every row of that matrix, which adds each bias term to the corre‐ sponding output, for every instance. Moreover, ϕ is then applied itemwise to each item in the resulting matrix. 

So, how is a perceptron trained? The perceptron training algorithm proposed by Rosenblatt was largely inspired by _Hebb’s rule_ . In his 1949 book _The Organization of Behavior_ (Wiley), Donald Hebb suggested that when a biological neuron triggers another neuron often, the connection between these two neurons grows stronger. Sie‐ grid Löwel later summarized Hebb’s idea in the catchy phrase, “Cells that fire together, wire together”; that is, the connection weight between two neurons tends to increase when they fire simultaneously. This rule later became known as Hebb’s rule (or _Hebbian learning_ ). Perceptrons are trained using a variant of this rule that takes into account the error made by the network when it makes a prediction; the perceptron learning rule reinforces connections that help reduce the error. More specifically, the perceptron is fed one training instance at a time, and for each instance it makes its predictions. For every output neuron that produced a wrong prediction, it reinforces the connection weights from the inputs that would have contributed to the correct prediction. The rule is shown in Equation 10-3. 

###### _Equation 10-3. Perceptron learning rule (weight update)_ 

wi , j next step = wi , j + η yj − y j xi 

**306 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

In this equation: 

- _wi_ , _j_ is the connection weight between the _i_<sup>th</sup> input and the _j_<sup>th</sup> neuron. 

- _xi_ is the _i_<sup>th</sup> input value of the current training instance. 

- y j<sup>is the output of the</sup><sup>_j_th output neuron for the current training instance.</sup> 

- _yj_ is the target output of the _j_<sup>th</sup> output neuron for the current training instance. 

- _η_ is the learning rate (see Chapter 4). 

The decision boundary of each output neuron is linear, so perceptrons are incapable of learning complex patterns (just like logistic regression classifiers). However, if the training instances are linearly separable, Rosenblatt demonstrated that this algorithm would converge to a solution.<sup>7</sup> This is called the _perceptron convergence theorem_ . 

Scikit-Learn provides a `Perceptron` class that can be used pretty much as you would expect—for example, on the iris dataset (introduced in Chapter 4): 

```
importnumpyasnp
fromsklearn.datasetsimportload_iris
fromsklearn.linear_modelimportPerceptron
iris=load_iris(as_frame=True)
X=iris.data[["petal length (cm)", "petal width (cm)"]].values
y= (iris.target==0)  # Iris setosa
```

```
per_clf=Perceptron(random_state=42)
per_clf.fit(X, y)
```

```
X_new= [[2, 0.5], [3, 1]]
y_pred=per_clf.predict(X_new)  # predicts True and False for these 2 flowers
```

You may have noticed that the perceptron learning algorithm strongly resembles sto‐ chastic gradient descent (introduced in Chapter 4). In fact, Scikit-Learn’s `Perceptron` class is equivalent to using an `SGDClassifier` with the following hyperparameters: `loss="perceptron"` , `learning_rate="constant"` , `eta0=1` (the learning rate), and `penalty=None` (no regularization). 

In their 1969 monograph _Perceptrons_ , Marvin Minsky and Seymour Papert high‐ lighted a number of serious weaknesses of perceptrons—in particular, the fact that they are incapable of solving some trivial problems (e.g., the _exclusive OR_ (XOR) classification problem; see the left side of Figure 10-6). This is true of any other linear classification model (such as logistic regression classifiers), but researchers had expected much more from perceptrons, and some were so disappointed that they 

> 7 Note that this solution is not unique: when data points are linearly separable, there is an infinity of hyper‐ planes that can separate them. 

**From Biological to Artificial Neurons | 307** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 



<!-- Start of picture text -->
[P<br>. ES<br>NE aver ga<br>. f MES FP<br>0 s x<br>oN ONG<br><!-- End of picture text -->





<!-- Start of picture text -->
oO oo oO Output layer<br>SERNODN -..<br>CI ACV ACI ACT Fl<br>QE PV (PY sen<br>NWPKZA~/\<br>x, X } Input layer<br><!-- End of picture text -->



For many years researchers struggled to find a way to train MLPs, without success. In the early 1960s several researchers discussed the possibility of using gradient descent to train neural networks, but as we saw in Chapter 4, this requires computing the gradients of the model’s error with regard to the model parameters; it wasn’t clear at the time how to do this efficiently with such a complex model containing so many parameters, especially with the computers they had back then. 

Then, in 1970, a researcher named Seppo Linnainmaa introduced in his master’s thesis a technique to compute all the gradients automatically and efficiently. This algorithm is now called _reverse-mode automatic differentiation_ (or _reverse-mode auto‐ diff_ for short). In just two passes through the network (one forward, one backward), it is able to compute the gradients of the neural network’s error with regard to every single model parameter. In other words, it can find out how each connection weight and each bias should be tweaked in order to reduce the neural network’s error. These gradients can then be used to perform a gradient descent step. If you repeat this process of computing the gradients automatically and taking a gradient descent step, the neural network’s error will gradually drop until it eventually reaches a minimum. This combination of reverse-mode autodiff and gradient descent is now called _backpropagation_ (or _backprop_ for short). 



There are various autodiff techniques, with different pros and cons. _Reverse-mode autodiff_ is well suited when the function to differ‐ entiate has many variables (e.g., connection weights and biases) and few outputs (e.g., one loss). If you want to learn more about autodiff, check out Appendix B. 

Backpropagation can actually be applied to all sorts of computational graphs, not just neural networks: indeed, Linnainmaa’s master’s thesis was not about neural nets, it was more general. It was several more years before backprop started to be used to train neural networks, but it still wasn’t mainstream. Then, in 1985, David Rumel‐ hart, Geoffrey Hinton, and Ronald Williams published a groundbreaking paper<sup>10</sup> analyzing how backpropagation allowed neural networks to learn useful internal representations. Their results were so impressive that backpropagation was quickly popularized in the field. Today, it is by far the most popular training technique for neural networks. 

> 10 David Rumelhart et al., “Learning Internal Representations by Error Propagation” (Defense Technical Infor‐ mation Center technical report, September 1985). 

**310 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

Let’s run through how backpropagation works again in a bit more detail: 

- It handles one mini-batch at a time (for example, containing 32 instances each), and it goes through the full training set multiple times. Each pass is called an _epoch_ . 

- Each mini-batch enters the network through the input layer. The algorithm then computes the output of all the neurons in the first hidden layer, for every instance in the mini-batch. The result is passed on to the next layer, its output is computed and passed to the next layer, and so on until we get the output of the last layer, the output layer. This is the _forward pass_ : it is exactly like making predictions, except all intermediate results are preserved since they are needed for the backward pass. 

- Next, the algorithm measures the network’s output error (i.e., it uses a loss function that compares the desired output and the actual output of the network, and returns some measure of the error). 

- Then it computes how much each output bias and each connection to the output layer contributed to the error. This is done analytically by applying the _chain rule_ (perhaps the most fundamental rule in calculus), which makes this step fast and precise. 

- The algorithm then measures how much of these error contributions came from each connection in the layer below, again using the chain rule, working backward until it reaches the input layer. As explained earlier, this reverse pass efficiently measures the error gradient across all the connection weights and biases in the network by propagating the error gradient backward through the network (hence the name of the algorithm). 

- Finally, the algorithm performs a gradient descent step to tweak all the connec‐ tion weights in the network, using the error gradients it just computed. 



It is important to initialize all the hidden layers’ connection weights randomly, or else training will fail. For example, if you initialize all weights and biases to zero, then all neurons in a given layer will be perfectly identical, and thus backpropagation will affect them in exactly the same way, so they will remain identical. In other words, despite having hundreds of neurons per layer, your model will act as if it had only one neuron per layer: it won’t be too smart. If instead you randomly initialize the weights, you _break the symmetry_ and allow backpropagation to train a diverse team of neurons. 

**From Biological to Artificial Neurons | 311** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

In short, backpropagation makes predictions for a mini-batch (forward pass), meas‐ ures the error, then goes through each layer in reverse to measure the error contribu‐ tion from each parameter (reverse pass), and finally tweaks the connection weights and biases to reduce the error (gradient descent step). 

In order for backprop to work properly, Rumelhart and his colleagues made a key change to the MLP’s architecture: they replaced the step function with the logistic function, _σ_ ( _z_ ) = 1 / (1 + exp(– _z_ )), also called the _sigmoid_ function. This was essential because the step function contains only flat segments, so there is no gradient to work with (gradient descent cannot move on a flat surface), while the sigmoid function has a well-defined nonzero derivative everywhere, allowing gradient descent to make some progress at every step. In fact, the backpropagation algorithm works well with many other activation functions, not just the sigmoid function. Here are two other popular choices: 

_The hyperbolic tangent function: tanh(z) = 2σ(2z) – 1_ 

Just like the sigmoid function, this activation function is _S_ -shaped, continuous, and differentiable, but its output value ranges from –1 to 1 (instead of 0 to 1 in the case of the sigmoid function). That range tends to make each layer’s output more or less centered around 0 at the beginning of training, which often helps speed up convergence. 

_The rectified linear unit function: ReLU(z) = max(0, z)_ 

The ReLU function is continuous but unfortunately not differentiable at _z_ = 0 (the slope changes abruptly, which can make gradient descent bounce around), and its derivative is 0 for _z_ < 0. In practice, however, it works very well and has the advantage of being fast to compute, so it has become the default.<sup>11</sup> Importantly, the fact that it does not have a maximum output value helps reduce some issues during gradient descent (we will come back to this in Chapter 11). 

These popular activation functions and their derivatives are represented in Fig‐ ure 10-8. But wait! Why do we need activation functions in the first place? Well, if you chain several linear transformations, all you get is a linear transformation. For example, if f( _x_ ) = 2 _x_ + 3 and g( _x_ ) = 5 _x_ – 1, then chaining these two linear functions gives you another linear function: f(g( _x_ )) = 2(5 _x_ – 1) + 3 = 10 _x_ + 1. So if you don’t have some nonlinearity between layers, then even a deep stack of layers is equivalent to a single layer, and you can’t solve very complex problems with that. Conversely, a large enough DNN with nonlinear activations can theoretically approximate any continuous function. 

> 11 Biological neurons seem to implement a roughly sigmoid ( _S_ -shaped) activation function, so researchers stuck to sigmoid functions for a very long time. But it turns out that ReLU generally works better in ANNs. This is one of the cases where the biological analogy was perhaps misleading. 

###### **312 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 



<!-- Start of picture text -->
Activation functions Derivatives<br>1D<br>2 1.0<br>0.8<br>1 ===<br>} a i — Heaviside 0.40.6<br>—-- ReLU<br>--- Sigmoid+ . 0.2<br>-0.2<br>-4 -3 —2 -1 0 1 2 3 4 -4 -3 -2 -1 0 1 2 3 4<br><!-- End of picture text -->

```
fromsklearn.datasetsimportfetch_california_housing
fromsklearn.metricsimportroot_mean_squared_error
fromsklearn.model_selectionimporttrain_test_split
fromsklearn.neural_networkimportMLPRegressor
fromsklearn.pipelineimportmake_pipeline
fromsklearn.preprocessingimportStandardScaler
housing=fetch_california_housing()
X_train_full, X_test, y_train_full, y_test=train_test_split(
housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid=train_test_split(
X_train_full, y_train_full, random_state=42)
```

```
mlp_reg=MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline=make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)
y_pred=pipeline.predict(X_valid)
rmse=root_mean_squared_error(y_valid, y_pred)  # about 0.505
```

We get a validation RMSE of about 0.505, which is comparable to what you would get with a random forest classifier. Not too bad for a first try! 

Note that this MLP does not use any activation function for the output layer, so it’s free to output any value it wants. This is generally fine, but if you want to guarantee that the output will always be positive, then you should use the ReLU activation function in the output layer, or the _softplus_ activation function, which is a smooth variant of ReLU: softplus( _z_ ) = log(1 + exp( _z_ )). Softplus is close to 0 when _z_ is negative, and close to _z_ when _z_ is positive. Finally, if you want to guarantee that the predictions will always fall within a given range of values, then you should use the sigmoid function or the hyperbolic tangent, and scale the targets to the appropriate range: 0 to 1 for sigmoid and –1 to 1 for tanh. Sadly, the `MLPRegressor` class does not support activation functions in the output layer. 



Building and training a standard MLP with Scikit-Learn in just a few lines of code is very convenient, but the neural net features are limited. This is why we will switch to Keras in the second part of this chapter. 

The `MLPRegressor` class uses the mean square error, which is usually what you want for regression, but if you have a lot of outliers in the training set, you may prefer to use the mean absolute error instead. Alternatively, you may want to use the _Huber loss_ , which is a combination of both. It is quadratic when the error is smaller than a threshold _δ_ (typically 1) but linear when the error is larger than _δ_ . The linear part makes it less sensitive to outliers than the mean square error, and the quadratic part allows it to converge faster and be more precise than the mean absolute error. However, `MLPRegressor` only supports the MSE. 

**314 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

Table 10-1 summarizes the typical architecture of a regression MLP. 

_Table 10-1. Typical regression MLP architecture_ 

|**Hyperparameter**|**Typical value**|
|---|---|
|# hidden layers|Depends on the problem, but typically 1 to 5|
|# neurons per hidden layer|Depends on the problem, but typically 10 to 100|
|# output neurons|1 per prediction dimension|
|Hidden activation|ReLU|
|Output activation|None, or ReLU/softplus (if positive outputs) or sigmoid/tanh (if bounded outputs)|
|Loss function|MSE, or Huber if outliers|



##### **Classification MLPs** 

MLPs can also be used for classification tasks. For a binary classification problem, you just need a single output neuron using the sigmoid activation function: the output will be a number between 0 and 1, which you can interpret as the estimated probability of the positive class. The estimated probability of the negative class is equal to one minus that number. 

MLPs can also easily handle multilabel binary classification tasks (see Chapter 3). For example, you could have an email classification system that predicts whether each incoming email is ham or spam, and simultaneously predicts whether it is an urgent or nonurgent email. In this case, you would need two output neurons, both using the sigmoid activation function: the first would output the probability that the email is spam, and the second would output the probability that it is urgent. More generally, you would dedicate one output neuron for each positive class. Note that the output probabilities do not necessarily add up to 1. This lets the model output any combination of labels: you can have nonurgent ham, urgent ham, nonurgent spam, and perhaps even urgent spam (although that would probably be an error). 

If each instance can belong only to a single class, out of three or more possible classes (e.g., classes 0 through 9 for digit image classification), then you need to have one output neuron per class, and you should use the softmax activation function for the whole output layer (see Figure 10-9). The softmax function (introduced in Chapter 4) will ensure that all the estimated probabilities are between 0 and 1 and that they add up to 1, since the classes are exclusive. As you saw in Chapter 3, this is called multiclass classification. 

Regarding the loss function, since we are predicting probability distributions, the cross-entropy loss (or _x-entropy_ or log loss for short, see Chapter 4) is generally a good choice. 

**From Biological to Artificial Neurons | 315** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 



<!-- Start of picture text -->
SISIe Output layer<br>ReLU [XxBx \\ Ss,<br>BORG }Hidden layer<br>NX ~\<br>X, X, ) Input layer<br><!-- End of picture text -->





<!-- Start of picture text -->
LN<br><!-- End of picture text -->

LN LN 



<!-- Start of picture text -->
LN<br><!-- End of picture text -->

##### **Building an Image Classifier Using the Sequential API** 

First, we need to load a dataset. We will use Fashion MNIST, which is a drop-in replacement of MNIST (introduced in Chapter 3). It has the exact same format as MNIST (70,000 grayscale images of 28 × 28 pixels each, with 10 classes), but the images represent fashion items rather than handwritten digits, so each class is more diverse, and the problem turns out to be significantly more challenging than MNIST. For example, a simple linear model reaches about 92% accuracy on MNIST, but only about 83% on Fashion MNIST. 

###### **Using Keras to load the dataset** 

Keras provides some utility functions to fetch and load common datasets, including MNIST, Fashion MNIST, and a few more. Let’s load Fashion MNIST. It’s already shuffled and split into a training set (60,000 images) and a test set (10,000 images), but we’ll hold out the last 5,000 images from the training set for validation: 

```
importtensorflowastf
```

```
fashion_mnist=tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) =fashion_mnist
X_train, y_train=X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid=X_train_full[-5000:], y_train_full[-5000:]
```



TensorFlow is usually imported as `tf` , and the Keras API is avail‐ able via `tf.keras` . 

When loading MNIST or Fashion MNIST using Keras rather than Scikit-Learn, one important difference is that every image is represented as a 28 × 28 array rather than a 1D array of size 784. Moreover, the pixel intensities are represented as integers (from 0 to 255) rather than floats (from 0.0 to 255.0). Let’s take a look at the shape and data type of the training set: 

```
>>> X_train.shape
(55000, 28, 28)
>>> X_train.dtype
dtype('uint8')
```

For simplicity, we’ll scale the pixel intensities down to the 0–1 range by dividing them by 255.0 (this also converts them to floats): 

```
X_train, X_valid, X_test=X_train/255., X_valid/255., X_test/255.
```

With MNIST, when the label is equal to 5, it means that the image represents the handwritten digit 5. Easy. For Fashion MNIST, however, we need the list of class names to know what we are dealing with: 

**318 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 



<!-- Start of picture text -->
Ankle boot T-shirt/top T-shirt/top Dress T-shirt/top Pullover Sneaker Pullover Sandal Sandal<br>-i \ ca<br>T-shirt/top Ankle boot Sandal Sandal Sneaker Ankle boot Trouser  T-shirt/top Shirt Coat<br>RAxsoc=—-a i f @ 4<br>Dress Trouser Coat Bag Coat Dress T-shirt/top Pullover Coat Coat<br>Sandal Dress Shirt Shirt T-shirt/top Bag Sandal Pullover Trouser Shirt<br><!-- End of picture text -->

- Next, we build the first layer (an `Input` layer) and add it to the model. We specify the input `shape` , which doesn’t include the batch size, only the shape of the instances. Keras needs to know the shape of the inputs so it can determine the shape of the connection weight matrix of the first hidden layer. 

- Then we add a `Flatten` layer. Its role is to convert each input image into a 1D array: for example, if it receives a batch of shape [32, 28, 28], it will reshape it to [32, 784]. In other words, if it receives input data `X` , it computes `X.reshape(-1, 784)` . This layer doesn’t have any parameters; it’s just there to do some simple preprocessing. 

- Next we add a `Dense` hidden layer with 300 neurons. It will use the ReLU activation function. Each `Dense` layer manages its own weight matrix, containing all the connection weights between the neurons and their inputs. It also manages a vector of bias terms (one per neuron). When it receives some input data, it computes Equation 10-2. 

- Then we add a second `Dense` hidden layer with 100 neurons, also using the ReLU activation function. 

- Finally, we add a `Dense` output layer with 10 neurons (one per class), using the softmax activation function because the classes are exclusive. 



Specifying `activation="relu"` is equivalent to specifying `activation=tf.keras.activations.relu` . Other activation func‐ tions are available in the `tf.keras.activations` package. We will use many of them in this book; see _https://keras.io/api/layers/activa tions_ for the full list. We will also define our own custom activation functions in Chapter 12. 

Instead of adding the layers one by one as we just did, it’s often more convenient to pass a list of layers when creating the `Sequential` model. You can also drop the `Input` layer and instead specify the `input_shape` in the first layer: 

```
model=tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=[28, 28]),
tf.keras.layers.Dense(300, activation="relu"),
tf.keras.layers.Dense(100, activation="relu"),
tf.keras.layers.Dense(10, activation="softmax")
```

```
])
```

**320 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

The model’s `summary()` method displays all the model’s layers,<sup>14</sup> including each layer’s name (which is automatically generated unless you set it when creating the layer), its output shape ( `None` means the batch size can be anything), and its number of parameters. The summary ends with the total number of parameters, including trainable and non-trainable parameters. Here we only have trainable parameters (you will see some non-trainable parameters later in this chapter): 

```
>>> model.summary()
Model: "sequential"
```

|`_________________________________________________________________`<br>`Layer (type)                Output Shape              Param #`<br>`=================================================================`|
|---|
|`flatten (Flatten)           (None, 784)               0`|
|`dense (Dense)               (None, 300)               235500`|
|`dense_1 (Dense)             (None, 100)               30100`|
|`dense_2 (Dense)             (None, 10)                1010`|
|`=================================================================`<br>`Total params: 266610 (1.02 MB)`|
|`Trainable params: 266610 (1.02 MB)`<br>`Non-trainable params: 0 (0.00 Byte)`<br>`_________________________________________________________________`|



Note that `Dense` layers often have a _lot_ of parameters. For example, the first hidden layer has 784 × 300 connection weights, plus 300 bias terms, which adds up to 235,500 parameters! This gives the model quite a lot of flexibility to fit the training data, but it also means that the model runs the risk of overfitting, especially when you do not have a lot of training data. We will come back to this later. 

Each layer in a model must have a unique name (e.g., `"dense_2"` ). You can set the layer names explicitly using the constructor’s `name` argument, but generally it’s sim‐ pler to let Keras name the layers automatically, as we just did. Keras takes the layer’s class name and converts it to snake case (e.g., a layer from the `MyCoolLayer` class is named `"my_cool_layer"` by default). Keras also ensures that the name is globally unique, even across models, by appending an index if needed, as in `"dense_2"` . But why does it bother making the names unique across models? Well, this makes it possible to merge models easily without getting name conflicts. 

> 14 You can also use `tf.keras.utils.plot_model()` to generate an image of your model. 

**Implementing MLPs with Keras | 321** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 



All global state managed by Keras is stored in a _Keras session_ , which you can clear using `tf.keras.backend.clear_session()` . In particular, this resets the name counters. 

You can easily get a model’s list of layers using the `layers` attribute, or use the `get_layer()` method to access a layer by name: 

```
>>> model.layers
[<keras.layers.reshaping.flatten.Flatten at 0x17380e9b0>,
 <keras.layers.core.dense.Dense at 0x1776211b0>,
 <keras.layers.core.dense.Dense at 0x177622410>,
 <keras.layers.core.dense.Dense at 0x176e78c40>]
>>> hidden1=model.layers[1]
>>> hidden1.name
'dense'
>>> model.get_layer('dense') ishidden1
True
```

All the parameters of a layer can be accessed using its `get_weights()` and `set_weights()` methods. For a `Dense` layer, this includes both the connection weights and the bias terms: 

```
>>> weights, biases=hidden1.get_weights()
>>> weights
array([[ 5.3297073e-02,  2.4198458e-02, -2.1023259e-02, ...,  4.6089381e-02],
       [ 2.2632368e-02,  5.9892908e-03,  1.4587238e-02, ...,  2.4750374e-02],
       ...,
       [-4.4557646e-02, -5.9672445e-02,  6.5973431e-02, ...,  5.1353276e-02],
       [-1.4996272e-02,  1.0063291e-02, -3.2075007e-02, ..., -6.4764827e-02]],
       dtype=float32)
>>> weights.shape
(784, 300)
>>> biases
array([0., 0., 0., 0., 0., 0., 0., 0., 0., ...,  0., 0., 0.], dtype=float32)
>>> biases.shape
(300,)
```

Notice that the `Dense` layer initialized the connection weights randomly (which is needed to break symmetry, as discussed earlier), and the biases were initialized to zeros, which is fine. If you want to use a different initialization method, you can set `kernel_initializer` ( _kernel_ is another name for the matrix of connection weights) or `bias_initializer` when creating the layer. We’ll discuss initializers further in Chapter 11, and the full list is at _https://keras.io/api/layers/initializers_ . 

**322 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 



<!-- Start of picture text -->
LN<br><!-- End of picture text -->

## LN 

LN 



<!-- Start of picture text -->
LN<br><!-- End of picture text -->



<!-- Start of picture text -->
™<br><!-- End of picture text -->

™ & 



<!-- Start of picture text -->
&<br><!-- End of picture text -->



<!-- Start of picture text -->
™<br><!-- End of picture text -->

# ™ 



<!-- Start of picture text -->
™<br><!-- End of picture text -->

# ™ 



<!-- Start of picture text -->
0.8 47%<br>\\<br>\<br>0.6 ry<br>\<br>\<br>\<br>wa<br>0.4 — ae<br>---- loss ae<br>024.-*- sparse_categorical_accuracy Ee ieee<br>— val_loss<br>—— val_sparse_categorical_accuracy<br>0.0<br>0 5 10 15 20 25<br>Epoch<br><!-- End of picture text -->

continue training. This is as simple as calling the `fit()` method again, since Keras just continues training where it left off: you should be able to reach about 89.8% validation accuracy, while the training accuracy will continue to rise up to 100% (this is not always the case). 

If you are not satisfied with the performance of your model, you should go back and tune the hyperparameters. The first one to check is the learning rate. If that doesn’t help, try another optimizer (and always retune the learning rate after changing any hyperparameter). If the performance is still not great, then try tuning model hyperparameters such as the number of layers, the number of neurons per layer, and the types of activation functions to use for each hidden layer. You can also try tuning other hyperparameters, such as the batch size (it can be set in the `fit()` method using the `batch_size` argument, which defaults to 32). We will get back to hyperparameter tuning at the end of this chapter. Once you are satisfied with your model’s validation accuracy, you should evaluate it on the test set to estimate the generalization error before you deploy the model to production. You can easily do this using the `evaluate()` method (it also supports several other arguments, such as `batch_size` and `sample_weight` ; please check the documentation for more details): 

```
>>> model.evaluate(X_test, y_test)
313/313 [==============================] - 1s 626us/step
  - loss: 0.3213 - sparse_categorical_accuracy: 0.8858
[0.3213411867618561, 0.8858000040054321]
```

As you saw in Chapter 2, it is common to get slightly lower performance on the test set than on the validation set, because the hyperparameters are tuned on the valida‐ tion set, not the test set (however, in this example, we did not do any hyperparameter tuning, so the lower accuracy is just bad luck). Remember to resist the temptation to tweak the hyperparameters on the test set, or else your estimate of the generalization error will be too optimistic. 

###### **Using the model to make predictions** 

Now let’s use the model’s `predict()` method to make predictions on new instances. Since we don’t have actual new instances, we’ll just use the first three instances of the test set: 

```
>>> X_new=X_test[:3]
>>> y_proba=model.predict(X_new)
>>> y_proba.round(2)
array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.02, 0.  , 0.97],
       [0.  , 0.  , 0.99, 0.  , 0.01, 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]],
      dtype=float32)
```

For each instance the model estimates one probability per class, from class 0 to class 9. This is similar to the output of the `predict_proba()` method in Scikit-Learn 

**Implementing MLPs with Keras | 327** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 



<!-- Start of picture text -->
Ankle boot Pullover Trouser<br><!-- End of picture text -->

Scikit-Learn’s `MLPRegressor` did. Moreover, in this example we don’t need a `Flatten` layer, and instead we’re using a `Normalization` layer as the first layer: it does the same thing as Scikit-Learn’s `StandardScaler` , but it must be fitted to the training data using its `adapt()` method _before_ you call the model’s `fit()` method. (Keras has other preprocessing layers, which will be covered in Chapter 13). Let’s take a look: 

```
tf.random.set_seed(42)
norm_layer=tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model=tf.keras.Sequential([
norm_layer,
tf.keras.layers.Dense(50, activation="relu"),
tf.keras.layers.Dense(50, activation="relu"),
tf.keras.layers.Dense(50, activation="relu"),
tf.keras.layers.Dense(1)
])
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)
history=model.fit(X_train, y_train, epochs=20,
validation_data=(X_valid, y_valid))
mse_test, rmse_test=model.evaluate(X_test, y_test)
X_new=X_test[:3]
y_pred=model.predict(X_new)
```



The `Normalization` layer learns the feature means and standard deviations in the training data when you call the `adapt()` method. Yet when you display the model’s summary, these statistics are listed as non-trainable. This is because these parameters are not affected by gradient descent. 

As you can see, the sequential API is quite clean and straightforward. However, although `Sequential` models are extremely common, it is sometimes useful to build neural networks with more complex topologies, or with multiple inputs or outputs. For this purpose, Keras offers the functional API. 

##### **Building Complex Models Using the Functional API** 

One example of a nonsequential neural network is a _Wide & Deep_ neural network. This neural network architecture was introduced in a 2016 paper by Heng-Tze Cheng et al.<sup>15</sup> It connects all or part of the inputs directly to the output layer, as shown in Figure 10-13. This architecture makes it possible for the neural network to learn both deep patterns (using the deep path) and simple rules (through the short path).<sup>16</sup> In 

> 15 Heng-Tze Cheng et al., “Wide & Deep Learning for Recommender Systems”, _Proceedings of the First Workshop on Deep Learning for Recommender Systems_ (2016): 7–10. 

> 16 The short path can also be used to provide manually engineered features to the neural network. 

**Implementing MLPs with Keras | 329** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 



<!-- Start of picture text -->
Output layer<br>Hidden 2<br>Wide Deep<br>Hidden 1<br>Input layer<br><!-- End of picture text -->

At a high level, the first five lines create all the layers we need to build the model, the next six lines use these layers just like functions to go from the input to the output, and the last line creates a Keras `Model` object by pointing to the input and the output. Let’s go through this code in more detail: 

- First, we create five layers: a `Normalization` layer to standardize the inputs, two `Dense` layers with 30 neurons each, using the ReLU activation function, a `Concatenate` layer, and one more `Dense` layer with a single neuron for the output layer, without any activation function. 

- Next, we create an `Input` object (the variable name `input_` is used to avoid overshadowing Python’s built-in `input()` function). This is a specification of the kind of input the model will get, including its `shape` and optionally its `dtype` , which defaults to 32-bit floats. A model may actually have multiple inputs, as you will see shortly. 

- Then we use the `Normalization` layer just like a function, passing it the `Input` object. This is why this is called the functional API. Note that we are just telling Keras how it should connect the layers together; no actual data is being processed yet, as the `Input` object is just a data specification. In other words, it’s a symbolic input. The output of this call is also symbolic: `normalized` doesn’t store any actual data, it’s just used to construct the model. 

- In the same way, we then pass `normalized` to `hidden_layer1` , which outputs `hidden1` , and we pass `hidden1` to `hidden_layer2` , which outputs `hidden2` . 

- So far we’ve connected the layers sequentially, but then we use the `concat_layer` to concatenate the input and the second hidden layer’s output. Again, no actual data is concatenated yet: it’s all symbolic, to build the model. 

- Then we pass `concat` to the `output_layer` , which gives us the final `output` . 

- Lastly, we create a Keras `Model` , specifying which inputs and outputs to use. 

Once you have built this Keras model, everything is exactly like earlier, so there’s no need to repeat it here: you compile the model, adapt the `Normalization` layer, fit the model, evaluate it, and use it to make predictions. 

But what if you want to send a subset of the features through the wide path and a different subset (possibly overlapping) through the deep path, as illustrated in Fig‐ ure 10-14? In this case, one solution is to use multiple inputs. For example, suppose we want to send five features through the wide path (features 0 to 4), and six features through the deep path (features 2 to 7). We can do this as follows: 

**Implementing MLPs with Keras | 331** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 



<!-- Start of picture text -->
Output layer<br>Hidden2<br>Hidden1<br><!-- End of picture text -->

Now we can compile the model as usual, but when we call the `fit()` method, instead of passing a single input matrix `X_train` , we must pass a pair of matrices `(X_train_wide, X_train_deep)` , one per input. The same is true for `X_valid` , and also for `X_test` and `X_new` when you call `evaluate()` or `predict()` : 

```
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
X_train_wide, X_train_deep=X_train[:, :5], X_train[:, 2:]
X_valid_wide, X_valid_deep=X_valid[:, :5], X_valid[:, 2:]
X_test_wide, X_test_deep=X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep=X_test_wide[:3], X_test_deep[:3]
```

```
norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history=model.fit((X_train_wide, X_train_deep), y_train, epochs=20,
validation_data=((X_valid_wide, X_valid_deep), y_valid))
mse_test=model.evaluate((X_test_wide, X_test_deep), y_test)
y_pred=model.predict((X_new_wide, X_new_deep))
```



Instead of passing a tuple `(X_train_wide, X_train_deep)` , you can pass a dictionary `{"input_wide": X_train_wide, "input_deep": X_train_deep}` , if you set `name="input_wide"` and `name="input_deep"` when creating the inputs. This is highly recommended when there are many inputs, to clarify the code and avoid getting the order wrong. 

There are also many use cases in which you may want to have multiple outputs: 

- The task may demand it. For instance, you may want to locate and classify the main object in a picture. This is both a regression tasks and a classification task. 

- Similarly, you may have multiple independent tasks based on the same data. Sure, you could train one neural network per task, but in many cases you will get better results on all tasks by training a single neural network with one output per task. This is because the neural network can learn features in the data that are useful across tasks. For example, you could perform _multitask classification_ on pictures of faces, using one output to classify the person’s facial expression (smiling, surprised, etc.) and another output to identify whether they are wearing glasses or not. 

- Another use case is as a regularization technique (i.e., a training constraint whose objective is to reduce overfitting and thus improve the model’s ability to generalize). For example, you may want to add an auxiliary output in a neural network architecture (see Figure 10-15) to ensure that the underlying part of the 

**Implementing MLPs with Keras | 333** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 



<!-- Start of picture text -->
Output layer Auxiliary output<br>Hidden 2<br>Input wide Input deep<br><!-- End of picture text -->



Instead of passing a tuple `loss=("mse", "mse")` , you can pass a dictionary `loss={"output": "mse", "aux_output": "mse"}` , assuming you created the output layers with `name="output"` and `name="aux_output"` . Just like for the inputs, this clarifies the code and avoids errors when there are several outputs. You can also pass a dictionary for `loss_weights` . 

Now when we train the model, we need to provide labels for each output. In this example, the main output and the auxiliary output should try to predict the same thing, so they should use the same labels. So instead of passing `y_train` , we need to pass `(y_train, y_train)` , or a dictionary `{"output": y_train, "aux_output": y_train}` if the outputs were named `"output"` and `"aux_output"` . The same goes for `y_valid` and `y_test` : 

```
norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history=model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=20,
validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid))
)
```

When we evaluate the model, Keras returns the weighted sum of the losses, as well as all the individual losses and metrics: 

```
eval_results=model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse=eval_results
```



If you set `return_dict=True` , then `evaluate()` will return a dictio‐ nary instead of a big tuple. 

Similarly, the `predict()` method will return predictions for each output: 

```
y_pred_main, y_pred_aux=model.predict((X_new_wide, X_new_deep))
```

The `predict()` method returns a tuple, and it does not have a `return_dict` argument to get a dictionary instead. However, you can create one using `model.output_names` : 

```
y_pred_tuple=model.predict((X_new_wide, X_new_deep))
y_pred=dict(zip(model.output_names, y_pred_tuple))
```

As you can see, you can build all sorts of architectures with the functional API. Next, we’ll look at one last way you can build Keras models. 

**Implementing MLPs with Keras | 335** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

##### **Using the Subclassing API to Build Dynamic Models** 

Both the sequential API and the functional API are declarative: you start by declaring which layers you want to use and how they should be connected, and only then can you start feeding the model some data for training or inference. This has many advantages: the model can easily be saved, cloned, and shared; its structure can be displayed and analyzed; the framework can infer shapes and check types, so errors can be caught early (i.e., before any data ever goes through the model). It’s also fairly straightforward to debug, since the whole model is a static graph of layers. But the flip side is just that: it’s static. Some models involve loops, varying shapes, conditional branching, and other dynamic behaviors. For such cases, or simply if you prefer a more imperative programming style, the subclassing API is for you. 

With this approach, you subclass the `Model` class, create the layers you need in the constructor, and use them to perform the computations you want in the `call()` method. For example, creating an instance of the following `WideAndDeepModel` class gives us an equivalent model to the one we just built with the functional API: 

```
classWideAndDeepModel(tf.keras.Model):
def __init__(self, units=30, activation="relu", **kwargs):
super().__init__(**kwargs)  # needed to support naming the model
self.norm_layer_wide=tf.keras.layers.Normalization()
self.norm_layer_deep=tf.keras.layers.Normalization()
self.hidden1=tf.keras.layers.Dense(units, activation=activation)
self.hidden2=tf.keras.layers.Dense(units, activation=activation)
self.main_output=tf.keras.layers.Dense(1)
self.aux_output=tf.keras.layers.Dense(1)
```

```
defcall(self, inputs):
input_wide, input_deep=inputs
norm_wide=self.norm_layer_wide(input_wide)
norm_deep=self.norm_layer_deep(input_deep)
hidden1=self.hidden1(norm_deep)
hidden2=self.hidden2(hidden1)
concat=tf.keras.layers.concatenate([norm_wide, hidden2])
output=self.main_output(concat)
aux_output=self.aux_output(hidden2)
returnoutput, aux_output
```

```
model=WideAndDeepModel(30, activation="relu", name="my_cool_model")
```

This example looks like the previous one, except we separate the creation of the layers<sup>17</sup> in the constructor from their usage in the `call()` method. And we don’t need to create the `Input` objects: we can use the `input` argument to the `call()` method. 

> 17 Keras models have an `output` attribute, so we cannot use that name for the main output layer, which is why we renamed it to `main_output` . 

**336 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

Now that we have a model instance, we can compile it, adapt its normal‐ ization layers (e.g., using `model.norm_layer_wide.adapt(...)` and `model.norm_ layer_deep.adapt(...)` ), fit it, evaluate it, and use it to make predictions, exactly like we did with the functional API. 

The big difference with this API is that you can include pretty much anything you want in the `call()` method: `for` loops, `if` statements, low-level TensorFlow operations—your imagination is the limit (see Chapter 12)! This makes it a great API when experimenting with new ideas, especially for researchers. However, this extra flexibility does come at a cost: your model’s architecture is hidden within the `call()` method, so Keras cannot easily inspect it; the model cannot be cloned using `tf.keras.models.clone_model()` ; and when you call the `summary()` method, you only get a list of layers, without any information on how they are connected to each other. Moreover, Keras cannot check types and shapes ahead of time, and it is easier to make mistakes. So unless you really need that extra flexibility, you should probably stick to the sequential API or the functional API. 



Keras models can be used just like regular layers, so you can easily combine them to build complex architectures. 

Now that you know how to build and train neural nets using Keras, you will want to save them! 

##### **Saving and Restoring a Model** 

Saving a trained Keras model is as simple as it gets: 

```
model.save("my_keras_model", save_format="tf")
```

When you set `save_format="tf"` ,<sup>18</sup> Keras saves the model using TensorFlow’s _Saved‐ Model_ format: this is a directory (with the given name) containing several files and subdirectories. In particular, the _saved_model.pb_ file contains the model’s architecture and logic in the form of a serialized computation graph, so you don’t need to deploy the model’s source code in order to use it in production; the SavedModel is sufficient (you will see how this works in Chapter 12). The _keras_metadata.pb_ file contains extra information needed by Keras. The _variables_ subdirectory contains all the parameter values (including the connection weights, the biases, the normalization statistics, and the optimizer’s parameters), possibly split across multiple files if the 

> 18 This is currently the default, but the Keras team is working on a new format that may become the default in upcoming versions, so I prefer to set the format explicitly to be future-proof. 

**Implementing MLPs with Keras | 337** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

model is very large. Lastly, the _assets_ directory may contain extra files, such as data samples, feature names, class names, and so on. By default, the _assets_ directory is empty. Since the optimizer is also saved, including its hyperparameters and any state it may have, after loading the model you can continue training if you want. 



If you set `save_format="h5"` or use a filename that ends with _.h5_ , _.hdf5_ , or _.keras_ , then Keras will save the model to a single file using a Keras-specific format based on the HDF5 format. How‐ ever, most TensorFlow deployment tools require the SavedModel format instead. 

You will typically have a script that trains a model and saves it, and one or more scripts (or web services) that load the model and use it to evaluate it or to make predictions. Loading the model is just as easy as saving it: 

```
model=tf.keras.models.load_model("my_keras_model")
y_pred_main, y_pred_aux=model.predict((X_new_wide, X_new_deep))
```

You can also use `save_weights()` and `load_weights()` to save and load only the parameter values. This includes the connection weights, biases, preprocessing stats, optimizer state, etc. The parameter values are saved in one or more files such as _my_weights.data-00004-of-00052_ , plus an index file like _my_weights.index_ . 

Saving just the weights is faster and uses less disk space than saving the whole model, so it’s perfect to save quick checkpoints during training. If you’re training a big model, and it takes hours or days, then you must save checkpoints regularly in case the computer crashes. But how can you tell the `fit()` method to save checkpoints? Use callbacks. 

##### **Using Callbacks** 

The `fit()` method accepts a `callbacks` argument that lets you specify a list of objects that Keras will call before and after training, before and after each epoch, and even before and after processing each batch. For example, the `ModelCheckpoint` callback saves checkpoints of your model at regular intervals during training, by default at the end of each epoch: 

```
checkpoint_cb=tf.keras.callbacks.ModelCheckpoint("my_checkpoints",
save_weights_only=True)
history=model.fit([...], callbacks=[checkpoint_cb])
```

**338 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

Moreover, if you use a validation set during training, you can set `save_ best_only=True` when creating the `ModelCheckpoint` . In this case, it will only save your model when its performance on the validation set is the best so far. This way, you do not need to worry about training for too long and overfitting the training set: simply restore the last saved model after training, and this will be the best model on the validation set. This is one way to implement early stopping (introduced in Chapter 4), but it won’t actually stop training. 

Another way is to use the `EarlyStopping` callback. It will interrupt training when it measures no progress on the validation set for a number of epochs (defined by the `patience` argument), and if you set `restore_best_weights=True` it will roll back to the best model at the end of training. You can combine both callbacks to save checkpoints of your model in case your computer crashes, and interrupt training early when there is no more progress, to avoid wasting time and resources and to reduce overfitting: 

```
early_stopping_cb=tf.keras.callbacks.EarlyStopping(patience=10,
restore_best_weights=True)
history=model.fit([...], callbacks=[checkpoint_cb, early_stopping_cb])
```

The number of epochs can be set to a large value since training will stop automati‐ cally when there is no more progress (just make sure the learning rate is not too small, or else it might keep making slow progress until the end). The `EarlyStopping` callback will store the weights of the best model in RAM, and it will restore them for you at the end of training. 



Many other callbacks are available in the `tf.keras.callbacks` package. 

If you need extra control, you can easily write your own custom callbacks. For example, the following custom callback will display the ratio between the validation loss and the training loss during training (e.g., to detect overfitting): 

```
classPrintValTrainRatioCallback(tf.keras.callbacks.Callback):
defon_epoch_end(self, epoch, logs):
ratio=logs["val_loss"] /logs["loss"]
print(f"Epoch={epoch}, val/train={ratio:.2f}")
```



<!-- Start of picture text -->
Implementing MLPs with Keras  |  339<br><!-- End of picture text -->

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

As you might expect, you can implement `on_train_begin()` , `on_train_end()` , `on_epoch_begin()` , `on_epoch_end()` , `on_batch_begin()` , and `on_batch_end()` . Call‐ backs can also be used during evaluation and predictions, should you ever need them (e.g., for debugging). For evaluation, you should implement `on_test_begin()` , `on_test_end()` , `on_test_batch_begin()` , or `on_test_batch_end()` , which are called by `evaluate()` . For prediction, you should implement `on_predict_begin()` , `on_predict_end()` , `on_predict_batch_begin()` , or `on_predict_batch_end()` , which are called by `predict()` . 

Now let’s take a look at one more tool you should definitely have in your toolbox when using Keras: TensorBoard. 

##### **Using TensorBoard for Visualization** 

TensorBoard is a great interactive visualization tool that you can use to view the learning curves during training, compare curves and metrics between multiple runs, visualize the computation graph, analyze training statistics, view images generated by your model, visualize complex multidimensional data projected down to 3D and automatically clustered for you, _profile_ your network (i.e., measure its speed to identify bottlenecks), and more! 

TensorBoard is installed automatically when you install TensorFlow. However, you will need a TensorBoard plug-in to visualize profiling data. If you followed the installation instructions at _https://homl.info/install_ to run everything locally, then you already have the plug-in installed, but if you are using Colab, then you must run the following command: 

```
%pipinstall-q-Utensorboard-plugin-profile
```

To use TensorBoard, you must modify your program so that it outputs the data you want to visualize to special binary logfiles called _event files_ . Each binary data record is called a _summary_ . The TensorBoard server will monitor the log directory, and it will automatically pick up the changes and update the visualizations: this allows you to visualize live data (with a short delay), such as the learning curves during training. In general, you want to point the TensorBoard server to a root log directory and configure your program so that it writes to a different subdirectory every time it runs. This way, the same TensorBoard server instance will allow you to visualize and compare data from multiple runs of your program, without getting everything mixed up. 

Let’s name the root log directory _my_logs_ , and let’s define a little function that generates the path of the log subdirectory based on the current date and time, so that it’s different at every run: 

**340 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

```
frompathlibimportPath
fromtimeimportstrftime
```

```
defget_run_logdir(root_logdir="my_logs"):
returnPath(root_logdir) /strftime("run_%Y_%m_%d_%H_%M_%S")
```

```
run_logdir=get_run_logdir()  # e.g., my_logs/run_2022_08_01_17_25_59
```

The good news is that Keras provides a convenient `TensorBoard()` callback that will take care of creating the log directory for you (along with its parent directories if needed), and it will create event files and write summaries to them during training. It will measure your model’s training and validation loss and metrics (in this case, the MSE and RMSE), and it will also profile your neural network. It is straightforward to use: 

```
tensorboard_cb=tf.keras.callbacks.TensorBoard(run_logdir,
profile_batch=(100, 200))
history=model.fit([...], callbacks=[tensorboard_cb])
```

That’s all there is to it! In this example, it will profile the network between batches 100 and 200 during the first epoch. Why 100 and 200? Well, it often takes a few batches for the neural network to “warm up”, so you don’t want to profile too early, and profiling uses resources, so it’s best not to do it for every batch. 

Next, try changing the learning rate from 0.001 to 0.002, and run the code again, with a new log subdirectory. You will end up with a directory structure similar to this one: 

```
my_logs
├── run_2022_08_01_17_25_59
│   ├── train
│   │   ├── events.out.tfevents.1659331561.my_host_name.42042.0.v2
│   │   ├── events.out.tfevents.1659331562.my_host_name.profile-empty
│   │   └── plugins
│   │       └── profile
│   │           └── 2022_08_01_17_26_02
│   │               ├── my_host_name.input_pipeline.pb
│   │               └── [...]
│   └── validation
│       └── events.out.tfevents.1659331562.my_host_name.42042.1.v2
└── run_2022_08_01_17_31_12
    └── [...]
```

There’s one directory per run, each containing one subdirectory for training logs and one for validation logs. Both contain event files, and the training logs also include profiling traces. 

**Implementing MLPs with Keras | 341** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 





<!-- Start of picture text -->
TensorBoard SCALARS GRAPHS TIMESERIES PROJECTOR PROFILE INACTIVE ¥ | @ uPLOAD | -e- CG 2x ©<br>(J Show data download links Q Filter tags (regular expressions supported)<br>Ignore outliers in chart scaling Refresh data<br>epoch_loss<br>Tooltipmethod: sorting default v_~<br>epoch_loss<br>tag: epoch_loss<br>Smoothing<br>——e——_ 06 07<br>a 06iN<br>train 0.45 —— |__| .<br>© {Un.2022_08.01-17_25.59/tral oa  [TEROQISS® maw | Smoothed learning curve<br>run_2022_08_01-17_31_12/trai 035 ~ | | | |<br>iv] SS<br>ae =<br>TOGGLE ALL RUNS 03 === -<br>('} 2 4 6 8 10 12 14 16 18<br>Raw learning curves<br><!-- End of picture text -->

You can also visualize the whole computation graph in the GRAPHS tab, the learned weights projected to 3D in the PROJECTOR tab, and the profiling traces in the PROFILE tab. The `TensorBoard()` callback has options to log extra data too (see the documentation for more details). You can click the refresh button ( ⟳ ) at the top right to make TensorBoard refresh data, and you can click the settings button ( ⚙ ) to activate auto-refresh and specify the refresh interval. 

Additionally, TensorFlow offers a lower-level API in the `tf.summary` package. The following code creates a `SummaryWriter` using the `create_file_writer()` function, and it uses this writer as a Python context to log scalars, histograms, images, audio, and text, all of which can then be visualized using TensorBoard: 

```
test_logdir=get_run_logdir()
writer=tf.summary.create_file_writer(str(test_logdir))
withwriter.as_default():
forstepinrange(1, 1000+1):
tf.summary.scalar("my_scalar", np.sin(step/10), step=step)
```

```
data= (np.random.randn(100) +2) *step/100# gets larger
tf.summary.histogram("my_hist", data, buckets=50, step=step)
```

```
images=np.random.rand(2, 32, 32, 3) *step/1000# gets brighter
tf.summary.image("my_images", images, step=step)
```

```
texts= ["The step is "+str(step), "Its square is "+str(step**2)]
tf.summary.text("my_text", texts, step=step)
```

```
sine_wave=tf.math.sin(tf.range(12000) /48000*2*np.pi*step)
audio=tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)
```

If you run this code and click the refresh button in TensorBoard, you will see several tabs appear: IMAGES, AUDIO, DISTRIBUTIONS, HISTOGRAMS, and TEXT. Try clicking the IMAGES tab, and use the slider above each image to view the images at different time steps. Similarly, go to the AUDIO tab and try listening to the audio at different time steps. As you can see, TensorBoard is a useful tool even beyond TensorFlow or deep learning. 



You can share your results online by publishing them to _https:// tensorboard.dev_ . For this, just run `!tensorboard dev upload --logdir ./my_logs` . The first time, it will ask you to accept the terms and conditions and authenticate. Then your logs will be uploaded, and you will get a permanent link to view your results in a TensorBoard interface. 

**Implementing MLPs with Keras | 343** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

Let’s summarize what you’ve learned so far in this chapter: you now know where neural nets came from, what an MLP is and how you can use it for classification and regression, how to use Keras’s sequential API to build MLPs, and how to use the functional API or the subclassing API to build more complex model architectures (including Wide & Deep models, as well as models with multiple inputs and outputs). You also learned how to save and restore a model and how to use callbacks for check‐ pointing, early stopping, and more. Finally, you learned how to use TensorBoard for visualization. You can already go ahead and use neural networks to tackle many problems! However, you may wonder how to choose the number of hidden layers, the number of neurons in the network, and all the other hyperparameters. Let’s look at this now. 

#### **Fine-Tuning Neural Network Hyperparameters** 

The flexibility of neural networks is also one of their main drawbacks: there are many hyperparameters to tweak. Not only can you use any imaginable network architecture, but even in a basic MLP you can change the number of layers, the number of neurons and the type of activation function to use in each layer, the weight initialization logic, the type of optimizer to use, its learning rate, the batch size, and more. How do you know what combination of hyperparameters is the best for your task? 

One option is to convert your Keras model to a Scikit-Learn estimator, and then use `GridSearchCV` or `RandomizedSearchCV` to fine-tune the hyperparameters, as you did in Chapter 2. For this, you can use the `KerasRegressor` and `KerasClassifier` wrapper classes from the SciKeras library (see _https://github.com/adriangb/scikeras_ for more details). However, there’s a better way: you can use the _Keras Tuner_ library, which is a hyperparameter tuning library for Keras models. It offers several tuning strategies, it’s highly customizable, and it has excellent integration with TensorBoard. Let’s see how to use it. 

If you followed the installation instructions at _https://homl.info/install_ to run every‐ thing locally, then you already have Keras Tuner installed, but if you are using Colab, you’ll need to run `%pip install -q -U keras-tuner` . Next, import `keras_tuner` , usually as `kt` , then write a function that builds, compiles, and returns a Keras model. The function must take a `kt.HyperParameters` object as an argument, which it can use to define hyperparameters (integers, floats, strings, etc.) along with their range of possible values, and these hyperparameters may be used to build and compile the model. For example, the following function builds and compiles an MLP to clas‐ sify Fashion MNIST images, using hyperparameters such as the number of hidden layers ( `n_hidden` ), the number of neurons per layer ( `n_neurons` ), the learning rate ( `learning_rate` ), and the type of optimizer to use ( `optimizer` ): 

**344 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

```
importkeras_tuneraskt
defbuild_model(hp):
n_hidden=hp.Int("n_hidden", min_value=0, max_value=8, default=2)
n_neurons=hp.Int("n_neurons", min_value=16, max_value=256)
learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
sampling="log")
optimizer=hp.Choice("optimizer", values=["sgd", "adam"])
ifoptimizer=="sgd":
optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate)
else:
optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
for_inrange(n_hidden):
model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
metrics=["accuracy"])
returnmodel
```

The first part of the function defines the hyperparameters. For example, `hp.Int("n_hidden", min_value=0, max_value=8, default=2)` checks whether a hyperparameter named `"n_hidden"` is already present in the `HyperParameters` object `hp` , and if so it returns its value. If not, then it registers a new integer hyperparameter named `"n_hidden"` , whose possible values range from 0 to 8 (inclusive), and it returns the default value, which is 2 in this case (when `default` is not set, then `min_value` is returned). The `"n_neurons"` hyperparameter is registered in a similar way. The `"learning_rate"` hyperparameter is registered as a float ranging from 10<sup>–4</sup> to 10<sup>–2</sup> , and since `sampling="log"` , learning rates of all scales will be sampled equally. Lastly, the `optimizer` hyperparameter is registered with two possible values: `"sgd"` or `"adam"` (the default value is the first one, which is `"sgd"` in this case). Depending on the value of `optimizer` , we create an `SGD` optimizer or an `Adam` optimizer with the given learning rate. 

The second part of the function just builds the model using the hyperparameter values. It creates a `Sequential` model starting with a `Flatten` layer, followed by the requested number of hidden layers (as determined by the `n_hidden` hyperparameter) using the ReLU activation function, and an output layer with 10 neurons (one per class) using the softmax activation function. Lastly, the function compiles the model and returns it. 

Now if you want to do a basic random search, you can create a `kt.RandomSearch` tuner, passing the `build_model` function to the constructor, and call the tuner’s `search()` method: 

**Fine-Tuning Neural Network Hyperparameters | 345** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

```
random_search_tuner=kt.RandomSearch(
build_model, objective="val_accuracy", max_trials=5, overwrite=True,
directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)
random_search_tuner.search(X_train, y_train, epochs=10,
validation_data=(X_valid, y_valid))
```

The `RandomSearch` tuner first calls `build_model()` once with an empty `Hyperparameters` object, just to gather all the hyperparameter specifications. Then, in this example, it runs 5 trials; for each trial it builds a model using hyperparame‐ ters sampled randomly within their respective ranges, then it trains that model for 10 epochs and saves it to a subdirectory of the _my_fashion_mnist/my_rnd_search_ directory. Since `overwrite=True` , the _my_rnd_search_ directory is deleted before train‐ ing starts. If you run this code a second time but with `overwrite=False` and `max_ trials=10` , the tuner will continue tuning where it left off, running 5 more trials: this means you don’t have to run all the trials in one shot. Lastly, since `objective` is set to `"val_accuracy"` , the tuner prefers models with a higher validation accuracy, so once the tuner has finished searching, you can get the best models like this: 

```
top3_models=random_search_tuner.get_best_models(num_models=3)
best_model=top3_models[0]
```

You can also call `get_best_hyperparameters()` to get the `kt.HyperParameters` of the best models: 

```
>>> top3_params=random_search_tuner.get_best_hyperparameters(num_trials=3)
>>> top3_params[0].values# best hyperparameter values
{'n_hidden': 7,
 'n_neurons': 100,
 'learning_rate': 0.0012482904754698163,
 'optimizer': 'sgd'}
```

Each tuner is guided by a so-called _oracle_ : before each trial, the tuner asks the oracle to tell it what the next trial should be. The `RandomSearch` tuner uses a `RandomSearch Oracle` , which is pretty basic: it just picks the next trial randomly, as we saw earlier. Since the oracle keeps track of all the trials, you can ask it to give you the best one, and you can display a summary of that trial: 

```
>>> best_trial=random_search_tuner.oracle.get_best_trials(num_trials=1)[0]
>>> best_trial.summary()
Trial 1 summary
Hyperparameters:
n_hidden: 7
n_neurons: 100
learning_rate: 0.0012482904754698163
optimizer: sgd
Score: 0.8596000075340271
```

**346 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

This shows the best hyperparameters (like earlier), as well as the validation accuracy. You can also access all the metrics directly: 

```
>>> best_trial.metrics.get_last_value("val_accuracy")
0.8596000075340271
```

If you are happy with the best model’s performance, you may continue training it for a few epochs on the full training set ( `X_train_full` and `y_train_full` ), then evaluate it on the test set, and deploy it to production (see Chapter 19): 

```
best_model.fit(X_train_full, y_train_full, epochs=10)
test_loss, test_accuracy=best_model.evaluate(X_test, y_test)
```

In some cases, you may want to fine-tune data preprocessing hyperparameters, or `model.fit()` arguments, such as the batch size. For this, you must use a slightly different technique: instead of writing a `build_model()` function, you must subclass the `kt.HyperModel` class and define two methods, `build()` and `fit()` . The `build()` method does the exact same thing as the `build_model()` function. The `fit()` method takes a `HyperParameters` object and a compiled model as an argument, as well as all the `model.fit()` arguments, and fits the model and returns the `History` object. Crucially, the `fit()` method may use hyperparameters to decide how to preprocess the data, tweak the batch size, and more. For example, the following class builds the same model as before, with the same hyperparameters, but it also uses a Boolean `"normalize"` hyperparameter to control whether or not to standardize the training data before fitting the model: 

```
classMyClassificationHyperModel(kt.HyperModel):
defbuild(self, hp):
returnbuild_model(hp)
```

```
deffit(self, hp, model, X, y, **kwargs):
ifhp.Boolean("normalize"):
norm_layer=tf.keras.layers.Normalization()
X=norm_layer(X)
returnmodel.fit(X, y, **kwargs)
```

You can then pass an instance of this class to the tuner of your choice, instead of passing the `build_model` function. For example, let’s build a `kt.Hyperband` tuner based on a `MyClassificationHyperModel` instance: 

```
hyperband_tuner=kt.Hyperband(
MyClassificationHyperModel(), objective="val_accuracy", seed=42,
max_epochs=10, factor=3, hyperband_iterations=2,
overwrite=True, directory="my_fashion_mnist", project_name="hyperband")
```

This tuner is similar to the `HalvingRandomSearchCV` class we discussed in Chapter 2: it starts by training many different models for few epochs, then it eliminates the worst models and keeps only the top `1 / factor` models (i.e., the top third in this 

**Fine-Tuning Neural Network Hyperparameters | 347** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

case), repeating this selection process until a single model is left.<sup>19</sup> The `max_epochs` argument controls the max number of epochs that the best model will be trained for. The whole process is repeated twice in this case ( `hyperband_iterations=2` ). The total number of training epochs across all models for each hyperband iteration is about `max_epochs * (log(max_epochs) / log(factor)) ** 2` , so it’s about 44 epochs in this example. The other arguments are the same as for `kt.RandomSearch` . 

Let’s run the Hyperband tuner now. We’ll use the `TensorBoard` callback, this time pointing to the root log directory (the tuner will take care of using a different subdirectory for each trial), as well as an `EarlyStopping` callback: 

```
root_logdir=Path(hyperband_tuner.project_dir) /"tensorboard"
tensorboard_cb=tf.keras.callbacks.TensorBoard(root_logdir)
early_stopping_cb=tf.keras.callbacks.EarlyStopping(patience=2)
hyperband_tuner.search(X_train, y_train, epochs=10,
validation_data=(X_valid, y_valid),
callbacks=[early_stopping_cb, tensorboard_cb])
```

Now if you open TensorBoard, pointing `--logdir` to the _my_fashion_mnist/hyper‐ band/tensorboard_ directory, you will see all the trial results as they unfold. Make sure to visit the HPARAMS tab: it contains a summary of all the hyperparameter combinations that were tried, along with the corresponding metrics. Notice that there are three tabs inside the HPARAMS tab: a table view, a parallel coordinates view, and a scatterplot matrix view. In the lower part of the left panel, uncheck all metrics except for `validation.epoch_accuracy` : this will make the graphs clearer. In the parallel coordinates view, try selecting a range of high values in the `validation.epoch_accuracy` column: this will filter only the hyperparameter combinations that reached a good performance. Click one of the hyperparameter combinations, and the corresponding learning curves will appear at the bottom of the page. Take some time to go through each tab; this will help you understand the effect of each hyperparameter on performance, as well as the interactions between the hyperparameters. 

Hyperband is smarter than pure random search in the way it allocates resources, but at its core it still explores the hyperparameter space randomly; it’s fast, but coarse. However, Keras Tuner also includes a `kt.BayesianOptimization` tuner: this algorithm gradually learns which regions of the hyperparameter space are most promising by fitting a probabilistic model called a _Gaussian process_ . This allows it to gradually zoom in on the best hyperparameters. The downside is that the algorithm has its own hyperparameters: `alpha` represents the level of noise you expect in the performance measures across trials (it defaults to 10<sup>–4</sup> ), and `beta` specifies how much 

> 19 Hyperband is actually a bit more sophisticated than successive halving; see the paper by Lisha Li et al., “Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization”, _Journal of Machine Learning Research_ 18 (April 2018): 1–52. 

**348 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

you want the algorithm to explore, instead of simply exploiting the known good regions of hyperparameter space (it defaults to 2.6). Other than that, this tuner can be used just like the previous ones: 

```
bayesian_opt_tuner=kt.BayesianOptimization(
MyClassificationHyperModel(), objective="val_accuracy", seed=42,
max_trials=10, alpha=1e-4, beta=2.6,
```

```
overwrite=True, directory="my_fashion_mnist", project_name="bayesian_opt")
bayesian_opt_tuner.search([...])
```

Hyperparameter tuning is still an active area of research, and many other approaches are being explored. For example, check out DeepMind’s excellent 2017 paper,<sup>20</sup> where the authors used an evolutionary algorithm to jointly optimize a population of mod‐ els and their hyperparameters. Google has also used an evolutionary approach, not just to search for hyperparameters but also to explore all sorts of model architectures: it powers their AutoML service on Google Vertex AI (see Chapter 19). The term _AutoML_ refers to any system that takes care of a large part of the ML workflow. Evolutionary algorithms have even been used successfully to train individual neural networks, replacing the ubiquitous gradient descent! For an example, see the 2017 post by Uber where the authors introduce their _Deep Neuroevolution_ technique. 

But despite all this exciting progress and all these tools and services, it still helps to have an idea of what values are reasonable for each hyperparameter so that you can build a quick prototype and restrict the search space. The following sections provide guidelines for choosing the number of hidden layers and neurons in an MLP and for selecting good values for some of the main hyperparameters. 

##### **Number of Hidden Layers** 

For many problems, you can begin with a single hidden layer and get reasonable results. An MLP with just one hidden layer can theoretically model even the most complex functions, provided it has enough neurons. But for complex problems, deep networks have a much higher _parameter efficiency_ than shallow ones: they can model complex functions using exponentially fewer neurons than shallow nets, allowing them to reach much better performance with the same amount of training data. 

To understand why, suppose you are asked to draw a forest using some drawing soft‐ ware, but you are forbidden to copy and paste anything. It would take an enormous amount of time: you would have to draw each tree individually, branch by branch, leaf by leaf. If you could instead draw one leaf, copy and paste it to draw a branch, then copy and paste that branch to create a tree, and finally copy and paste this tree to make a forest, you would be finished in no time. Real-world data is often structured 

> 20 Max Jaderberg et al., “Population Based Training of Neural Networks”, arXiv preprint arXiv:1711.09846 (2017). 

**Fine-Tuning Neural Network Hyperparameters | 349** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

in such a hierarchical way, and deep neural networks automatically take advantage of this fact: lower hidden layers model low-level structures (e.g., line segments of various shapes and orientations), intermediate hidden layers combine these low-level struc‐ tures to model intermediate-level structures (e.g., squares, circles), and the highest hidden layers and the output layer combine these intermediate structures to model high-level structures (e.g., faces). 

Not only does this hierarchical architecture help DNNs converge faster to a good solution, but it also improves their ability to generalize to new datasets. For example, if you have already trained a model to recognize faces in pictures and you now want to train a new neural network to recognize hairstyles, you can kickstart the training by reusing the lower layers of the first network. Instead of randomly initializing the weights and biases of the first few layers of the new neural network, you can initialize them to the values of the weights and biases of the lower layers of the first network. This way the network will not have to learn from scratch all the low-level structures that occur in most pictures; it will only have to learn the higher-level structures (e.g., hairstyles). This is called _transfer learning_ . 

In summary, for many problems you can start with just one or two hidden layers and the neural network will work just fine. For instance, you can easily reach above 97% accuracy on the MNIST dataset using just one hidden layer with a few hundred neurons, and above 98% accuracy using two hidden layers with the same total number of neurons, in roughly the same amount of training time. For more complex problems, you can ramp up the number of hidden layers until you start overfitting the training set. Very complex tasks, such as large image classification or speech recognition, typically require networks with dozens of layers (or even hundreds, but not fully connected ones, as you will see in Chapter 14), and they need a huge amount of training data. You will rarely have to train such networks from scratch: it is much more common to reuse parts of a pretrained state-of-the-art network that performs a similar task. Training will then be a lot faster and require much less data (we will discuss this in Chapter 11). 

##### **Number of Neurons per Hidden Layer** 

The number of neurons in the input and output layers is determined by the type of input and output your task requires. For example, the MNIST task requires 28 × 28 = 784 inputs and 10 output neurons. 

As for the hidden layers, it used to be common to size them to form a pyramid, with fewer and fewer neurons at each layer—the rationale being that many low-level features can coalesce into far fewer high-level features. A typical neural network for MNIST might have 3 hidden layers, the first with 300 neurons, the second with 200, and the third with 100. However, this practice has been largely abandoned because it seems that using the same number of neurons in all hidden layers performs just 

**350 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

as well in most cases, or even better; plus, there is only one hyperparameter to tune, instead of one per layer. That said, depending on the dataset, it can sometimes help to make the first hidden layer bigger than the others. 

Just like the number of layers, you can try increasing the number of neurons gradu‐ ally until the network starts overfitting. Alternatively, you can try building a model with slightly more layers and neurons than you actually need, then use early stopping and other regularization techniques to prevent it from overfitting too much. Vincent Vanhoucke, a scientist at Google, has dubbed this the “stretch pants” approach: instead of wasting time looking for pants that perfectly match your size, just use large stretch pants that will shrink down to the right size. With this approach, you avoid bottleneck layers that could ruin your model. Indeed, if a layer has too few neurons, it will not have enough representational power to preserve all the useful information from the inputs (e.g., a layer with two neurons can only output 2D data, so if it gets 3D data as input, some information will be lost). No matter how big and powerful the rest of the network is, that information will never be recovered. 



In general you will get more bang for your buck by increasing the number of layers instead of the number of neurons per layer. 

##### **Learning Rate, Batch Size, and Other Hyperparameters** 

The number of hidden layers and neurons are not the only hyperparameters you can tweak in an MLP. Here are some of the most important ones, as well as tips on how to set them: 

###### _Learning rate_ 

The learning rate is arguably the most important hyperparameter. In general, the optimal learning rate is about half of the maximum learning rate (i.e., the learn‐ ing rate above which the training algorithm diverges, as we saw in Chapter 4). One way to find a good learning rate is to train the model for a few hundred iter‐ ations, starting with a very low learning rate (e.g., 10<sup>–5</sup> ) and gradually increasing it up to a very large value (e.g., 10). This is done by multiplying the learning rate by a constant factor at each iteration (e.g., by (10 / 10<sup>-5</sup> )<sup>1 / 500</sup> to go from 10<sup>–5</sup> to 10 in 500 iterations). If you plot the loss as a function of the learning rate (using a log scale for the learning rate), you should see it dropping at first. But after a while, the learning rate will be too large, so the loss will shoot back up: the optimal learning rate will be a bit lower than the point at which the loss starts to climb (typically about 10 times lower than the turning point). You can then reinitialize your model and train it normally using this good learning rate. We will look at more learning rate optimization techniques in Chapter 11. 

**Fine-Tuning Neural Network Hyperparameters | 351** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

###### _Optimizer_ 

Choosing a better optimizer than plain old mini-batch gradient descent (and tuning its hyperparameters) is also quite important. We will examine several advanced optimizers in Chapter 11. 

###### _Batch size_ 

The batch size can have a significant impact on your model’s performance and training time. The main benefit of using large batch sizes is that hardware accel‐ erators like GPUs can process them efficiently (see Chapter 19), so the training algorithm will see more instances per second. Therefore, many researchers and practitioners recommend using the largest batch size that can fit in GPU RAM. There’s a catch, though: in practice, large batch sizes often lead to training instabilities, especially at the beginning of training, and the resulting model may not generalize as well as a model trained with a small batch size. In April 2018, Yann LeCun even tweeted “Friends don’t let friends use mini-batches larger than 32”, citing a 2018 paper<sup>21</sup> by Dominic Masters and Carlo Luschi which concluded that using small batches (from 2 to 32) was preferable because small batches led to better models in less training time. Other research points in the opposite direction, however. For example, in 2017, papers by Elad Hoffer et al.<sup>22</sup> and Priya Goyal et al.<sup>23</sup> showed that it was possible to use very large batch sizes (up to 8,192) along with various techniques such as warming up the learning rate (i.e., starting training with a small learning rate, then ramping it up, as discussed in Chapter 11) and to obtain very short training times, without any generalization gap. So, one strategy is to try to using a large batch size, with learning rate warmup, and if training is unstable or the final performance is disappointing, then try using a small batch size instead. 

###### _Activation function_ 

We discussed how to choose the activation function earlier in this chapter: in general, the ReLU activation function will be a good default for all hidden layers, but for the output layer it really depends on your task. 

###### _Number of iterations_ 

In most cases, the number of training iterations does not actually need to be tweaked: just use early stopping instead. 

- 21 Dominic Masters and Carlo Luschi, “Revisiting Small Batch Training for Deep Neural Networks”, arXiv preprint arXiv:1804.07612 (2018). 

- 22 Elad Hoffer et al., “Train Longer, Generalize Better: Closing the Generalization Gap in Large Batch Training of Neural Networks”, _Proceedings of the 31st International Conference on Neural Information Processing Systems_ (2017): 1729–1739. 

- 23 Priya Goyal et al., “Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour”, arXiv preprint arXiv:1706.02677 (2017). 

###### **352 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 



The optimal learning rate depends on the other hyperparameters— especially the batch size—so if you modify any hyperparameter, make sure to update the learning rate as well. 

For more best practices regarding tuning neural network hyperparameters, check out the excellent 2018 paper<sup>24</sup> by Leslie Smith. 

This concludes our introduction to artificial neural networks and their implemen‐ tation with Keras. In the next few chapters, we will discuss techniques to train very deep nets. We will also explore how to customize models using TensorFlow’s lower-level API and how to load and preprocess data efficiently using the tf.data API. And we will dive into other popular neural network architectures: convolutional neural networks for image processing, recurrent neural networks and transformers for sequential data and text, autoencoders for representation learning, and generative adversarial networks to model and generate data.<sup>25</sup> 

#### **Exercises** 

**1.** The TensorFlow playground is a handy neural network simulator built by the TensorFlow team. In this exercise, you will train several binary classifiers in just a few clicks, and tweak the model’s architecture and its hyperparameters to gain some intuition on how neural networks work and what their hyperparameters do. Take some time to explore the following: 

   - **a.** The patterns learned by a neural net. Try training the default neural network by clicking the Run button (top left). Notice how it quickly finds a good solution for the classification task. The neurons in the first hidden layer have learned simple patterns, while the neurons in the second hidden layer have learned to combine the simple patterns of the first hidden layer into more complex patterns. In general, the more layers there are, the more complex the patterns can be. 

   - **b.** Activation functions. Try replacing the tanh activation function with a ReLU activation function, and train the network again. Notice that it finds a solution even faster, but this time the boundaries are linear. This is due to the shape of the ReLU function. 

   - **c.** The risk of local minima. Modify the network architecture to have just one hidden layer with three neurons. Train it multiple times (to reset the network 

> 24 Leslie N. Smith, “A Disciplined Approach to Neural Network Hyper-Parameters: Part 1—Learning Rate, Batch Size, Momentum, and Weight Decay”, arXiv preprint arXiv:1803.09820 (2018). 

> 25 A few extra ANN architectures are presented in the online notebook at _https://homl.info/extra-anns_ . 

**Exercises | 353** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

weights, click the Reset button next to the Play button). Notice that the train‐ ing time varies a lot, and sometimes it even gets stuck in a local minimum. 

   - **d.** What happens when neural nets are too small. Remove one neuron to keep just two. Notice that the neural network is now incapable of finding a good solution, even if you try multiple times. The model has too few parameters and systematically underfits the training set. 

   - **e.** What happens when neural nets are large enough. Set the number of neurons to eight, and train the network several times. Notice that it is now consistently fast and never gets stuck. This highlights an important finding in neural network theory: large neural networks rarely get stuck in local minima, and even when they do these local optima are often almost as good as the global optimum. However, they can still get stuck on long plateaus for a long time. 

   - **f.** The risk of vanishing gradients in deep networks. Select the spiral dataset (the bottom-right dataset under “DATA”), and change the network architecture to have four hidden layers with eight neurons each. Notice that training takes much longer and often gets stuck on plateaus for long periods of time. Also notice that the neurons in the highest layers (on the right) tend to evolve faster than the neurons in the lowest layers (on the left). This problem, called the _vanishing gradients_ problem, can be alleviated with better weight initializa‐ tion and other techniques, better optimizers (such as AdaGrad or Adam), or batch normalization (discussed in Chapter 11). 

   - **g.** Go further. Take an hour or so to play around with other parameters and get a feel for what they do, to build an intuitive understanding about neural networks. 

**2.** Draw an ANN using the original artificial neurons (like the ones in Figure 10-3) that computes _A_ ⊕ _B_ (where ⊕ represents the XOR operation). Hint: _A_ ⊕ _B_ = ( _A_ ∧ ¬ _B_ ) ∨ (¬ _A_ ∧ _B_ ). 

**3.** Why is it generally preferable to use a logistic regression classifier rather than a classic perceptron (i.e., a single layer of threshold logic units trained using the perceptron training algorithm)? How can you tweak a perceptron to make it equivalent to a logistic regression classifier? 

**4.** Why was the sigmoid activation function a key ingredient in training the first MLPs? 

**5.** Name three popular activation functions. Can you draw them? 

**6.** Suppose you have an MLP composed of one input layer with 10 passthrough neurons, followed by one hidden layer with 50 artificial neurons, and finally one output layer with 3 artificial neurons. All artificial neurons use the ReLU activation function. 

   - **a.** What is the shape of the input matrix **X** ? 

###### **354 | Chapter 10: Introduction to Artificial Neural Networks with Keras** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

   - **b.** What are the shapes of the hidden layer’s weight matrix **W** _h_ and bias vector **b** _h_ ? 

   - **c.** What are the shapes of the output layer’s weight matrix **W** _o_ and bias vector **b** _o_ ? 

   - **d.** What is the shape of the network’s output matrix **Y** ? 

   - **e.** Write the equation that computes the network’s output matrix **Y** as a function of **X** , **W** _h_ , **b** _h_ , **W** _o_ , and **b** _o_ . 

**7.** How many neurons do you need in the output layer if you want to classify email into spam or ham? What activation function should you use in the output layer? If instead you want to tackle MNIST, how many neurons do you need in the output layer, and which activation function should you use? What about for getting your network to predict housing prices, as in Chapter 2? 

**8.** What is backpropagation and how does it work? What is the difference between backpropagation and reverse-mode autodiff? 

**9.** Can you list all the hyperparameters you can tweak in a basic MLP? If the MLP overfits the training data, how could you tweak these hyperparameters to try to solve the problem? 

**10.** Train a deep MLP on the MNIST dataset (you can load it using `tf.keras. datasets.mnist.load_data()` ). See if you can get over 98% accuracy by man‐ ually tuning the hyperparameters. Try searching for the optimal learning rate by using the approach presented in this chapter (i.e., by growing the learning rate exponentially, plotting the loss, and finding the point where the loss shoots up). Next, try tuning the hyperparameters using Keras Tuner with all the bells and whistles—save checkpoints, use early stopping, and plot learning curves using TensorBoard. 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

**Exercises | 355** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:01. 

