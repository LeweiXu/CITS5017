# **CHAPTER 11 Training Deep Neural Networks** 

In Chapter 10 you built, trained, and fine-tuned your first artificial neural networks. But they were shallow nets, with just a few hidden layers. What if you need to tackle a complex problem, such as detecting hundreds of types of objects in high-resolution images? You may need to train a much deeper ANN, perhaps with 10 layers or many more, each containing hundreds of neurons, linked by hundreds of thousands of connections. Training a deep neural network isn’t a walk in the park. Here are some of the problems you could run into: 

- You may be faced with the problem of gradients growing ever smaller or larger, when flowing backward through the DNN during training. Both of these prob‐ lems make lower layers very hard to train. 

- You might not have enough training data for such a large network, or it might be too costly to label. 

- Training may be extremely slow. 

- A model with millions of parameters would severely risk overfitting the training set, especially if there are not enough training instances or if they are too noisy. 

In this chapter we will go through each of these problems and present techniques to solve them. We will start by exploring the vanishing and exploding gradients problems and some of their most popular solutions. Next, we will look at transfer learning and unsupervised pretraining, which can help you tackle complex tasks even when you have little labeled data. Then we will discuss various optimizers that can speed up training large models tremendously. Finally, we will cover a few popular regularization techniques for large neural networks. 

With these tools, you will be able to train very deep nets. Welcome to deep learning! 

**357** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

## **The Vanishing/Exploding Gradients Problems** 

As discussed in Chapter 10, the backpropagation algorithm’s second phase works by going from the output layer to the input layer, propagating the error gradient along the way. Once the algorithm has computed the gradient of the cost function with regard to each parameter in the network, it uses these gradients to update each parameter with a gradient descent step. 

Unfortunately, gradients often get smaller and smaller as the algorithm progresses down to the lower layers. As a result, the gradient descent update leaves the lower layers’ connection weights virtually unchanged, and training never converges to a good solution. This is called the _vanishing gradients_ problem. In some cases, the opposite can happen: the gradients can grow bigger and bigger until layers get insanely large weight updates and the algorithm diverges. This is the _exploding gradients_ problem, which surfaces most often in recurrent neural networks (see Chapter 15). More generally, deep neural networks suffer from unstable gradients; different layers may learn at widely different speeds. 

This unfortunate behavior was empirically observed long ago, and it was one of the reasons deep neural networks were mostly abandoned in the early 2000s. It wasn’t clear what caused the gradients to be so unstable when training a DNN, but some light was shed in a 2010 paper by Xavier Glorot and Yoshua Bengio.<sup>1</sup> The authors found a few suspects, including the combination of the popular sigmoid (logistic) activation function and the weight initialization technique that was most popular at the time (i.e., a normal distribution with a mean of 0 and a standard deviation of 1). In short, they showed that with this activation function and this initialization scheme, the variance of the outputs of each layer is much greater than the variance of its inputs. Going forward in the network, the variance keeps increasing after each layer until the activation function saturates at the top layers. This saturation is actually made worse by the fact that the sigmoid function has a mean of 0.5, not 0 (the hyperbolic tangent function has a mean of 0 and behaves slightly better than the sigmoid function in deep networks). 

Looking at the sigmoid activation function (see Figure 11-1), you can see that when inputs become large (negative or positive), the function saturates at 0 or 1, with a derivative extremely close to 0 (i.e., the curve is flat at both extremes). Thus, when backpropagation kicks in it has virtually no gradient to propagate back through the network, and what little gradient exists keeps getting diluted as backpropagation progresses down through the top layers, so there is really nothing left for the lower layers. 

> 1 Xavier Glorot and Yoshua Bengio, “Understanding the Difficulty of Training Deep Feedforward Neural Networks”, _Proceedings of the 13th International Conference on Artificial Intelligence and Statistics_ (2010): 249– 256. 

###### **358 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 



<!-- Start of picture text -->
1.2<br>1<br>—_ O(Z)=.—— -<br>0.8 SaturatingF<br>0.6<br>Saturating<br>02 Linear<br>-0.2 “<br>-4 2 () 2 4<br>Zz<br><!-- End of picture text -->

_Equation 11-1. Glorot initialization (when using the sigmoid activation function)_ 

1 Normal distribution with mean 0 and variance σ<sup>2</sup> = fan avg 3 Or a uniform distribution between − r and + r , with r = fan avg 

If you replace _fan_ avg with _fan_ in in Equation 11-1, you get an initialization strategy that Yann LeCun proposed in the 1990s. He called it _LeCun initialization_ . Genevieve Orr and Klaus-Robert Müller even recommended it in their 1998 book _Neural Networks: Tricks of the Trade_ (Springer). LeCun initialization is equivalent to Glorot initializa‐ tion when _fan_ in = _fan_ out. It took over a decade for researchers to realize how important this trick is. Using Glorot initialization can speed up training considerably, and it is one of the practices that led to the success of deep learning. 

Some papers<sup>3</sup> have provided similar strategies for different activation functions. These strategies differ only by the scale of the variance and whether they use _fan_ avg or _fan_ in, as shown in Table 11-1 (for the uniform distribution, just use r = 3 σ<sup>2</sup> ). The initialization strategy proposed for the ReLU activation function and its variants is called _He initialization_ or _Kaiming initialization_ , after the paper’s first author. For SELU, use Yann LeCun’s initialization method, preferably with a normal distribution. We will cover all these activation functions shortly. 

_Table 11-1. Initialization parameters for each type of activation function_ 

|**Initialization**|**Activation functions**|**_σ_² (Normal)**|
|---|---|---|
|Glorot|None, tanh, sigmoid, softmax|1 /_fan_avg|
|He|ReLU, Leaky ReLU, ELU, GELU, Swish, Mish|2 /_fan_in|
|LeCun|SELU|1 /_fan_in|



By default, Keras uses Glorot initialization with a uniform distribution. When you create a layer, you can switch to He initialization by setting `kernel_initializer= "he_uniform"` or `kernel_initializer="he_normal"` like this: 

```
importtensorflowastf
dense=tf.keras.layers.Dense(50, activation="relu",
kernel_initializer="he_normal")
```

Alternatively, you can obtain any of the initializations listed in Table 11-1 and more using the `VarianceScaling` initializer. For example, if you want He initialization 

> 3 E.g., Kaiming He et al., “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,” _Proceedings of the 2015 IEEE International Conference on Computer Vision_ (2015): 1026–1034. 

**360 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

with a uniform distribution and based on _fan_ avg (rather than _fan_ in), you can use the following code: 

```
he_avg_init=tf.keras.initializers.VarianceScaling(scale=2., mode="fan_avg",
distribution="uniform")
dense=tf.keras.layers.Dense(50, activation="sigmoid",
kernel_initializer=he_avg_init)
```

### **Better Activation Functions** 

One of the insights in the 2010 paper by Glorot and Bengio was that the problems with unstable gradients were in part due to a poor choice of activation function. Until then most people had assumed that if Mother Nature had chosen to use roughly sigmoid activation functions in biological neurons, they must be an excellent choice. But it turns out that other activation functions behave much better in deep neural networks—in particular, the ReLU activation function, mostly because it does not saturate for positive values, and also because it is very fast to compute. 

Unfortunately, the ReLU activation function is not perfect. It suffers from a problem known as the _dying ReLUs_ : during training, some neurons effectively “die”, meaning they stop outputting anything other than 0. In some cases, you may find that half of your network’s neurons are dead, especially if you used a large learning rate. A neuron dies when its weights get tweaked in such a way that the input of the ReLU function (i.e., the weighted sum of the neuron’s inputs plus its bias term) is negative for all instances in the training set. When this happens, it just keeps outputting zeros, and gradient descent does not affect it anymore because the gradient of the ReLU function is zero when its input is negative.<sup>4</sup> 

To solve this problem, you may want to use a variant of the ReLU function, such as the _leaky ReLU_ . 

##### **Leaky ReLU** 

The leaky ReLU activation function is defined as LeakyReLU _α_ ( _z_ ) = max( _αz_ , _z_ ) (see Figure 11-2). The hyperparameter _α_ defines how much the function “leaks”: it is the slope of the function for _z_ < 0. Having a slope for _z_ < 0 ensures that leaky ReLUs never die; they can go into a long coma, but they have a chance to eventually wake up. A 2015 paper by Bing Xu et al.<sup>5</sup> compared several variants of the ReLU activation function, and one of its conclusions was that the leaky variants always 

> 4 A dead neuron may come back to life if its inputs evolve over time and eventually return within a range where the ReLU activation function gets a positive input again. For example, this may happen if gradient descent tweaks the neurons in the layers below the dead neuron. 

> 5 Bing Xu et al., “Empirical Evaluation of Rectified Activations in Convolutional Network,” arXiv preprint arXiv:1505.00853 (2015). 

**The Vanishing/Exploding Gradients Problems | 361** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 



<!-- Start of picture text -->
34 LeakyReLU(z) = max(az, Z)<br>2<br>1<br>Leak<br>|0a-4 -2 0 | 2 4<br>Zz<br><!-- End of picture text -->

ReLU, leaky ReLU, and PReLU all suffer from the fact that they are not smooth functions: their derivatives abruptly change (at _z_ = 0). As we saw in Chapter 4 when we discussed lasso, this sort of discontinuity can make gradient descent bounce around the optimum, and slow down convergence. So now we will look at some smooth variants of the ReLU activation function, starting with ELU and SELU. 

##### **ELU and SELU** 

In 2015, a paper by Djork-Arné Clevert et al.<sup>6</sup> proposed a new activation function, called the _exponential linear unit_ (ELU), that outperformed all the ReLU variants in the authors’ experiments: training time was reduced, and the neural network performed better on the test set. Equation 11-2 shows this activation function’s definition. 

_Equation 11-2. ELU activation function_ 



The ELU activation function looks a lot like the ReLU function (see Figure 11-3), with a few major differences: 

- It takes on negative values when _z_ < 0, which allows the unit to have an average output closer to 0 and helps alleviate the vanishing gradients problem. The hyperparameter _α_ defines the opposite of the value that the ELU function approaches when _z_ is a large negative number. It is usually set to 1, but you can tweak it like any other hyperparameter. 

- It has a nonzero gradient for _z_ < 0, which avoids the dead neurons problem. 

- If _α_ is equal to 1 then the function is smooth everywhere, including around _z_ = 0, which helps speed up gradient descent since it does not bounce as much to the left and right of _z_ = 0. 

Using ELU with Keras is as easy as setting `activation="elu"` , and like with other ReLU variants, you should use He initialization. The main drawback of the ELU activation function is that it is slower to compute than the ReLU function and its variants (due to the use of the exponential function). Its faster convergence rate during training may compensate for that slow computation, but still, at test time an ELU network will be a bit slower than a ReLU network. 

> 6 Djork-Arné Clevert et al., “Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs),” _Proceedings of the International Conference on Learning Representations_ , arXiv preprint (2015). 

**The Vanishing/Exploding Gradients Problems | 363** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 



<!-- Start of picture text -->
3 "7<br>— ELU,(z) = a(e?— 1) if z<0, elsez 7<br>24.777 SELU(zZ)= 1.05 ELUj 67(z)<br>1<br>0<br>7<br>Poked ada heteeeeeereeeeeeeereeee) See e eee eee eee eee eee ee eee eee eee eee)<br>><br>-4 —2 0 2 4<br>Zz<br><!-- End of picture text -->



<!-- Start of picture text -->
2.0<br>©] — GELU(z) =z9(z)<br>=== Swish(z)<br>= za(z)<br>1.0 weeeee Swishg = 0.6(Z) = Z0(0.6 2)<br>ost" Mish(z) =ztanh(softplus(z))<br>-1.0a<br>-4 -3 -2 -1 0 I 3<br>Zz<br><!-- End of picture text -->

The GELU paper also introduced the _sigmoid linear unit_ (SiLU) activation function, which is equal to _z_ σ( _z_ ), but it was outperformed by GELU in the authors’ tests. Inter‐ estingly, a 2017 paper by Prajit Ramachandran et al.<sup>10</sup> rediscovered the SiLU function by automatically searching for good activation functions. The authors named it _Swish_ , and the name caught on. In their paper, Swish outperformed every other function, including GELU. Ramachandran et al. later generalized Swish by adding an extra hyperparameter _β_ to scale the sigmoid function’s input. The generalized Swish func‐ tion is Swish _β_ ( _z_ ) = _z_ σ( _βz_ ), so GELU is approximately equal to the generalized Swish function using _β_ = 1.702. You can tune _β_ like any other hyperparameter. Alternatively, it’s also possible to make _β_ trainable and let gradient descent optimize it: much like PReLU, this can make your model more powerful, but it also runs the risk of overfitting the data. 

Another quite similar activation function is _Mish_ , which was introduced in a 2019 paper by Diganta Misra.<sup>11</sup> It is defined as mish( _z_ ) = _z_ tanh(softplus( _z_ )), where soft‐ plus( _z_ ) = log(1 + exp( _z_ )). Just like GELU and Swish, it is a smooth, nonconvex, and nonmonotonic variant of ReLU, and once again the author ran many experiments and found that Mish generally outperformed other activation functions—even Swish and GELU, by a tiny margin. Figure 11-4 shows GELU, Swish (both with the default _β_ = 1 and with _β_ = 0.6), and lastly Mish. As you can see, Mish overlaps almost perfectly with Swish when _z_ is negative, and almost perfectly with GELU when _z_ is positive. 



So, which activation function should you use for the hidden layers of your deep neural networks? ReLU remains a good default for simple tasks: it’s often just as good as the more sophisticated activa‐ tion functions, plus it’s very fast to compute, and many libraries and hardware accelerators provide ReLU-specific optimizations. However, Swish is probably a better default for more complex tasks, and you can even try parametrized Swish with a learnable _β_ parameter for the most complex tasks. Mish may give you slightly better results, but it requires a bit more compute. If you care a lot about runtime latency, then you may prefer leaky ReLU, or parametrized leaky ReLU for more complex tasks. For deep MLPs, give SELU a try, but make sure to respect the constraints listed earlier. If you have spare time and computing power, you can use cross-validation to evaluate other activation functions as well. 

10 Prajit Ramachandran et al., “Searching for Activation Functions”, arXiv preprint arXiv:1710.05941 (2017). 

11 Diganta Misra, “Mish: A Self Regularized Non-Monotonic Activation Function”, arXiv preprint arXiv:1908.08681 (2019). 

###### **366 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

Keras supports GELU and Swish out of the box; just use `activation="gelu"` or `activation="swish"` . However, it does not support Mish or the generalized Swish activation function yet (but see Chapter 12 to see how to implement your own activation functions and layers). 

That’s all for activation functions! Now, let’s look at a completely different way to solve the unstable gradients problem: batch normalization. 

### **Batch Normalization** 

Although using He initialization along with ReLU (or any of its variants) can sig‐ nificantly reduce the danger of the vanishing/exploding gradients problems at the beginning of training, it doesn’t guarantee that they won’t come back during training. 

In a 2015 paper,<sup>12</sup> Sergey Ioffe and Christian Szegedy proposed a technique called _batch normalization_ (BN) that addresses these problems. The technique consists of adding an operation in the model just before or after the activation function of each hidden layer. This operation simply zero-centers and normalizes each input, then scales and shifts the result using two new parameter vectors per layer: one for scaling, the other for shifting. In other words, the operation lets the model learn the optimal scale and mean of each of the layer’s inputs. In many cases, if you add a BN layer as the very first layer of your neural network, you do not need to standardize your training set. That is, there’s no need for `StandardScaler` or `Normalization` ; the BN layer will do it for you (well, approximately, since it only looks at one batch at a time, and it can also rescale and shift each input feature). 

In order to zero-center and normalize the inputs, the algorithm needs to estimate each input’s mean and standard deviation. It does so by evaluating the mean and standard deviation of the input over the current mini-batch (hence the name “batch normalization”). The whole operation is summarized step by step in Equation 11-4. 

> 12 Sergey Ioffe and Christian Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift”, _Proceedings of the 32nd International Conference on Machine Learning_ (2015): 448– 456. 

**The Vanishing/Exploding Gradients Problems | 367** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

_Equation 11-4. Batch normalization algorithm_ 



In this algorithm: 

- **μ** _B_ is the vector of input means, evaluated over the whole mini-batch _B_ (it contains one mean per input). 

- _mB_ is the number of instances in the mini-batch. 

- **σ** _B_ is the vector of input standard deviations, also evaluated over the whole mini-batch (it contains one standard deviation per input). 

- x ( _i_ ) is the vector of zero-centered and normalized inputs for instance _i_ . 

- _ε_ is a tiny number that avoids division by zero and ensures the gradients don’t grow too large (typically 10<sup>–5</sup> ). This is called a _smoothing term_ . 

- **γ** is the output scale parameter vector for the layer (it contains one scale parame‐ ter per input). 

- ⊗ represents element-wise multiplication (each input is multiplied by its corre‐ sponding output scale parameter). 

- **β** is the output shift (offset) parameter vector for the layer (it contains one offset parameter per input). Each input is offset by its corresponding shift parameter. 

- **z**<sup>(</sup><sup>_i_)</sup> is the output of the BN operation. It is a rescaled and shifted version of the inputs. 

So during training, BN standardizes its inputs, then rescales and offsets them. Good! What about at test time? Well, it’s not that simple. Indeed, we may need to make predictions for individual instances rather than for batches of instances: in this case, we will have no way to compute each input’s mean and standard deviation. Moreover, even if we do have a batch of instances, it may be too small, or the instances may not be independent and identically distributed, so computing statistics over the batch instances would be unreliable. One solution could be to wait until the end of training, then run the whole training set through the neural network and compute the mean and standard deviation of each input of the BN layer. These “final” input means 

**368 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

and standard deviations could then be used instead of the batch input means and standard deviations when making predictions. However, most implementations of batch normalization estimate these final statistics during training by using a moving average of the layer’s input means and standard deviations. This is what Keras does automatically when you use the `BatchNormalization` layer. To sum up, four parame‐ ter vectors are learned in each batch-normalized layer: **γ** (the output scale vector) and **β** (the output offset vector) are learned through regular backpropagation, and **μ** (the final input mean vector) and **σ** (the final input standard deviation vector) are estimated using an exponential moving average. Note that **μ** and **σ** are estimated during training, but they are used only after training (to replace the batch input means and standard deviations in Equation 11-4). 

Ioffe and Szegedy demonstrated that batch normalization considerably improved all the deep neural networks they experimented with, leading to a huge improvement in the ImageNet classification task (ImageNet is a large database of images classified into many classes, commonly used to evaluate computer vision systems). The vanishing gradients problem was strongly reduced, to the point that they could use saturating activation functions such as the tanh and even the sigmoid activation function. The networks were also much less sensitive to the weight initialization. The authors were able to use much larger learning rates, significantly speeding up the learning process. Specifically, they note that: 

Applied to a state-of-the-art image classification model, batch normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. […] Using an ensemble of batch-normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters. 

Finally, like a gift that keeps on giving, batch normalization acts like a regularizer, reducing the need for other regularization techniques (such as dropout, described later in this chapter). 

Batch normalization does, however, add some complexity to the model (although it can remove the need for normalizing the input data, as discussed earlier). Moreover, there is a runtime penalty: the neural network makes slower predictions due to the extra computations required at each layer. Fortunately, it’s often possible to fuse the BN layer with the previous layer after training, thereby avoiding the runtime penalty. This is done by updating the previous layer’s weights and biases so that it directly produces outputs of the appropriate scale and offset. For example, if the previous layer computes **XW** + **b** , then the BN layer will compute **γ** ⊗ ( **XW** + **b** – **μ** ) / **σ** + ′ **β** (ignoring the smoothing term _ε_ in the denominator). If we define **W** = **γ** ⊗ **W** / ′ ′ ′ **σ** and **b** = **γ** ⊗ ( **b** – **μ** ) / **σ** + **β** , the equation simplifies to **XW** + **b** . So, if we replace the previous layer’s weights and biases ( **W** and **b** ) with the updated weights and biases ( **W** ′ and **b** ′ ), we can get rid of the BN layer (TFLite’s converter does this automatically; see Chapter 19). 

**The Vanishing/Exploding Gradients Problems | 369** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 



You may find that training is rather slow, because each epoch takes much more time when you use batch normalization. This is usually counterbalanced by the fact that convergence is much faster with BN, so it will take fewer epochs to reach the same performance. All in all, _wall time_ will usually be shorter (this is the time measured by the clock on your wall). 

##### **Implementing batch normalization with Keras** 

As with most things with Keras, implementing batch normalization is straightforward and intuitive. Just add a `BatchNormalization` layer before or after each hidden layer’s activation function. You may also add a BN layer as the first layer in your model, but a plain `Normalization` layer generally performs just as well in this location (its only drawback is that you must first call its `adapt()` method). For example, this model applies BN after every hidden layer and as the first layer in the model (after flattening the input images): 

```
model=tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=[28, 28]),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dense(300, activation="relu",
kernel_initializer="he_normal"),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dense(100, activation="relu",
kernel_initializer="he_normal"),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dense(10, activation="softmax")
])
```

That’s all! In this tiny example with just two hidden layers batch normalization is unlikely to have a large impact, but for deeper networks it can make a tremendous difference. 

Let’s display the model summary: 

```
>>> model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten (Flatten)            (None, 784)               0
_________________________________________________________________
batch_normalization (BatchNo (None, 784)               3136
_________________________________________________________________
dense (Dense)                (None, 300)               235500
_________________________________________________________________
batch_normalization_1 (Batch (None, 300)               1200
_________________________________________________________________
dense_1 (Dense)              (None, 100)               30100
_________________________________________________________________
```

###### **370 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

```
batch_normalization_2 (Batch (None, 100)               400
```

```
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1010
=================================================================
Total params: 271,346
Trainable params: 268,978
Non-trainable params: 2,368
```

```
_________________________________________________________________
```

As you can see, each BN layer adds four parameters per input: **γ** , **β** , **μ** , and **σ** (for example, the first BN layer adds 3,136 parameters, which is 4 × 784). The last two parameters, **μ** and **σ** , are the moving averages; they are not affected by backpropaga‐ tion, so Keras calls them “non-trainable”<sup>13</sup> (if you count the total number of BN parameters, 3,136 + 1,200 + 400, and divide by 2, you get 2,368, which is the total number of non-trainable parameters in this model). 

Let’s look at the parameters of the first BN layer. Two are trainable (by backpropaga‐ tion), and two are not: 

```
>>> [(var.name, var.trainable) forvarinmodel.layers[1].variables]
```

- `[('batch_normalization/gamma:0', True),` 

- `('batch_normalization/beta:0', True),` 

- `('batch_normalization/moving_mean:0', False),` 

- `('batch_normalization/moving_variance:0', False)]` 

The authors of the BN paper argued in favor of adding the BN layers before the activation functions, rather than after (as we just did). There is some debate about this, as which is preferable seems to depend on the task—you can experiment with this too to see which option works best on your dataset. To add the BN layers before the activation function, you must remove the activation functions from the hidden layers and add them as separate layers after the BN layers. Moreover, since a batch normalization layer includes one offset parameter per input, you can remove the bias term from the previous layer by passing `use_bias=False` when creating it. Lastly, you can usually drop the first BN layer to avoid sandwiching the first hidden layer between two BN layers. The updated code looks like this: 

```
model=tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=[28, 28]),
tf.keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation("relu"),
tf.keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation("relu"),
tf.keras.layers.Dense(10, activation="softmax")
])
```

13 However, they are estimated during training based on the training data, so arguably they _are_ trainable. In Keras, “non-trainable” really means “untouched by backpropagation”. 

**The Vanishing/Exploding Gradients Problems | 371** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

The `BatchNormalization` class has quite a few hyperparameters you can tweak. The defaults will usually be fine, but you may occasionally need to tweak the `momentum` . This hyperparameter is used by the `BatchNormalization` layer when it updates the exponential moving averages; given a new value **v** (i.e., a new vector of input means or standard deviations computed over the current batch), the layer updates the running average v using the following equation: 

v v × momentum + v × 1 −momentum 

A good momentum value is typically close to 1; for example, 0.9, 0.99, or 0.999. You want more 9s for larger datasets and for smaller mini-batches. 

Another important hyperparameter is `axis` : it determines which axis should be normalized. It defaults to –1, meaning that by default it will normalize the last axis (using the means and standard deviations computed across the _other_ axes). When the input batch is 2D (i.e., the batch shape is [ _batch size, features_ ]), this means that each input feature will be normalized based on the mean and standard deviation computed across all the instances in the batch. For example, the first BN layer in the previous code example will independently normalize (and rescale and shift) each of the 784 input features. If we move the first BN layer before the `Flatten` layer, then the input batches will be 3D, with shape [ _batch size, height, width_ ]; therefore, the BN layer will compute 28 means and 28 standard deviations (1 per column of pixels, computed across all instances in the batch and across all rows in the column), and it will normalize all pixels in a given column using the same mean and standard deviation. There will also be just 28 scale parameters and 28 shift parameters. If instead you still want to treat each of the 784 pixels independently, then you should set `axis=[1, 2]` . 

Batch normalization has become one of the most-used layers in deep neural net‐ works, especially deep convolutional neural networks discussed in (Chapter 14), to the point that it is often omitted in the architecture diagrams: it is assumed that BN is added after every layer. Now let’s look at one last technique to stabilize gradients during training: gradient clipping. 

### **Gradient Clipping** 

Another technique to mitigate the exploding gradients problem is to clip the gradi‐ ents during backpropagation so that they never exceed some threshold. This is called _gradient clipping_ .<sup>14</sup> This technique is generally used in recurrent neural networks, where using batch normalization is tricky (as you will see in Chapter 15). 

> 14 Razvan Pascanu et al., “On the Difficulty of Training Recurrent Neural Networks”, _Proceedings of the 30th International Conference on Machine Learning_ (2013): 1310–1318. 

###### **372 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 





<!-- Start of picture text -->
rialens<br>; Hidden 4<br>Hidden 4 a Trainable weights<br>Hidden 3 Reuse Hidden 3<br>(a<br>Hidden 2 ‘| Hidden2 ji (O)) Fixed weights<br>—_— ===!<br>Input layer Input layer<br>Existing DNN New DNN for<br>for taskA similar task B<br><!-- End of picture text -->



You can iterate until you find the right number of layers to reuse. If you have plenty of training data, you may try replacing the top hidden layers instead of dropping them, and even adding more hidden layers. 

### **Transfer Learning with Keras** 

Let’s look at an example. Suppose the Fashion MNIST dataset only contained eight classes—for example, all the classes except for sandal and shirt. Someone built and trained a Keras model on that set and got reasonably good performance (>90% accuracy). Let’s call this model A. You now want to tackle a different task: you have images of T-shirts and pullovers, and you want to train a binary classifier: positive for T-shirts (and tops), negative for sandals. Your dataset is quite small; you only have 200 labeled images. When you train a new model for this task (let’s call it model B) with the same architecture as model A, you get 91.85% test accuracy. While drinking your morning coffee, you realize that your task is quite similar to task A, so perhaps transfer learning can help? Let’s find out! 

First, you need to load model A and create a new model based on that model’s layers. You decide to reuse all the layers except for the output layer: 

```
[...]  # Assuming model A was already trained and saved to "my_model_A"
model_A=tf.keras.models.load_model("my_model_A")
model_B_on_A=tf.keras.Sequential(model_A.layers[:-1])
model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))
```

Note that `model_A` and `model_B_on_A` now share some layers. When you train `model_B_on_A` , it will also affect `model_A` . If you want to avoid that, you need to _clone_ `model_A` before you reuse its layers. To do this, you clone model A’s architecture with `clone_model()` , then copy its weights: 

```
model_A_clone=tf.keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
```



`tf.keras.models.clone_model()` only clones the architecture, not the weights. If you don’t copy them manually using `set_weights()` , they will be initialized randomly when the cloned model is first used. 

Now you could train `model_B_on_A` for task B, but since the new output layer was initialized randomly it will make large errors (at least during the first few epochs), so there will be large error gradients that may wreck the reused weights. To avoid this, one approach is to freeze the reused layers during the first few epochs, giving the new layer some time to learn reasonable weights. To do this, set every layer’s `trainable` attribute to `False` and compile the model: 

**Reusing Pretrained Layers | 375** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

```
forlayerinmodel_B_on_A.layers[:-1]:
layer.trainable=False
optimizer=tf.keras.optimizers.SGD(learning_rate=0.001)
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
metrics=["accuracy"])
```



You must always compile your model after you freeze or unfreeze layers. 

Now you can train the model for a few epochs, then unfreeze the reused layers (which requires compiling the model again) and continue training to fine-tune the reused layers for task B. After unfreezing the reused layers, it is usually a good idea to reduce the learning rate, once again to avoid damaging the reused weights. 

```
history=model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
validation_data=(X_valid_B, y_valid_B))
forlayerinmodel_B_on_A.layers[:-1]:
layer.trainable=True
optimizer=tf.keras.optimizers.SGD(learning_rate=0.001)
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
metrics=["accuracy"])
history=model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
validation_data=(X_valid_B, y_valid_B))
```

So, what’s the final verdict? Well, this model’s test accuracy is 93.85%, up exactly two percentage points from 91.85%! This means that transfer learning reduced the error rate by almost 25%: 

```
>>> model_B_on_A.evaluate(X_test_B, y_test_B)
[0.2546142041683197, 0.9384999871253967]
```

Are you convinced? You shouldn’t be: I cheated! I tried many configurations until I found one that demonstrated a strong improvement. If you try to change the classes or the random seed, you will see that the improvement generally drops, or even vanishes or reverses. What I did is called “torturing the data until it confesses”. When a paper just looks too positive, you should be suspicious: perhaps the flashy new technique does not actually help much (in fact, it may even degrade performance), but the authors tried many variants and reported only the best results (which may be due to sheer luck), without mentioning how many failures they encountered on the way. Most of the time, this is not malicious at all, but it is part of the reason so many results in science can never be reproduced. 

**376 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

Why did I cheat? It turns out that transfer learning does not work very well with small dense networks, presumably because small networks learn few patterns, and dense networks learn very specific patterns, which are unlikely to be useful in other tasks. Transfer learning works best with deep convolutional neural networks, which tend to learn feature detectors that are much more general (especially in the lower layers). We will revisit transfer learning in Chapter 14, using the techniques we just discussed (and this time there will be no cheating, I promise!). 

### **Unsupervised Pretraining** 

Suppose you want to tackle a complex task for which you don’t have much labeled training data, but unfortunately you cannot find a model trained on a similar task. Don’t lose hope! First, you should try to gather more labeled training data, but if you can’t, you may still be able to perform _unsupervised pretraining_ (see Figure 11-6). Indeed, it is often cheap to gather unlabeled training examples, but expensive to label them. If you can gather plenty of unlabeled training data, you can try to use it to train an unsupervised model, such as an autoencoder or a generative adversarial network (GAN; see Chapter 17). Then you can reuse the lower layers of the autoencoder or the lower layers of the GAN’s discriminator, add the output layer for your task on top, and fine-tune the final network using supervised learning (i.e., with the labeled training examples). 

It is this technique that Geoffrey Hinton and his team used in 2006, and which led to the revival of neural networks and the success of deep learning. Until 2010, unsu‐ pervised pretraining—typically with restricted Boltzmann machines (RBMs; see the notebook at _https://homl.info/extra-anns_ )—was the norm for deep nets, and only after the vanishing gradients problem was alleviated did it become much more common to train DNNs purely using supervised learning. Unsupervised pretraining (today typically using autoencoders or GANs rather than RBMs) is still a good option when you have a complex task to solve, no similar model you can reuse, and little labeled training data but plenty of unlabeled training data. 

Note that in the early days of deep learning it was difficult to train deep models, so people would use a technique called _greedy layer-wise pretraining_ (depicted in Fig‐ ure 11-6). They would first train an unsupervised model with a single layer, typically an RBM, then they would freeze that layer and add another one on top of it, then train the model again (effectively just training the new layer), then freeze the new layer and add another layer on top of it, train the model again, and so on. Nowadays, things are much simpler: people generally train the full unsupervised model in one shot and use autoencoders or GANs rather than RBMs. 

**Reusing Pretrained Layers | 377** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 



<!-- Start of picture text -->
Hidden 3 Hidden 3<br>Hidden 2 Hidden2 (2) | Hidden 2<br>(a) O (a)<br>Input layer Input layer Input layer Input layer<br>Unlabeled data Labeled data<br>Unsupervised (e.g., autoencoders or GANs) Supervised<br>Train layer 1 Train layer 2 Train layer 3<br><!-- End of picture text -->

sentence “What ___ you saying?” is probably “are” or “were”). If you can train a model to reach good performance on this task, then it will already know quite a lot about language, and you can certainly reuse it for your actual task and fine-tune it on your labeled data (we will discuss more pretraining tasks in Chapter 15). 



_Self-supervised learning_ is when you automatically generate the labels from the data itself, as in the text-masking example, then you train a model on the resulting “labeled” dataset using supervised learning techniques. 

## **Faster Optimizers** 

Training a very large deep neural network can be painfully slow. So far we have seen four ways to speed up training (and reach a better solution): applying a good initialization strategy for the connection weights, using a good activation function, using batch normalization, and reusing parts of a pretrained network (possibly built for an auxiliary task or using unsupervised learning). Another huge speed boost comes from using a faster optimizer than the regular gradient descent optimizer. In this section we will present the most popular optimization algorithms: momentum, Nesterov accelerated gradient, AdaGrad, RMSProp, and finally Adam and its variants. 

### **Momentum** 

Imagine a bowling ball rolling down a gentle slope on a smooth surface: it will start out slowly, but it will quickly pick up momentum until it eventually reaches terminal velocity (if there is some friction or air resistance). This is the core idea behind _momentum optimization_ , proposed by Boris Polyak in 1964.<sup>15</sup> In contrast, regular gradient descent will take small steps when the slope is gentle and big steps when the slope is steep, but it will never pick up speed. As a result, regular gradient descent is generally much slower to reach the minimum than momentum optimization. 

Recall that gradient descent updates the weights **θ** by directly subtracting the gradient of the cost function _J_ ( **θ** ) with regard to the weights ( ∇ **θ** _J_ ( **θ** )) multiplied by the learning rate _η_ . The equation is **θ** ← **θ** – _η_ ∇ **θ** _J_ ( **θ** ). It does not care about what the earlier gradients were. If the local gradient is tiny, it goes very slowly. 

Momentum optimization cares a great deal about what previous gradients were: at each iteration, it subtracts the local gradient from the _momentum vector_ **m** (multi‐ plied by the learning rate _η_ ), and it updates the weights by adding this momentum vector (see Equation 11-5). In other words, the gradient is used as an acceleration, not 

> 15 Boris T. Polyak, “Some Methods of Speeding Up the Convergence of Iteration Methods”, _USSR Computational Mathematics and Mathematical Physics_ 4, no. 5 (1964): 1–17. 

**Faster Optimizers | 379** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

as a speed. To simulate some sort of friction mechanism and prevent the momentum from growing too large, the algorithm introduces a new hyperparameter _β_ , called the _momentum_ , which must be set between 0 (high friction) and 1 (no friction). A typical momentum value is 0.9. 

_Equation 11-5. Momentum algorithm_ 





You can verify that if the gradient remains constant, the terminal velocity (i.e., the maximum size of the weight updates) is equal to that gradient multiplied by the learning rate _η_ multiplied by 1 / (1 – _β_ ) (ignoring the sign). For example, if _β_ = 0.9, then the terminal velocity is equal to 10 times the gradient times the learning rate, so momentum optimization ends up going 10 times faster than gradient descent! This allows momentum optimization to escape from plateaus much faster than gradient descent. We saw in Chapter 4 that when the inputs have very different scales, the cost function will look like an elongated bowl (see Figure 4-7). Gradient descent goes down the steep slope quite fast, but then it takes a very long time to go down the valley. In contrast, momentum optimization will roll down the valley faster and faster until it reaches the bottom (the optimum). In deep neural networks that don’t use batch normalization, the upper layers will often end up having inputs with very different scales, so using momentum optimization helps a lot. It can also help roll past local optima. 



Due to the momentum, the optimizer may overshoot a bit, then come back, overshoot again, and oscillate like this many times before stabilizing at the minimum. This is one of the reasons it’s good to have a bit of friction in the system: it gets rid of these oscillations and thus speeds up convergence. 

Implementing momentum optimization in Keras is a no-brainer: just use the `SGD` optimizer and set its `momentum` hyperparameter, then lie back and profit! 

```
optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
```

The one drawback of momentum optimization is that it adds yet another hyperpara‐ meter to tune. However, the momentum value of 0.9 usually works well in practice and almost always goes faster than regular gradient descent. 

**380 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

< 

( 

) 

<< 



<!-- Start of picture text -->
8, COE cost<br>Starting<br>point<br>pert Regular momentum<br>X d “nV ae update<br>4 pm ms. Vy<br>x — o be<br>_— Vj 8S<br>Nesterov update<br>9;<br><!-- End of picture text -->

the weights across a valley, ∇ 1 continues to push farther across the valley, while ∇ 2 pushes back toward the bottom of the valley. This helps reduce oscillations and thus NAG converges faster. 

To use NAG, simply set `nesterov=True` when creating the `SGD` optimizer: 

```
optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9,
nesterov=True)
```

### **AdaGrad** 

Consider the elongated bowl problem again: gradient descent starts by quickly going down the steepest slope, which does not point straight toward the global optimum, then it very slowly goes down to the bottom of the valley. It would be nice if the algorithm could correct its direction earlier to point a bit more toward the global optimum. The _AdaGrad_ algorithm<sup>17</sup> achieves this correction by scaling down the gradient vector along the steepest dimensions (see Equation 11-7). 

_Equation 11-7. AdaGrad algorithm_ 





The first step accumulates the square of the gradients into the vector **s** (recall that the ⊗ symbol represents the element-wise multiplication). This vectorized form is equivalent to computing _si_ ← _si_ + (∂ _J_ ( **θ** ) / ∂ _θi_ )<sup>2</sup> for each element _si_ of the vector **s** ; in other words, each _si_ accumulates the squares of the partial derivative of the cost function with regard to parameter _θi_ . If the cost function is steep along the _i_<sup>th</sup> dimension, then _si_ will get larger and larger at each iteration. 

The second step is almost identical to gradient descent, but with one big difference: the gradient vector is scaled down by a factor of s + ε (the ⊘ symbol represents the element-wise division, and _ε_ is a smoothing term to avoid division by zero, typically set to 10<sup>–10</sup> ). This vectorized form is equivalent to simultaneously computing θi θi − η ∂ J θ / ∂ θi / si + ε for all parameters _θi_ . 

In short, this algorithm decays the learning rate, but it does so faster for steep dimen‐ sions than for dimensions with gentler slopes. This is called an _adaptive learning rate_ . It helps point the resulting updates more directly toward the global optimum (see Figure 11-8). One additional benefit is that it requires much less tuning of the learning rate hyperparameter _η_ . 

> 17 John Duchi et al., “Adaptive Subgradient Methods for Online Learning and Stochastic Optimization”, _Journal of Machine Learning Research_ 12 (2011): 2121–2159. 

###### **382 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 



<!-- Start of picture text -->
(Steep dimension) CLOUD Cost<br>8,<br>AdaGrad<br>oa SoS ett, os<br>Con>->--~-------==<br>Gradient descent<br>6, (Flatter dimension)<br><!-- End of picture text -->



<!-- Start of picture text -->
- ( ) () ()<br>< () f<br><!-- End of picture text -->

The decay rate _ρ_ is typically set to 0.9.<sup>19</sup> Yes, it is once again a new hyperparameter, but this default value often works well, so you may not need to tune it at all. 

As you might expect, Keras has an `RMSprop` optimizer: 

```
optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
```

Except on very simple problems, this optimizer almost always performs much better than AdaGrad. In fact, it was the preferred optimization algorithm of many research‐ ers until Adam optimization came around. 

### **Adam** 

_Adam_ ,<sup>20</sup> which stands for _adaptive moment estimation_ , combines the ideas of momen‐ tum optimization and RMSProp: just like momentum optimization, it keeps track of an exponentially decaying average of past gradients; and just like RMSProp, it keeps track of an exponentially decaying average of past squared gradients (see Equation 11-9). These are estimations of the mean and (uncentered) variance of the gradients. The mean is often called the while the variance is often called the _second first moment moment_ , hence the name of the algorithm. 

_Equation 11-9. Adam algorithm_ 

- 1 . m β 1 m − 1 − β 1 ∇θJ θ 

- 2 . s β 2 s + 1 − β 2 ∇θJ θ ⊗∇θJ θ 



In this equation, _t_ represents the iteration number (starting at 1). 

If you just look at steps 1, 2, and 5, you will notice Adam’s close similarity to both momentum optimization and RMSProp: _β_ 1 corresponds to _β_ in momentum optimization, and _β_ 2 corresponds to _ρ_ in RMSProp. The only difference is that step 1 computes an exponentially decaying average rather than an exponentially decaying sum, but these are actually equivalent except for a constant factor (the decaying average is just 1 – _β_ 1 times the decaying sum). Steps 3 and 4 are somewhat of a 

> 19 _ρ_ is the Greek letter rho. 

> 20 Diederik P. Kingma and Jimmy Ba, “Adam: A Method for Stochastic Optimization”, arXiv preprint arXiv:1412.6980 (2014). 

**384 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

technical detail: since **m** and **s** are initialized at 0, they will be biased toward 0 at the beginning of training, so these two steps will help boost **m** and **s** at the beginning of training. 

The momentum decay hyperparameter _β_ 1 is typically initialized to 0.9, while the scaling decay hyperparameter _β_ 2 is often initialized to 0.999. As earlier, the smoothing term _ε_ is usually initialized to a tiny number such as 10<sup>–7</sup> . These are the default values for the `Adam` class. Here is how to create an Adam optimizer using Keras: 

```
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
beta_2=0.999)
```

Since Adam is an adaptive learning rate algorithm, like AdaGrad and RMSProp, it requires less tuning of the learning rate hyperparameter _η_ . You can often use the default value _η_ = 0.001, making Adam even easier to use than gradient descent. 



If you are starting to feel overwhelmed by all these different tech‐ niques and are wondering how to choose the right ones for your task, don’t worry: some practical guidelines are provided at the end of this chapter. 

Finally, three variants of Adam are worth mentioning: AdaMax, Nadam, and AdamW. 

### **AdaMax** 

The Adam paper also introduced AdaMax. Notice that in step 2 of Equation 11-9, Adam accumulates the squares of the gradients in **s** (with a greater weight for more recent gradients). In step 5, if we ignore _ε_ and steps 3 and 4 (which are technical details anyway), Adam scales down the parameter updates by the square root of **s** . In short, Adam scales down the parameter updates by the ℓ2 norm of the time-decayed gradients (recall that the ℓ2 norm is the square root of the sum of squares). 

AdaMax replaces the ℓ2 norm with the ℓ∞ norm (a fancy way of saying the max). Specifically, it replaces step 2 in Equation 11-9 with s max β 2 s , abs( ∇θJ θ , it drops step 4, and in step 5 it scales down the gradient updates by a factor of **s** , which is the max of the absolute value of the time-decayed gradients. 

In practice, this can make AdaMax more stable than Adam, but it really depends on the dataset, and in general Adam performs better. So, this is just one more optimizer you can try if you experience problems with Adam on some task. 

**Faster Optimizers | 385** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

### **Nadam** 

Nadam optimization is Adam optimization plus the Nesterov trick, so it will often converge slightly faster than Adam. In his report introducing this technique,<sup>21</sup> the researcher Timothy Dozat compares many different optimizers on various tasks and finds that Nadam generally outperforms Adam but is sometimes outperformed by RMSProp. 

### **AdamW** 

AdamW<sup>22</sup> is a variant of Adam that integrates a regularization technique called _weight decay_ . Weight decay reduces the size of the model’s weights at each training iteration by multiplying them by a decay factor such as 0.99. This may remind you of ℓ2 regularization (introduced in Chapter 4), which also aims to keep the weights small, and indeed it can be shown mathematically that ℓ2 regularization is equivalent to weight decay when using SGD. However, when using Adam or its variants, ℓ2 regularization and weight decay are _not_ equivalent: in practice, combining Adam with ℓ2 regularization results in models that often don’t generalize as well as those produced by SGD. AdamW fixes this issue by properly combining Adam with weight decay. 



Adaptive optimization methods (including RMSProp, Adam, Ada‐ Max, Nadam, and AdamW optimization) are often great, converg‐ ing fast to a good solution. However, a 2017 paper<sup>23</sup> by Ashia C. Wilson et al. showed that they can lead to solutions that generalize poorly on some datasets. So when you are disappointed by your model’s performance, try using NAG instead: your dataset may just be allergic to adaptive gradients. Also check out the latest research, because it’s moving fast. 

To use Nadam, AdaMax, or AdamW in Keras, replace `tf.keras.optimiz ers.Adam` with `tf.keras.optimizers.Nadam` , `tf.keras.optimizers.Adamax` , or `tf.keras.optimizers.AdamW` . For AdamW, you probably want to tune the `weight_decay` hyperparameter. 

All the optimization techniques discussed so far only rely on the _first-order partial derivatives_ ( _Jacobians_ ). The optimization literature also contains amazing algorithms 

- 21 Timothy Dozat, “Incorporating Nesterov Momentum into Adam” (2016). 

- 22 Ilya Loshchilov, and Frank Hutter, “Decoupled Weight Decay Regularization”, arXiv preprint arXiv:1711.05101 (2017). 

- 23 Ashia C. Wilson et al., “The Marginal Value of Adaptive Gradient Methods in Machine Learning”, _Advances in Neural Information Processing Systems_ 30 (2017): 4148–4158. 

###### **386 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

based on the _second-order partial derivatives_ (the _Hessians_ , which are the partial derivatives of the Jacobians). Unfortunately, these algorithms are very hard to apply to deep neural networks because there are _n_<sup>2</sup> Hessians per output (where _n_ is the number of parameters), as opposed to just _n_ Jacobians per output. Since DNNs typi‐ cally have tens of thousands of parameters or more, the second-order optimization algorithms often don’t even fit in memory, and even when they do, computing the Hessians is just too slow. 

#### **Training Sparse Models** 

All the optimization algorithms we just discussed produce dense models, meaning that most parameters will be nonzero. If you need a blazingly fast model at runtime, or if you need it to take up less memory, you may prefer to end up with a sparse model instead. 

One way to achieve this is to train the model as usual, then get rid of the tiny weights (set them to zero). However, this will typically not lead to a very sparse model, and it may degrade the model’s performance. 

A better option is to apply strong ℓ1 regularization during training (you’ll see how later in this chapter), as it pushes the optimizer to zero out as many weights as it can (as discussed in “Lasso Regression” on page 158). 

If these techniques remain insufficient, check out the TensorFlow Model Optimiza‐ tion Toolkit (TF-MOT), which provides a pruning API capable of iteratively remov‐ ing connections during training based on their magnitude. 

Table 11-2 compares all the optimizers we’ve discussed so far (* is bad, ** is average, and *** is good). 

_Table 11-2. Optimizer comparison_ 

|**Class**|**Convergence speed **|**Convergence quality**|
|---|---|---|
|`SGD`|*|***|
|`SGD(momentum=...)`|**|***|
|`SGD(momentum=..., nesterov=True)`|**|***|
|`Adagrad`|***|* (stops too early)|
|`RMSprop`|***|** or ***|
|`Adam`|***|** or ***|
|`AdaMax`|***|** or ***|
|`Nadam`|***|** or ***|
|`AdamW`|***|** or ***|



**Faster Optimizers | 387** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 



<!-- Start of picture text -->
Loss<br>n Way too high: diverges<br>Ss 7 too small: slow<br>“7S 7 too high: suboptimal<br>0 7 just right Epoch<br>Start with a high learning rate then reduce it: perfect!<br><!-- End of picture text -->

###### _Exponential scheduling_ 

Set the learning rate to _η_ ( _t_ ) = _η_ 0 0.1<sup>_t/s_</sup> . The learning rate will gradually drop by a factor of 10 every _s_ steps. While power scheduling reduces the learning rate more and more slowly, exponential scheduling keeps slashing it by a factor of 10 every _s_ steps. 

###### _Piecewise constant scheduling_ 

Use a constant learning rate for a number of epochs (e.g., _η_ 0 = 0.1 for 5 epochs), then a smaller learning rate for another number of epochs (e.g., _η_ 1 = 0.001 for 50 epochs), and so on. Although this solution can work very well, it requires fiddling around to figure out the right sequence of learning rates and how long to use each of them. 

###### _Performance scheduling_ 

Measure the validation error every _N_ steps (just like for early stopping), and reduce the learning rate by a factor of _λ_ when the error stops dropping. 

###### _Power scheduling_ 

Set the learning rate to a function of the iteration number _t_ : _η_ ( _t_ ) = _η_ 0 / (1 + _t_ / _s_ )<sup>_c_</sup> . The initial learning rate _η_ 0, the power _c_ (typically set to 1), and the steps _s_ are hyperparameters. The learning rate drops at each step. After _s_ steps, the learning rate is down to _η_ 0 / 2. After _s_ more steps it is down to _η_ 0 / 3, then it goes down to _η_ 0 / 4, then _η_ 0 / 5, and so on. As you can see, this schedule first drops quickly, then more and more slowly. Of course, power scheduling requires tuning _η_ 0 and _s_ (and possibly _c_ ). 

###### _1cycle scheduling_ 

1cycle was introduced in a 2018 paper by Leslie Smith.<sup>24</sup> Contrary to the other approaches, it starts by increasing the initial learning rate _η_ 0, growing linearly up to _η_ 1 halfway through training. Then it decreases the learning rate linearly down to _η_ 0 again during the second half of training, finishing the last few epochs by dropping the rate down by several orders of magnitude (still linearly). The maximum learning rate _η_ 1 is chosen using the same approach we used to find the optimal learning rate, and the initial learning rate _η_ 0 is usually 10 times lower. When using a momentum, we start with a high momentum first (e.g., 0.95), then drop it down to a lower momentum during the first half of training (e.g., down to 0.85, linearly), and then bring it back up to the maximum value (e.g., 0.95) dur‐ ing the second half of training, finishing the last few epochs with that maximum value. Smith did many experiments showing that this approach was often able to speed up training considerably and reach better performance. For example, 

> 24 Leslie N. Smith, “A Disciplined Approach to Neural Network Hyper-Parameters: Part 1—Learning Rate, Batch Size, Momentum, and Weight Decay”, arXiv preprint arXiv:1803.09820 (2018). 

**Learning Rate Scheduling | 389** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

on the popular CIFAR10 image dataset, this approach reached 91.9% validation accuracy in just 100 epochs, compared to 90.3% accuracy in 800 epochs through a standard approach (with the same neural network architecture). This feat was dubbed _super-convergence_ . 

A 2013 paper by Andrew Senior et al.<sup>25</sup> compared the performance of some of the most popular learning schedules when using momentum optimization to train deep neural networks for speech recognition. The authors concluded that, in this setting, both performance scheduling and exponential scheduling performed well. They favored exponential scheduling because it was easy to tune and it converged slightly faster to the optimal solution. They also mentioned that it was easier to implement than performance scheduling, but in Keras both options are easy. That said, the 1cycle approach seems to perform even better. 

Implementing exponential scheduling and piecewise scheduling can be done using a `LearningRateScheduler` callback. You first need to define a function that takes the current epoch and returns the learning rate: 

```
defexponential_decay_fn(epoch):
return0.01*0.1** (epoch/20)
```

If you do not want to hardcode _η_ 0 and _s_ , you can create a function that returns a configured function: 

```
defexponential_decay(lr0, s):
defexponential_decay_fn(epoch):
returnlr0*0.1** (epoch/s)
returnexponential_decay_fn
```

```
exponential_decay_fn=exponential_decay(lr0=0.01, s=20)
```

Next, create a `LearningRateScheduler` callback, giving it the schedule function, and pass this callback to the `fit()` method: 

```
lr_scheduler=tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history=model.fit(X_train, y_train, [...], callbacks=[lr_scheduler])
```

The `LearningRateScheduler` will update the optimizer’s `learning_rate` attribute at the beginning of each epoch. Updating the learning rate once per epoch is usually enough, but if you want it to be updated more often, for example at every step, you can always write your own callback (see the “Exponential Scheduling” section of this chapter’s notebook for an example). Updating the learning rate at every step may help if there are many steps per epoch. Alternatively, you can use the `tf.keras. optimizers.schedules` approach, described shortly. 

> 25 Andrew Senior et al., “An Empirical Study of Learning Rates in Deep Neural Networks for Speech Recogni‐ tion”, _Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing_ (2013): 6724–6728. 

**390 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 



After training, `history.history["lr"]` gives you access to the list of learning rates used during training. 

The schedule function can optionally take the current learning rate as a second argu‐ ment. For example, the following schedule function multiplies the previous learning rate by 0.1<sup>1/20</sup> , which results in the same exponential decay (except the decay now starts at the beginning of epoch 0 instead of 1): 

```
defexponential_decay_fn(epoch, lr):
returnlr*0.1** (1/20)
```

This implementation relies on the optimizer’s initial learning rate (contrary to the previous implementation), so make sure to set it appropriately. 

When you save a model, the optimizer and its learning rate get saved along with it. This means that with this new schedule function, you could just load a trained model and continue training where it left off, no problem. Things are not so simple if your schedule function uses the `epoch` argument, however: the epoch does not get saved, and it gets reset to 0 every time you call the `fit()` method. If you were to continue training a model where it left off, this could lead to a very large learning rate, which would likely damage your model’s weights. One solution is to manually set the `fit()` method’s `initial_epoch` argument so the `epoch` starts at the right value. 

For piecewise constant scheduling, you can use a schedule function like the following one (as earlier, you can define a more general function if you want; see the “Piece‐ wise Constant Scheduling” section of the notebook for an example), then create a `LearningRateScheduler` callback with this function and pass it to the `fit()` method, just like for exponential scheduling: 

```
defpiecewise_constant_fn(epoch):
ifepoch<5:
return0.01
elifepoch<15:
return0.005
else:
return0.001
```

For performance scheduling, use the `ReduceLROnPlateau` callback. For example, if you pass the following callback to the `fit()` method, it will multiply the learning rate by 0.5 whenever the best validation loss does not improve for five consecutive epochs (other options are available; please check the documentation for more details): 

```
lr_scheduler=tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
history=model.fit(X_train, y_train, [...], callbacks=[lr_scheduler])
```

**Learning Rate Scheduling | 391** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

Lastly, Keras offers an alternative way to implement learning rate scheduling: you can define a scheduled learning rate using one of the classes available in `tf.keras. optimizers.schedules` , then pass it to any optimizer. This approach updates the learning rate at each step rather than at each epoch. For example, here is how to implement the same exponential schedule as the `exponential_decay_fn()` function we defined earlier: 

```
batch_size=32
n_epochs=20
n_steps=n_epochs*np.ceil(len(X_train) /batch_size)
scheduled_learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate=0.01, decay_steps=n_steps, decay_rate=0.1)
optimizer=tf.keras.optimizers.SGD(learning_rate=scheduled_learning_rate)
```

This is nice and simple, plus when you save the model, the learning rate and its schedule (including its state) get saved as well. 

You can also use the `InverseTimeDecay` learning rate schedule to implement power scheduling, for example: 

```
lr_schedule=tf.keras.optimizers.schedules.InverseTimeDecay(
initial_learning_rate=0.01,
decay_steps=10_000,
decay_rate=1.0,
staircase=False
)
optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule)
```

Note that Keras assumes that _c_ is equal to 1, and _s_ is equal to `decay_steps / decay_rate` . 

As for 1cycle, Keras does not support it, but it’s possible to implement it in less than 30 lines of code by creating a custom callback that modifies the learning rate at each iteration. To update the optimizer’s learning rate from within the callback’s `on_batch_begin()` method, you need to call `tf.keras. backend.set_value(self.model.optimizer.learning_rate` , `new_learning_rate)` . See the “1Cycle Scheduling” section of the notebook for an example. 

To sum up, exponential decay, performance scheduling, and 1cycle can considerably speed up convergence, so give them a try! 

## **Avoiding Overfitting Through Regularization** 

With four parameters I can fit an elephant and with five I can make him wiggle his trunk. 

- —John von Neumann, cited by Enrico Fermi in _Nature_ 427 

###### **392 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

With thousands of parameters, you can fit the whole zoo. Deep neural networks typi‐ cally have tens of thousands of parameters, sometimes even millions. This gives them an incredible amount of freedom and means they can fit a huge variety of complex datasets. But this great flexibility also makes the network prone to overfitting the training set. Regularization is often needed to prevent this. 

We already implemented one of the best regularization techniques in Chapter 10: early stopping. Moreover, even though batch normalization was designed to solve the unstable gradients problems, it also acts like a pretty good regularizer. In this section we will examine other popular regularization techniques for neural networks: ℓ1 and ℓ2 regularization, dropout, and max-norm regularization. 

### **ℓ1 and ℓ2 Regularization** 

Just like you did in Chapter 4 for simple linear models, you can use ℓ2 regularization to constrain a neural network’s connection weights, and/or ℓ1 regularization if you want a sparse model (with many weights equal to 0). Here is how to apply ℓ2 regulari‐ zation to a Keras layer’s connection weights, using a regularization factor of 0.01: 

```
layer=tf.keras.layers.Dense(100, activation="relu",
kernel_initializer="he_normal",
kernel_regularizer=tf.keras.regularizers.l2(0.01))
```

The `l2()` function returns a regularizer that will be called at each step during training to compute the regularization loss. This is then added to the final loss. As you might expect, you can just use `tf.keras.regularizers.l1()` if you want ℓ1 regularization; if you want both ℓ1 and ℓ2 regularization, use `tf.keras.regularizers.l1_l2()` (specifying both regularization factors). 

Since you will typically want to apply the same regularizer to all layers in your network, as well as using the same activation function and the same initialization strategy in all hidden layers, you may find yourself repeating the same arguments. This makes the code ugly and error-prone. To avoid this, you can try refactoring your code to use loops. Another option is to use Python’s `functools.partial()` function, which lets you create a thin wrapper for any callable, with some default argument values: 

```
fromfunctoolsimportpartial
```

```
RegularizedDense=partial(tf.keras.layers.Dense,
activation="relu",
kernel_initializer="he_normal",
kernel_regularizer=tf.keras.regularizers.l2(0.01))
```

```
model=tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=[28, 28]),
RegularizedDense(100),
RegularizedDense(100),
```

**Avoiding Overfitting Through Regularization | 393** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

```
RegularizedDense(10, activation="softmax")
```

```
])
```



As we saw earlier, ℓ2 regularization is fine when using SGD, momentum optimization, and Nesterov momentum optimization, but not with Adam and its variants. If you want to use Adam with weight decay, then do not use ℓ2 regularization: use AdamW instead. 

### **Dropout** 

_Dropout_ is one of the most popular regularization techniques for deep neural net‐ works. It was proposed in a paper<sup>26</sup> by Geoffrey Hinton et al. in 2012 and further detailed in a 2014 paper<sup>27</sup> by Nitish Srivastava et al., and it has proven to be highly successful: many state-of-the-art neural networks use dropout, as it gives them a 1%–2% accuracy boost. This may not sound like a lot, but when a model already has 95% accuracy, getting a 2% accuracy boost means dropping the error rate by almost 40% (going from 5% error to roughly 3%). 

It is a fairly simple algorithm: at every training step, every neuron (including the input neurons, but always excluding the output neurons) has a probability _p_ of being temporarily “dropped out”, meaning it will be entirely ignored during this training step, but it may be active during the next step (see Figure 11-10). The hyperparameter _p_ is called the _dropout rate_ , and it is typically set between 10% and 50%: closer to 20%–30% in recurrent neural nets (see Chapter 15), and closer to 40%–50% in convo‐ lutional neural networks (see Chapter 14). After training, neurons don’t get dropped anymore. And that’s all (except for a technical detail we will discuss momentarily). 

It’s surprising at first that this destructive technique works at all. Would a company perform better if its employees were told to toss a coin every morning to decide whether or not to go to work? Well, who knows; perhaps it would! The company would be forced to adapt its organization; it could not rely on any single person to work the coffee machine or perform any other critical tasks, so this expertise would have to be spread across several people. Employees would have to learn to cooperate with many of their coworkers, not just a handful of them. The company would become much more resilient. If one person quit, it wouldn’t make much of a difference. It’s unclear whether this idea would actually work for companies, but it certainly does for neural networks. Neurons trained with dropout cannot co-adapt with their neighboring neurons; they have to be as useful as possible on their own. 

> 26 Geoffrey E. Hinton et al., “Improving Neural Networks by Preventing Co-Adaptation of Feature Detectors”, arXiv preprint arXiv:1207.0580 (2012). 

> 27 Nitish Srivastava et al., “Dropout: A Simple Way to Prevent Neural Networks from Overfitting”, _Journal of Machine Learning Research_ 15 (2014): 1929–1958. 

###### **394 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 



<!-- Start of picture text -->
©@<br>\<br>'<br>t ‘\ \<br>\ 1?? .‘a41<br>] ?<br>‘\ 4<br>\ W “7 /<br>‘\ ‘14? \e%‘ /;<br>K S| 7<br>Xj Xp<br><!-- End of picture text -->



In practice, you can usually apply dropout only to the neurons in the top one to three layers (excluding the output layer). 

There is one small but important technical detail. Suppose _p_ = 75%: on average only 25% of all neurons are active at each step during training. This means that after training, a neuron would be connected to four times as many input neurons as it would be during training. To compensate for this fact, we need to multiply each neuron’s input connection weights by four during training. If we don’t, the neural network will not perform well as it will see different data during and after training. More generally, we need to divide the connection weights by the _keep probability_ (1 – _p_ ) during training. 

To implement dropout using Keras, you can use the `tf.keras.layers.Dropout` layer. During training, it randomly drops some inputs (setting them to 0) and divides the remaining inputs by the keep probability. After training, it does nothing at all; it just passes the inputs to the next layer. The following code applies dropout regularization before every dense layer, using a dropout rate of 0.2: 

```
model=tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=[28, 28]),
tf.keras.layers.Dropout(rate=0.2),
tf.keras.layers.Dense(100, activation="relu",
kernel_initializer="he_normal"),
tf.keras.layers.Dropout(rate=0.2),
tf.keras.layers.Dense(100, activation="relu",
kernel_initializer="he_normal"),
tf.keras.layers.Dropout(rate=0.2),
tf.keras.layers.Dense(10, activation="softmax")
])
[...]  # compile and train the model
```



Since dropout is only active during training, comparing the train‐ ing loss and the validation loss can be misleading. In particular, a model may be overfitting the training set and yet have similar training and validation losses. So, make sure to evaluate the train‐ ing loss without dropout (e.g., after training). 

If you observe that the model is overfitting, you can increase the dropout rate. Conversely, you should try decreasing the dropout rate if the model underfits the training set. It can also help to increase the dropout rate for large layers, and reduce it for small ones. Moreover, many state-of-the-art architectures only use dropout after the last hidden layer, so you may want to try this if full dropout is too strong. 

**396 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

Dropout does tend to significantly slow down convergence, but it often results in a better model when tuned properly. So, it is generally well worth the extra time and effort, especially for large models. 



If you want to regularize a self-normalizing network based on the SELU activation function (as discussed earlier), you should use _alpha dropout_ : this is a variant of dropout that preserves the mean and standard deviation of its inputs. It was introduced in the same paper as SELU, as regular dropout would break self-normalization. 

### **Monte Carlo (MC) Dropout** 

In 2016, a paper<sup>28</sup> by Yarin Gal and Zoubin Ghahramani added a few more good reasons to use dropout: 

- First, the paper established a profound connection between dropout networks (i.e., neural networks containing `Dropout` layers) and approximate Bayesian inference,<sup>29</sup> giving dropout a solid mathematical justification. 

- Second, the authors introduced a powerful technique called _MC dropout_ , which can boost the performance of any trained dropout model without having to retrain it or even modify it at all. It also provides a much better measure of the model’s uncertainty, and it can be implemented in just a few lines of code. 

If this all sounds like some “one weird trick” clickbait, then take a look at the following code. It is the full implementation of MC dropout, boosting the dropout model we trained earlier without retraining it: 

```
importnumpyasnp
```

```
y_probas=np.stack([model(X_test, training=True)
forsampleinrange(100)])
y_proba=y_probas.mean(axis=0)
```

Note that `model(X)` is similar to `model.predict(X)` except it returns a tensor rather than a NumPy array, and it supports the `training` argument. In this code example, setting `training=True` ensures that the `Dropout` layer remains active, so all predic‐ tions will be a bit different. We just make 100 predictions over the test set, and we compute their average. More specifically, each call to the model returns a matrix with one row per instance and one column per class. Because there are 10,000 instances 

> 28 Yarin Gal and Zoubin Ghahramani, “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning”, _Proceedings of the 33rd International Conference on Machine Learning_ (2016): 1050–1059. 

> 29 Specifically, they show that training a dropout network is mathematically equivalent to approximate Bayesian inference in a specific type of probabilistic model called a _deep Gaussian process_ . 

**Avoiding Overfitting Through Regularization | 397** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

in the test set and 10 classes, this is a matrix of shape [10000, 10]. We stack 100 such matrices, so `y_probas` is a 3D array of shape [100, 10000, 10]. Once we average over the first dimension ( `axis=0` ) we get `y_proba` , an array of shape [10000, 10], like we would get with a single prediction. That’s all! Averaging over multiple predictions with dropout turned on gives us a Monte Carlo estimate that is generally more reliable than the result of a single prediction with dropout turned off. For example, let’s look at the model’s prediction for the first instance in the Fashion MNIST test set, with dropout turned off: 

```
>>> model.predict(X_test[:1]).round(3)
array([[0.   , 0.   , 0.   , 0.   , 0.   , 0.024, 0.   , 0.132, 0.   ,
        0.844]], dtype=float32)
```

The model is fairly confident (84.4%) that this image belongs to class 9 (ankle boot). Compare this with the MC dropout prediction: 

```
>>> y_proba[0].round(3)
array([0.   , 0.   , 0.   , 0.   , 0.   , 0.067, 0.   , 0.209, 0.001,
       0.723], dtype=float32)
```

The model still seems to prefer class 9, but its confidence dropped down to 72.3%, and the estimated probabilities for classes 5 (sandal) and 7 (sneaker) have increased, which makes sense given they’re also footwear. 

MC dropout tends to improve the reliability of the model’s probability estimates. This means that it’s less likely to be confident but wrong, which can be dangerous: just imagine a self-driving car confidently ignoring a stop sign. It’s also useful to know exactly which other classes are most likely. Additionally, you can take a look at the standard deviation of the probability estimates: 

```
>>> y_std=y_probas.std(axis=0)
>>> y_std[0].round(3)
array([0.   , 0.   , 0.   , 0.001, 0.   , 0.096, 0.   , 0.162, 0.001,
       0.183], dtype=float32)
```

Apparently there’s quite a lot of variance in the probability estimates for class 9: the standard deviation is 0.183, which should be compared to the estimated probability of 0.723: if you were building a risk-sensitive system (e.g., a medical or financial sys‐ tem), you would probably treat such an uncertain prediction with extreme caution. You would definitely not treat it like an 84.4% confident prediction. The model’s accuracy also got a (very) small boost from 87.0% to 87.2%: 

```
>>> y_pred=y_proba.argmax(axis=1)
>>> accuracy= (y_pred==y_test).sum() /len(y_test)
>>> accuracy
0.8717
```

###### **398 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 



The number of Monte Carlo samples you use (100 in this example) is a hyperparameter you can tweak. The higher it is, the more accurate the predictions and their uncertainty estimates will be. However, if you double it, inference time will also be doubled. Moreover, above a certain number of samples, you will notice little improvement. Your job is to find the right trade-off between latency and accuracy, depending on your application. 

If your model contains other layers that behave in a special way during training (such as `BatchNormalization` layers), then you should not force training mode like we just did. Instead, you should replace the `Dropout` layers with the following `MCDropout` class:<sup>30</sup> 

```
classMCDropout(tf.keras.layers.Dropout):
defcall(self, inputs, training=False):
returnsuper().call(inputs, training=True)
```

Here, we just subclass the `Dropout` layer and override the `call()` method to force its `training` argument to `True` (see Chapter 12). Similarly, you could define an `MCAlpha Dropout` class by subclassing `AlphaDropout` instead. If you are creating a model from scratch, it’s just a matter of using `MCDropout` rather than `Dropout` . But if you have a model that was already trained using `Dropout` , you need to create a new model that’s identical to the existing model except with `Dropout` instead of `MCDropout` , then copy the existing model’s weights to your new model. 

In short, MC dropout is a great technique that boosts dropout models and provides better uncertainty estimates. And of course, since it is just regular dropout during training, it also acts like a regularizer. 

### **Max-Norm Regularization** 

Another popular regularization technique for neural networks is called _max-norm regularization_ : for each neuron, it constrains the weights **w** of the incoming connec‐ tions such that ∥ **w** ∥ 2 ≤ _r_ , where _r_ is the max-norm hyperparameter and ∥ · ∥ 2 is the ℓ2 norm. 

Max-norm regularization does not add a regularization loss term to the overall loss function. Instead, it is typically implemented by computing ∥ **w** ∥ 2 after each training step and rescaling **w** if needed ( **w** ← **w** _r_ / ∥ **w** ∥ 2). 

> 30 This `MCDropout` class will work with all Keras APIs, including the sequential API. If you only care about the functional API or the subclassing API, you do not have to create an `MCDropout` class; you can create a regular `Dropout` layer and call it with `training=True` . 

**Avoiding Overfitting Through Regularization | 399** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

Reducing _r_ increases the amount of regularization and helps reduce overfitting. Max-norm regularization can also help alleviate the unstable gradients problems (if you are not using batch normalization). 

To implement max-norm regularization in Keras, set the `kernel_constraint` argu‐ ment of each hidden layer to a `max_norm()` constraint with the appropriate max value, like this: 

```
dense=tf.keras.layers.Dense(
100, activation="relu", kernel_initializer="he_normal",
kernel_constraint=tf.keras.constraints.max_norm(1.))
```

After each training iteration, the model’s `fit()` method will call the object returned by `max_norm()` , passing it the layer’s weights and getting rescaled weights in return, which then replace the layer’s weights. As you’ll see in Chapter 12, you can define your own custom constraint function if necessary and use it as the `kernel_ constraint` . You can also constrain the bias terms by setting the `bias_constraint` argument. 

The `max_norm()` function has an `axis` argument that defaults to `0` . A `Dense` layer usually has weights of shape [ _number of inputs_ , _number of neurons_ ], so using `axis=0` means that the max-norm constraint will apply independently to each neuron’s weight vector. If you want to use max-norm with convolutional layers (see Chap‐ ter 14), make sure to set the `max_norm()` constraint’s `axis` argument appropriately (usually `axis=[0, 1, 2]` ). 

## **Summary and Practical Guidelines** 

In this chapter we have covered a wide range of techniques, and you may be won‐ dering which ones you should use. This depends on the task, and there is no clear consensus yet, but I have found the configuration in Table 11-3 to work fine in most cases, without requiring much hyperparameter tuning. That said, please do not consider these defaults as hard rules! 

_Table 11-3. Default DNN configuration_ 

|**Hyperparameter**|**Default value**|
|---|---|
|Kernel initializer|He initialization|
|Activation function|ReLU if shallow; Swish if deep|
|Normalization|None if shallow; batch norm if deep|
|Regularization|Early stopping; weight decay if needed|
|Optimizer|Nesterov accelerated gradients or AdamW|
|Learning rate schedule|Performance scheduling or 1cycle|



###### **400 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

If the network is a simple stack of dense layers, then it can self-normalize, and you should use the configuration in Table 11-4 instead. 

_Table 11-4. DNN configuration for a self-normalizing net_ 

|**Hyperparameter**|**Default value**|
|---|---|
|Kernel initializer|LeCun initialization|
|Activation function|SELU|
|Normalization|None (self-normalization)|
|Regularization|Alpha dropout if needed|
|Optimizer|Nesterov accelerated gradients|
|Learning rate schedule|Performance scheduling or 1cycle|



Don’t forget to normalize the input features! You should also try to reuse parts of a pretrained neural network if you can find one that solves a similar problem, or use unsupervised pretraining if you have a lot of unlabeled data, or use pretraining on an auxiliary task if you have a lot of labeled data for a similar task. 

While the previous guidelines should cover most cases, here are some exceptions: 

- If you need a sparse model, you can use ℓ1 regularization (and optionally zero out the tiny weights after training). If you need an even sparser model, you can use the TensorFlow Model Optimization Toolkit. This will break self-normalization, so you should use the default configuration in this case. 

- If you need a low-latency model (one that performs lightning-fast predictions), you may need to use fewer layers, use a fast activation function such as ReLU or leaky ReLU, and fold the batch normalization layers into the previous layers after training. Having a sparse model will also help. Finally, you may want to reduce the float precision from 32 bits to 16 or even 8 bits (see “Deploying a Model to a Mobile or Embedded Device” on page 741). Again, check out TF-MOT. 

- If you are building a risk-sensitive application, or inference latency is not very important in your application, you can use MC dropout to boost performance and get more reliable probability estimates, along with uncertainty estimates. 

With these guidelines, you are now ready to train very deep nets! I hope you are now convinced that you can go quite a long way using just the convenient Keras API. There may come a time, however, when you need to have even more control; for example, to write a custom loss function or to tweak the training algorithm. For such cases you will need to use TensorFlow’s lower-level API, as you will see in the next chapter. 

**Summary and Practical Guidelines | 401** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

## **Exercises** 

**1.** What is the problem that Glorot initialization and He initialization aim to fix? 

**2.** Is it OK to initialize all the weights to the same value as long as that value is selected randomly using He initialization? 

**3.** Is it OK to initialize the bias terms to 0? 

**4.** In which cases would you want to use each of the activation functions we discussed in this chapter? 

**5.** What may happen if you set the `momentum` hyperparameter too close to 1 (e.g., 0.99999) when using an `SGD` optimizer? 

**6.** Name three ways you can produce a sparse model. 

**7.** Does dropout slow down training? Does it slow down inference (i.e., making predictions on new instances)? What about MC dropout? 

**8.** Practice training a deep neural network on the CIFAR10 image dataset: 

   - **a.** Build a DNN with 20 hidden layers of 100 neurons each (that’s too many, but it’s the point of this exercise). Use He initialization and the Swish activation function. 

   - **b.** Using Nadam optimization and early stopping, train the network on the CIFAR10 dataset. You can load it with `tf.keras.datasets.cifar10.load_ data()` . The dataset is composed of 60,000 32 × 32–pixel color images (50,000 for training, 10,000 for testing) with 10 classes, so you’ll need a softmax output layer with 10 neurons. Remember to search for the right learning rate each time you change the model’s architecture or hyperparameters. 

   - **c.** Now try adding batch normalization and compare the learning curves: is it converging faster than before? Does it produce a better model? How does it affect training speed? 

   - **d.** Try replacing batch normalization with SELU, and make the necessary adjust‐ ments to ensure the network self-normalizes (i.e., standardize the input fea‐ tures, use LeCun normal initialization, make sure the DNN contains only a sequence of dense layers, etc.). 

   - **e.** Try regularizing the model with alpha dropout. Then, without retraining your model, see if you can achieve better accuracy using MC dropout. 

   - **f.** Retrain your model using 1cycle scheduling and see if it improves training speed and model accuracy. 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

###### **402 | Chapter 11: Training Deep Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:17:08. 

