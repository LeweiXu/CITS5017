### **CHAPTER 14** 

# **Deep Computer Vision Using Convolutional Neural Networks** 

Although IBM’s Deep Blue supercomputer beat the chess world champion Garry Kasparov back in 1996, it wasn’t until fairly recently that computers were able to reliably perform seemingly trivial tasks such as detecting a puppy in a picture or recognizing spoken words. Why are these tasks so effortless to us humans? The answer lies in the fact that perception largely takes place outside the realm of our consciousness, within specialized visual, auditory, and other sensory modules in our brains. By the time sensory information reaches our consciousness, it is already adorned with high-level features; for example, when you look at a picture of a cute puppy, you cannot choose _not_ to see the puppy, _not_ to notice its cuteness. Nor can you explain _how_ you recognize a cute puppy; it’s just obvious to you. Thus, we cannot trust our subjective experience: perception is not trivial at all, and to understand it we must look at how our sensory modules work. 

_Convolutional neural networks_ (CNNs) emerged from the study of the brain’s visual cortex, and they have been used in computer image recognition since the 1980s. Over the last 10 years, thanks to the increase in computational power, the amount of available training data, and the tricks presented in Chapter 11 for training deep nets, CNNs have managed to achieve superhuman performance on some complex visual tasks. They power image search services, self-driving cars, automatic video classifica‐ tion systems, and more. Moreover, CNNs are not restricted to visual perception: they are also successful at many other tasks, such as voice recognition and natural language processing. However, we will focus on visual applications for now. 

**479** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 



<!-- Start of picture text -->
7  Ch 33299<br>OX TRE aa<br>7 ~ \ | = = ?<br><!-- End of picture text -->

Moreover, the authors showed that some neurons react only to images of horizontal lines, while others react only to lines with different orientations (two neurons may have the same receptive field but react to different line orientations). They also noticed that some neurons have larger receptive fields, and they react to more com‐ plex patterns that are combinations of the lower-level patterns. These observations led to the idea that the higher-level neurons are based on the outputs of neighboring lower-level neurons (in Figure 14-1, notice that each neuron is connected only to nearby neurons from the previous layer). This powerful architecture is able to detect all sorts of complex patterns in any area of the visual field. 

These studies of the visual cortex inspired the neocognitron,<sup>4</sup> introduced in 1980, which gradually evolved into what we now call convolutional neural networks. An important milestone was a 1998 paper<sup>5</sup> by Yann LeCun et al. that introduced the famous _LeNet-5_ architecture, which became widely used by banks to recognize hand‐ written digits on checks. This architecture has some building blocks that you already know, such as fully connected layers and sigmoid activation functions, but it also introduces two new building blocks: _convolutional layers_ and _pooling layers_ . Let’s look at them now. 



Why not simply use a deep neural network with fully connec‐ ted layers for image recognition tasks? Unfortunately, although this works fine for small images (e.g., MNIST), it breaks down for larger images because of the huge number of parameters it requires. For example, a 100 × 100–pixel image has 10,000 pixels, and if the first layer has just 1,000 neurons (which already severely restricts the amount of information transmitted to the next layer), this means a total of 10 million connections. And that’s just the first layer. CNNs solve this problem using partially connected layers and weight sharing. 

## **Convolutional Layers** 

The most important building block of a CNN is the _convolutional layer_ :<sup>6</sup> neurons in the first convolutional layer are not connected to every single pixel in the input image (like they were in the layers discussed in previous chapters), but only to pixels in their 

> 4 Kunihiko Fukushima, “Neocognitron: A Self-Organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position”, _Biological Cybernetics_ 36 (1980): 193–202. 

> 5 Yann LeCun et al., “Gradient-Based Learning Applied to Document Recognition”, _Proceedings of the IEEE_ 86, no. 11 (1998): 2278–2324. 

> 6 A convolution is a mathematical operation that slides one function over another and measures the integral of their pointwise multiplication. It has deep connections with the Fourier transform and the Laplace transform and is heavily used in signal processing. Convolutional layers actually use cross-correlations, which are very similar to convolutions (see _https://homl.info/76_ for more details). 

**Convolutional Layers | 481** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 



<!-- Start of picture text -->
J Ze,Px8 * / fron layer 2<br>¢ .<br>ves - 4- wd<br>fonts * Convolutional layer 1<br>owe a<br>/ AT== 6 Input layer<br><!-- End of picture text -->





<!-- Start of picture text -->
——— j<br>77 7Z_Z_ZZ<br>| JZ ZLZLZLZ LLY<br>[7 7 77<br>[777scam/7/7Z/_72_Z+<br>[YT 7-7) 7 7<br>f.=3 I<br>AY A A A A<br>UZ A A A MY<br>Mu. oh KW 4. WL’<br>fy=3 Zero<br>padding<br><!-- End of picture text -->



<!-- Start of picture text -->
[7_7_Z<br>he 7<br>wes ca<br>S,=2 [TW7 \Z,2Z SP ALZ A77 \<br>Fe ew Aa a AY A ME<br>(4. [giz 777 Z7.-<br>7 7Zi2Z Z'7 7 7 7.»<br>s. Pe) bee /<br>Swe<br><!-- End of picture text -->



<!-- Start of picture text -->
Feature Feature ——<br>*<br>map 1 . ” ' 4 map 2 =<br>- PEa av PNAA AAA athANAS ait iae=,"<br>Vertical filter il = Horizontal filter<br>Input =<br><!-- End of picture text -->



<!-- Start of picture text -->
Convolution layer 2<br>Feature<br>T<br>+<br>*<br>Filters e: Troy<br>woovy<br>al Vy<br>ms Convolution. layer<br>oe ij on 1<br>af \ \<br>Ma ] iYa) fs)~a-a--b-}K fit<br>! tt 1 1<br>I tt 1) !<br>Val =a<br>° ott<br>e tot To<br>° i ay : 7 Zz<br>Oe SL<br>fee<br>1 \ K \<br>im) m1 N VN<br>i H i BN \ Xy | 1<br>peeere- ya Input layer<br>Channels to174 raeifoe7 ¥17<br>Red |<br>GreenBlue Se<br><!-- End of picture text -->



The fact that all neurons in a feature map share the same parame‐ ters dramatically reduces the number of parameters in the model. Once the CNN has learned to recognize a pattern in one location, it can recognize it in any other location. In contrast, once a fully connected neural network has learned to recognize a pattern in one location, it can only recognize it in that particular location. 

Input images are also composed of multiple sublayers: one per _color channel_ . As men‐ tioned in Chapter 9, there are typically three: red, green, and blue (RGB). Grayscale images have just one channel, but some images may have many more—for example, satellite images that capture extra light frequencies (such as infrared). 

Specifically, a neuron located in row _i_ , column _j_ of the feature map _k_ in a given convolutional layer _l_ is connected to the outputs of the neurons in the previous layer _l_ – 1, located in rows _i_ × _sh_ to _i_ × _sh_ + _fh_ – 1 and columns _j_ × _sw_ to _j_ × _sw_ + _fw_ – 1, across all feature maps (in layer _l_ – _1_ ). Note that, within a layer, all neurons located in the same row _i_ and column _j_ but in different feature maps are connected to the outputs of the exact same neurons in the previous layer. 

Equation 14-1 summarizes the preceding explanations in one big mathematical equa‐ tion: it shows how to compute the output of a given neuron in a convolutional layer. It is a bit ugly due to all the different indices, but all it does is calculate the weighted sum of all the inputs, plus the bias term. 

_Equation 14-1. Computing the output of a neuron in a convolutional layer_ 



In this equation: 

- _zi_ , _j_ , _k_ is the output of the neuron located in row _i_ , column _j_ in feature map _k_ of the convolutional layer (layer _l_ ). 

- As explained earlier, _sh_ and _sw_ are the vertical and horizontal strides, _fh_ and _fw_ are the height and width of the receptive field, and _fn_ ′ is the number of feature maps in the previous layer (layer _l_ – 1). 

- _xi_ ′ , _j_ ′ , _k_ ′ is the output of the neuron located in layer _l_ – 1, row _i_ ′ , column _j_ ′ , feature map _k_ ′ (or channel _k_ ′ if the previous layer is the input layer). 

- _bk_ is the bias term for feature map _k_ (in layer _l_ ). You can think of it as a knob that tweaks the overall brightness of the feature map _k_ . 

**486 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

- _wu_ , _v_ , _k_ ′ , _k_ is the connection weight between any neuron in feature map _k_ of the layer _l_ and its input located at row _u_ , column _v_ (relative to the neuron’s receptive field), and feature map _k_ ′ . 

Let’s see how to create and use a convolutional layer using Keras. 

#### **Implementing Convolutional Layers with Keras** 

First, let’s load and preprocess a couple of sample images, using Scikit-Learn’s `load_sample_image()` function and Keras’s `CenterCrop` and `Rescaling` layers (all of which were introduced in Chapter 13): 

```
fromsklearn.datasetsimportload_sample_images
importtensorflowastf
```

```
images=load_sample_images()["images"]
images=tf.keras.layers.CenterCrop(height=70, width=120)(images)
images=tf.keras.layers.Rescaling(scale=1/255)(images)
```

Let’s look at the shape of the `images` tensor: 

```
>>> images.shape
TensorShape([2, 70, 120, 3])
```

Yikes, it’s a 4D tensor; we haven’t seen this before! What do all these dimensions mean? Well, there are two sample images, which explains the first dimension. Then each image is 70 × 120, since that’s the size we specified when creating the `Center Crop` layer (the original images were 427 × 640). This explains the second and third dimensions. And lastly, each pixel holds one value per color channel, and there are three of them—red, green, and blue—which explains the last dimension. 

Now let’s create a 2D convolutional layer and feed it these images to see what comes out. For this, Keras provides a `Convolution2D` layer, alias `Conv2D` . Under the hood, this layer relies on TensorFlow’s `tf.nn.conv2d()` operation. Let’s create a convolutional layer with 32 filters, each of size 7 × 7 (using `kernel_size=7` , which is equivalent to using `kernel_size=(7 , 7)` ), and apply this layer to our small batch of two images: 

```
conv_layer=tf.keras.layers.Conv2D(filters=32, kernel_size=7)
fmaps=conv_layer(images)
```



When we talk about a 2D convolutional layer, “2D” refers to the number of _spatial_ dimensions (height and width), but as you can see, the layer takes 4D inputs: as we saw, the two additional dimen‐ sions are the batch size (first dimension) and the channels (last dimension). 

**Convolutional Layers | 487** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

Now let’s look at the output’s shape: 

```
>>> fmaps.shape
TensorShape([2, 64, 114, 32])
```

The output shape is similar to the input shape, with two main differences. First, there are 32 channels instead of 3. This is because we set `filters=32` , so we get 32 output feature maps: instead of the intensity of red, green, and blue at each location, we now have the intensity of each feature at each location. Second, the height and width have both shrunk by 6 pixels. This is due to the fact that the `Conv2D` layer does not use any zero-padding by default, which means that we lose a few pixels on the sides of the output feature maps, depending on the size of the filters. In this case, since the kernel size is 7, we lose 6 pixels horizontally and 6 pixels vertically (i.e., 3 pixels on each side). 



The default option is surprisingly named `padding="valid"` , which actually means no zero-padding at all! This name comes from the fact that in this case every neuron’s receptive field lies strictly within _valid_ positions inside the input (it does not go out of bounds). It’s not a Keras naming quirk: everyone uses this odd nomenclature. 

If instead we set `padding="same"` , then the inputs are padded with enough zeros on all sides to ensure that the output feature maps end up with the _same_ size as the inputs (hence the name of this option): 

```
>>> conv_layer=tf.keras.layers.Conv2D(filters=32, kernel_size=7,
... padding="same")
...
>>> fmaps=conv_layer(images)
>>> fmaps.shape
TensorShape([2, 70, 120, 32])
```

These two padding options are illustrated in Figure 14-7. For simplicity, only the horizontal dimension is shown here, but of course the same logic applies to the vertical dimension as well. 

If the stride is greater than 1 (in any direction), then the output size will not be equal to the input size, even if `padding="same"` . For example, if you set `strides=2` (or equivalently `strides=(2, 2)` ), then the output feature maps will be 35 × 60: halved both vertically and horizontally. Figure 14-8 shows what happens when `strides=2` , with both padding options. 

**488 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 



<!-- Start of picture text -->
Ct TTT ttt ty<br>|_|<br>|_|<br>mi<br>|<br>|_|<br>+} — Zero<br>— padding<br>padding=" ") lofofo] | |padding="T | | TT", tT fele])<br>kernel_size=7, strides=1 kernel_size=7, strides=1<br><!-- End of picture text -->



<!-- Start of picture text -->
Ignored Zero<br>| padding<br>| 3 : : | | a (lolo} |} | TT TT I Yt fefelo)4<br>padding=" ") padding=" ")<br>kernel_size=7, strides=2 kernel_size=7, strides=2<br><!-- End of picture text -->

Now let’s look at the layer’s weights (which were noted _wu_ , _v_ , _k_ ′ , _k_ and _bk_ in Equation 14-1). Just like a `Dense` layer, a `Conv2D` layer holds all the layer’s weights, including the kernels and biases. The kernels are initialized randomly, while the biases are initialized to zero. These weights are accessible as TF variables via the `weights` attribute, or as NumPy arrays via the `get_weights()` method: 

```
>>> kernels, biases=conv_layer.get_weights()
>>> kernels.shape
(7, 7, 3, 32)
>>> biases.shape
(32,)
```

The `kernels` array is 4D, and its shape is [ _kernel_height_ , _kernel_width_ , _input_channels_ , _output_channels_ ]. The `biases` array is 1D, with shape [ _output_channels_ ]. The number of output channels is equal to the number of output feature maps, which is also equal to the number of filters. 

Most importantly, note that the height and width of the input images do not appear in the kernel’s shape: this is because all the neurons in the output feature maps share the same weights, as explained earlier. This means that you can feed images of any size to this layer, as long as they are at least as large as the kernels, and if they have the right number of channels (three in this case). 

Lastly, you will generally want to specify an activation function (such as ReLU) when creating a `Conv2D` layer, and also specify the corresponding kernel initializer (such as He initialization). This is for the same reason as for `Dense` layers: a convolutional layer performs a linear operation, so if you stacked multiple convolutional layers without any activation functions they would all be equivalent to a single convolu‐ tional layer, and they wouldn’t be able to learn anything really complex. 

As you can see, convolutional layers have quite a few hyperparameters: `filters` , `kernel_size` , `padding` , `strides` , `activation` , `kernel_initializer` , etc. As always, you can use cross-validation to find the right hyperparameter values, but this is very time-consuming. We will discuss common CNN architectures later in this chapter, to give you some idea of which hyperparameter values work best in practice. 

#### **Memory Requirements** 

Another challenge with CNNs is that the convolutional layers require a huge amount of RAM. This is especially true during training, because the reverse pass of backpro‐ pagation requires all the intermediate values computed during the forward pass. 

For example, consider a convolutional layer with 200 5 × 5 filters, with stride 1 and `"same"` padding. If the input is a 150 × 100 RGB image (three channels), then the number of parameters is (5 × 5 × 3 + 1) × 200 = 15,200 (the + 1 corresponds to 

**490 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

the bias terms), which is fairly small compared to a fully connected layer.<sup>7</sup> However, each of the 200 feature maps contains 150 × 100 neurons, and each of these neurons needs to compute a weighted sum of its 5 × 5 × 3 = 75 inputs: that’s a total of 225 million float multiplications. Not as bad as a fully connected layer, but still quite computationally intensive. Moreover, if the feature maps are represented using 32-bit floats, then the convolutional layer’s output will occupy 200 × 150 × 100 × 32 = 96 million bits (12 MB) of RAM.<sup>8</sup> And that’s just for one instance—if a training batch contains 100 instances, then this layer will use up 1.2 GB of RAM! 

During inference (i.e., when making a prediction for a new instance) the RAM occupied by one layer can be released as soon as the next layer has been computed, so you only need as much RAM as required by two consecutive layers. But during training everything computed during the forward pass needs to be preserved for the reverse pass, so the amount of RAM needed is (at least) the total amount of RAM required by all layers. 



If training crashes because of an out-of-memory error, you can try reducing the mini-batch size. Alternatively, you can try reducing dimensionality using a stride, removing a few layers, using 16-bit floats instead of 32-bit floats, or distributing the CNN across multi‐ ple devices (you will see how to do this in Chapter 19). 

Now let’s look at the second common building block of CNNs: the _pooling layer_ . 

## **Pooling Layers** 

Once you understand how convolutional layers work, the pooling layers are quite easy to grasp. Their goal is to _subsample_ (i.e., shrink) the input image in order to reduce the computational load, the memory usage, and the number of parameters (thereby limiting the risk of overfitting). 

Just like in convolutional layers, each neuron in a pooling layer is connected to the outputs of a limited number of neurons in the previous layer, located within a small rectangular receptive field. You must define its size, the stride, and the padding type, just like before. However, a pooling neuron has no weights; all it does is aggregate the inputs using an aggregation function such as the max or mean. Figure 14-9 shows a _max pooling layer_ , which is the most common type of pooling layer. In this example, 

> 7 To produce the same size outputs, a fully connected layer would need 200 × 150 × 100 neurons, each connected to all 150 × 100 × 3 inputs. It would have 200 × 150 × 100 × (150 × 100 × 3 + 1) ≈ 135 billion parameters! 

> 8 In the international system of units (SI), 1 MB = 1,000 KB = 1,000 × 1,000 bytes = 1,000 × 1,000 × 8 bits. And 1 MiB = 1,024 kiB = 1,024 × 1,024 bytes. So 12 MB ≈ 11.44 MiB. 

**Pooling Layers | 491** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 



<!-- Start of picture text -->
eo<br>a aan<br>a 4<br>177 7/7<br>Max OTT tT<br>on<br>Zz j<br><!-- End of picture text -->





<!-- Start of picture text -->
YT||L TTEyET YT||L TTTTEyET YL || | |<br>PTT|  TTT Tt |} EETtT TT |) LETT TTT<br>PTET Pt ty LE Pty LL |<br>PE EeLEtT tT | LETey Ee | LEELEEET<br>|PE| | Tt| |} EtCE Pt} ELTLEE ||<br>PEet Te tty LET ey LETT ET<br>an PEt) ULL rt) CLL B<br>A B C<br><!-- End of picture text -->



<!-- Start of picture text -->
Depthwise max pooling layer yn<br>Max==aC<br>Convolution layer TE, &Learned filters<br>: 7/7 5<br><!-- End of picture text -->

This layer reshapes its inputs to split the channels into groups of the desired size ( `pool_size` ), then it uses `tf.reduce_max()` to compute the max of each group. This implementation assumes that the stride is equal to the pool size, which is generally what you want. Alternatively, you could use TensorFlow’s `tf.nn.max_pool()` opera‐ tion, and wrap in a `Lambda` layer to use it inside a Keras model, but sadly this op does not implement depthwise pooling for the GPU, only for the CPU. 

One last type of pooling layer that you will often see in modern architectures is the _global average pooling layer_ . It works very differently: all it does is compute the mean of each entire feature map (it’s like an average pooling layer using a pooling kernel with the same spatial dimensions as the inputs). This means that it just outputs a single number per feature map and per instance. Although this is of course extremely destructive (most of the information in the feature map is lost), it can be useful just before the output layer, as you will see later in this chapter. To create such a layer, simply use the `GlobalAveragePooling2D` class, alias `GlobalAvgPool2D` : 

```
global_avg_pool=tf.keras.layers.GlobalAvgPool2D()
```

It’s equivalent to the following `Lambda` layer, which computes the mean over the spatial dimensions (height and width): 

```
global_avg_pool=tf.keras.layers.Lambda(
lambdaX: tf.reduce_mean(X, axis=[1, 2]))
```

For example, if we apply this layer to the input images, we get the mean intensity of red, green, and blue for each image: 

```
>>> global_avg_pool(images)
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[0.64338624, 0.5971759 , 0.5824972 ],
       [0.76306933, 0.26011038, 0.10849128]], dtype=float32)>
```

Now you know all the building blocks to create convolutional neural networks. Let’s see how to assemble them. 

## **CNN Architectures** 

Typical CNN architectures stack a few convolutional layers (each one generally fol‐ lowed by a ReLU layer), then a pooling layer, then another few convolutional layers (+ReLU), then another pooling layer, and so on. The image gets smaller and smaller as it progresses through the network, but it also typically gets deeper and deeper (i.e., with more feature maps), thanks to the convolutional layers (see Figure 14-12). At the top of the stack, a regular feedforward neural network is added, composed of a few fully connected layers (+ReLUs), and the final layer outputs the prediction (e.g., a softmax layer that outputs estimated class probabilities). 

**CNN Architectures | 495** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 





Let’s go through this code: 

- We use the `functools.partial()` function (introduced in Chapter 11) to define `DefaultConv2D` , which acts just like `Conv2D` but with different default arguments: a small kernel size of 3, `"same"` padding, the ReLU activation function, and its corresponding He initializer. 

- Next, we create the `Sequential` model. Its first layer is a `DefaultConv2D` with 64 fairly large filters (7 × 7). It uses the default stride of 1 because the input images are not very large. It also sets `input_shape=[28, 28, 1]` , because the images are 28 × 28 pixels, with a single color channel (i.e., grayscale). When you load the Fashion MNIST dataset, make sure each image has this shape: you may need to use `np.reshape()` or `np.expanddims()` to add the channels dimension. Alternatively, you could use a `Reshape` layer as the first layer in the model. 

- We then add a max pooling layer that uses the default pool size of 2, so it divides each spatial dimension by a factor of 2. 

- Then we repeat the same structure twice: two convolutional layers followed by a max pooling layer. For larger images, we could repeat this structure several more times. The number of repetitions is a hyperparameter you can tune. 

- Note that the number of filters doubles as we climb up the CNN toward the output layer (it is initially 64, then 128, then 256): it makes sense for it to grow, since the number of low-level features is often fairly low (e.g., small circles, horizontal lines), but there are many different ways to combine them into higherlevel features. It is a common practice to double the number of filters after each pooling layer: since a pooling layer divides each spatial dimension by a factor of 2, we can afford to double the number of feature maps in the next layer without fear of exploding the number of parameters, memory usage, or computational load. 

- Next is the fully connected network, composed of two hidden dense layers and a dense output layer. Since it’s a classification task with 10 classes, the output layer has 10 units, and it uses the softmax activation function. Note that we must flatten the inputs just before the first dense layer, since it expects a 1D array of features for each instance. We also add two dropout layers, with a dropout rate of 50% each, to reduce overfitting. 

If you compile this model using the `"sparse_categorical_crossentropy"` loss and you fit the model to the Fashion MNIST training set, it should reach over 92% accuracy on the test set. It’s not state of the art, but it is pretty good, and clearly much better than what we achieved with dense networks in Chapter 10. 

**CNN Architectures | 497** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

Over the years, variants of this fundamental architecture have been developed, lead‐ ing to amazing advances in the field. A good measure of this progress is the error rate in competitions such as the ILSVRC ImageNet challenge. In this competition, the top-five error rate for image classification—that is, the number of test images for which the system’s top five predictions did _not_ include the correct answer—fell from over 26% to less than 2.3% in just six years. The images are fairly large (e.g., 256 pixels high) and there are 1,000 classes, some of which are really subtle (try distinguishing 120 dog breeds). Looking at the evolution of the winning entries is a good way to understand how CNNs work, and how research in deep learning progresses. 

We will first look at the classical LeNet-5 architecture (1998), then several winners of the ILSVRC challenge: AlexNet (2012), GoogLeNet (2014), ResNet (2015), and SENet (2017). Along the way, we will also look at a few more architectures, including Xception, ResNeXt, DenseNet, MobileNet, CSPNet, and EfficientNet. 

#### **LeNet-5** 

The LeNet-5 architecture<sup>10</sup> is perhaps the most widely known CNN architecture. As mentioned earlier, it was created by Yann LeCun in 1998 and has been widely used for handwritten digit recognition (MNIST). It is composed of the layers shown in Table 14-1. 

_Table 14-1. LeNet-5 architecture_ 

|**Layer **|**Type**|**Maps**|**Size**|**Kernel size**|**Stride**|**Activation**|
|---|---|---|---|---|---|---|
|Out|Fully connected|–|10|–|–|RBF|
|F6|Fully connected|–|84|–|–|tanh|
|C5|Convolution|120|1 × 1|5 × 5|1|tanh|
|S4|Avg pooling|16|5 × 5|2 × 2|2|tanh|
|C3|Convolution|16|10 × 10|5 × 5|1|tanh|
|S2|Avg pooling|6|14 × 14|2 × 2|2|tanh|
|C1|Convolution|6|28 × 28|5 × 5|1|tanh|
|In|Input|1|32 × 32|–|–|–|



As you can see, this looks pretty similar to our Fashion MNIST model: a stack of convolutional layers and pooling layers, followed by a dense network. Perhaps the main difference with more modern classification CNNs is the activation functions: today, we would use ReLU instead of tanh and softmax instead of RBF. There were 

> 10 Yann LeCun et al., “Gradient-Based Learning Applied to Document Recognition”, _Proceedings of the IEEE_ 86, no. 11 (1998): 2278–2324. 

**498 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

several other minor differences that don’t really matter much, but in case you are interested, they are listed in this chapter’s notebook at _https://homl.info/colab3_ . Yann LeCun’s website also features great demos of LeNet-5 classifying digits. 

#### **AlexNet** 

The AlexNet CNN architecture<sup>11</sup> won the 2012 ILSVRC challenge by a large margin: it achieved a top-five error rate of 17%, while the second best competitor achieved only 26%! AlexaNet was developed by Alex Krizhevsky (hence the name), Ilya Sutsk‐ ever, and Geoffrey Hinton. It is similar to LeNet-5, only much larger and deeper, and it was the first to stack convolutional layers directly on top of one another, instead of stacking a pooling layer on top of each convolutional layer. Table 14-2 presents this architecture. 

_Table 14-2. AlexNet architecture_ 

|**Layer**|**Type**|**Maps**|**Size**|**Kernel size **|**Stride**|**Padding **|**Activation**|
|---|---|---|---|---|---|---|---|
|Out|Fully connected|–|1,000|–|–|–|Softmax|
|F10|Fully connected|–|4,096|–|–|–|ReLU|
|F9|Fully connected|–|4,096|–|–|–|ReLU|
|S8|Max pooling|256|6 × 6|3 × 3|2|`valid`|–|
|C7|Convolution|256|13 × 13|3 × 3|1|`same`|ReLU|
|C6|Convolution|384|13 × 13|3 × 3|1|`same`|ReLU|
|C5|Convolution|384|13 × 13|3 × 3|1|`same`|ReLU|
|S4|Max pooling|256|13 × 13|3 × 3|2|`valid`|–|
|C3|Convolution|256|27 × 27|5 × 5|1|`same`|ReLU|
|S2|Max pooling|96|27 × 27|3 × 3|2|`valid`|–|
|C1|Convolution|96|55 × 55|11 × 11|4|`valid`|ReLU|
|In|Input|3 (RGB)|227 × 227|–|–|–|–|



To reduce overfitting, the authors used two regularization techniques. First, they applied dropout (introduced in Chapter 11) with a 50% dropout rate during training to the outputs of layers F9 and F10. Second, they performed data augmentation by randomly shifting the training images by various offsets, flipping them horizontally, and changing the lighting conditions. 

> 11 Alex Krizhevsky et al., “ImageNet Classification with Deep Convolutional Neural Networks”, _Proceedings of the 25th International Conference on Neural Information Processing Systems_ 1 (2012): 1097–1105. 

**CNN Architectures | 499** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 



<!-- Start of picture text -->
g r- (me<br>OD a 43 i 7 oe x, aatee4<br>cow x<br>4a1M : :<br>ai. ‘i ;Py .mg2ee .|<br>Z‘ en —<br>% a Nel. :<br>: 4 a GT iD<br>wat tee aa ~ A gf, +<br>©<br>< | ee 4<br>4 oe<br>| ~<br>o !<br><!-- End of picture text -->

AlexNet also uses a competitive normalization step immediately after the ReLU step of layers C1 and C3, called _local response normalization_ (LRN): the most strongly activated neurons inhibit other neurons located at the same position in neighboring feature maps. Such competitive activation has been observed in biological neurons. This encourages different feature maps to specialize, pushing them apart and forcing them to explore a wider range of features, ultimately improving generalization. Equa‐ tion 14-2 shows how to apply LRN. 

_Equation 14-2. Local response normalization (LRN)_ 



In this equation: 

- _bi_ is the normalized output of the neuron located in feature map _i_ , at some row _u_ and column _v_ (note that in this equation we consider only neurons located at this row and column, so _u_ and _v_ are not shown). 

- _ai_ is the activation of that neuron after the ReLU step, but before normalization. 

- _k_ , _α_ , _β_ , and _r_ are hyperparameters. _k_ is called the _bias_ , and _r_ is called the _depth radius_ . 

- _fn_ is the number of feature maps. 

For example, if _r_ = 2 and a neuron has a strong activation, it will inhibit the activation of the neurons located in the feature maps immediately above and below its own. 

In AlexNet, the hyperparameters are set as: _r_ = 5, _α_ = 0.0001, _β_ = 0.75, and _k_ = 2. You can implement this step by using the `tf.nn.local_response_normalization()` function (which you can wrap in a `Lambda` layer if you want to use it in a Keras model). 

A variant of AlexNet called _ZF Net_<sup>12</sup> was developed by Matthew Zeiler and Rob Fergus and won the 2013 ILSVRC challenge. It is essentially AlexNet with a few tweaked hyperparameters (number of feature maps, kernel size, stride, etc.). 

> 12 Matthew D. Zeiler and Rob Fergus, “Visualizing and Understanding Convolutional Networks”, _Proceedings of the European Conference on Computer Vision_ (2014): 818–833. 

**CNN Architectures | 501** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 



<!-- Start of picture text -->
Depth<br>concat<br>Convolution Convolution Convolution Convolution<br>1x1 + 1(S) 3x3 + 1(S) 5x5 + 1(S) 1x1 + 1(S)<br>Convolution Convolution Max pool<br>1x1 + 1(S) 1x1 + 1(S) 3x3+1(S)<br><!-- End of picture text -->

You may wonder why inception modules have convolutional layers with 1 × 1 ker‐ nels. Surely these layers cannot capture any features because they look at only one pixel at a time, right? In fact, these layers serve three purposes: 

- Although they cannot capture spatial patterns, they can capture patterns along the depth dimension (i.e., across channels). 

- They are configured to output fewer feature maps than their inputs, so they serve as _bottleneck layers_ , meaning they reduce dimensionality. This cuts the computational cost and the number of parameters, speeding up training and improving generalization. 

- Each pair of convolutional layers ([1 × 1, 3 × 3] and [1 × 1, 5 × 5]) acts like a single powerful convolutional layer, capable of capturing more complex patterns. A convolutional layer is equivalent to sweeping a dense layer across the image (at each location, it only looks at a small receptive field), and these pairs of convolutional layers are equivalent to sweeping two-layer neural networks across the image. 

In short, you can think of the whole inception module as a convolutional layer on steroids, able to output feature maps that capture complex patterns at various scales. 

Now let’s look at the architecture of the GoogLeNet CNN (see Figure 14-15). The number of feature maps output by each convolutional layer and each pooling layer is shown before the kernel size. The architecture is so deep that it has to be represented in three columns, but GoogLeNet is actually one tall stack, including nine inception modules (the boxes with the spinning tops). The six numbers in the inception modules represent the number of feature maps output by each convolutional layer in the module (in the same order as in Figure 14-14). Note that all the convolutional layers use the ReLU activation function. 

Let’s go through this network: 

- The first two layers divide the image’s height and width by 4 (so its area is divided by 16), to reduce the computational load. The first layer uses a large kernel size, 7 × 7, so that much of the information is preserved. 

- Then the local response normalization layer ensures that the previous layers learn a wide variety of features (as discussed earlier). 

- Two convolutional layers follow, where the first acts like a bottleneck layer. As mentioned, you can think of this pair as a single smarter convolutional layer. 

- Again, a local response normalization layer ensures that the previous layers capture a wide variety of patterns. 

- Next, a max pooling layer reduces the image height and width by 2, again to speed up computations. 

**CNN Architectures | 503** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 



<!-- Start of picture text -->
112 288 64 64<br>Max pool ef 14432 Fully connected<br>Local response x 128 24 Dropout 40%<br>normalization 160 224 64 64<br>Convolution sf 11224 Global avg pool<br>Convolution x 9616 384 384 128 128<br>64, 1*1+1(S) sf 19248<br>Local response 480, 3x3+2(S) 256 320 128 128<br>Max pool sf 12832 Max pool<br>Convolution x 9612 256 320 128 128<br>64, 7x7+2(S) N¢ 160 32<br>x = Inception module ;<br><!-- End of picture text -->

The original GoogLeNet architecture included two auxiliary classifiers plugged on top of the third and sixth inception modules. They were both composed of one aver‐ age pooling layer, one convolutional layer, two fully connected layers, and a softmax activation layer. During training, their loss (scaled down by 70%) was added to the overall loss. The goal was to fight the vanishing gradients problem and regularize the network, but it was later shown that their effect was relatively minor. 

Several variants of the GoogLeNet architecture were later proposed by Google researchers, including Inception-v3 and Inception-v4, using slightly different incep‐ tion modules to reach even better performance. 

#### **VGGNet** 

The runner-up in the ILSVRC 2014 challenge was VGGNet,<sup>15</sup> Karen Simonyan and Andrew Zisserman, from the Visual Geometry Group (VGG) research lab at Oxford University, developed a very simple and classical architecture; it had 2 or 3 convolu‐ tional layers and a pooling layer, then again 2 or 3 convolutional layers and a pooling layer, and so on (reaching a total of 16 or 19 convolutional layers, depending on the VGG variant), plus a final dense network with 2 hidden layers and the output layer. It used small 3 × 3 filters, but it had many of them. 

#### **ResNet** 

Kaiming He et al. won the ILSVRC 2015 challenge using a Residual Network (ResNet)<sup>16</sup> that delivered an astounding top-five error rate under 3.6%. The winning variant used an extremely deep CNN composed of 152 layers (other variants had 34, 50, and 101 layers). It confirmed the general trend: computer vision models were getting deeper and deeper, with fewer and fewer parameters. The key to being able to train such a deep network is to use _skip connections_ (also called _shortcut connections_ ): the signal feeding into a layer is also added to the output of a layer located higher up the stack. Let’s see why this is useful. 

When training a neural network, the goal is to make it model a target function _h_ ( **x** ). If you add the input **x** to the output of the network (i.e., you add a skip connection), then the network will be forced to model _f_ ( **x** ) = _h_ ( **x** ) – **x** rather than _h_ ( **x** ). This is called _residual learning_ (see Figure 14-16). 

> 15 Karen Simonyan and Andrew Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recog‐ nition”, arXiv preprint arXiv:1409.1556 (2014). 

> 16 Kaiming He et al., “Deep Residual Learning for Image Recognition”, arXiv preprint arXiv:1512:03385 (2015). 

**CNN Architectures | 505** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 



<!-- Start of picture text -->
h(x)<br>h(x)<br>I(x) Skip “hiyy.<br>connection f(x)=h(x)-x<br><!-- End of picture text -->



<!-- Start of picture text -->
a =<br>=<br>Pae<cr [| x<br>| [> >><c7 | Residual units<br>(ae Y<br>jl x =Layerbackpropagation blocking |1<br>; =Layer not learning ;<br>a|<br><!-- End of picture text -->



<!-- Start of picture text -->
iy/ Convolution:<br>Fully connected / Convolution<br>1000 units 1 128, 3x3+1(S)<br>Global avg pool / Fad Convolution<br>1024 G ' 128, 3x3+1(S)<br>Convolution<br>MM 128, 3x3+2(S) — pay — ReLU<br>onvoiution<br>a) +>+—— Skip<br>9 Ea 7 «— Batch norm<br>64, 3x3+2(S) ‘ | 64, 3x3+1(S) Convolution + e<br>Convolution ‘ Convolution 64, 3x3+1(S)<br>64,7*742(5) J 64, 3x3+1(5) Residual unit<br>\\ Convolution.<br>\:<br>\.<br>\ :<br><!-- End of picture text -->



<!-- Start of picture text -->
ReLU<br>poorer rer er renee<br>H<br>H <— BN<br>Convolution 128, 3x3+1(S)<br>128, 1x1+2(S) j y¢— BN+ReLU<br>Convolution<br>128, 3x3+2(S)<br>i]<br><!-- End of picture text -->



output 256 maps, then 8 RUs with 512 maps, a whopping 36 RUs with 1,024 maps, and finally 3 RUs with 2,048 maps. 

#### **Xception** 

Another variant of the GoogLeNet architecture is worth noting: Xception<sup>19</sup> (which stands for _Extreme Inception_ ) was proposed in 2016 by François Chollet (the author of Keras), and it significantly outperformed Inception-v3 on a huge vision task (350 million images and 17,000 classes). Just like Inception-v4, it merges the ideas of GoogLeNet and ResNet, but it replaces the inception modules with a special type of layer called a _depthwise separable convolution layer_ (or _separable convolution layer_ for short<sup>20</sup> ). These layers had been used before in some CNN architectures, but they were not as central as in the Xception architecture. While a regular convolutional layer uses filters that try to simultaneously capture spatial patterns (e.g., an oval) and cross-channel patterns (e.g., mouth + nose + eyes = face), a separable convolutional layer makes the strong assumption that spatial patterns and cross-channel patterns can be modeled separately (see Figure 14-20). Thus, it is composed of two parts: the first part applies a single spatial filter to each input feature map, then the second part looks exclusively for cross-channel patterns—it is just a regular convolutional layer with 1 × 1 filters. 

Since separable convolutional layers only have one spatial filter per input channel, you should avoid using them after layers that have too few channels, such as the input layer (granted, that’s what Figure 14-20 represents, but it is just for illustration pur‐ poses). For this reason, the Xception architecture starts with 2 regular convolutional layers, but then the rest of the architecture uses only separable convolutions (34 in all), plus a few max pooling layers and the usual final layers (a global average pooling layer and a dense output layer). 

You might wonder why Xception is considered a variant of GoogLeNet, since it con‐ tains no inception modules at all. Well, as discussed earlier, an inception module con‐ tains convolutional layers with 1 × 1 filters: these look exclusively for cross-channel patterns. However, the convolutional layers that sit on top of them are regular convo‐ lutional layers that look both for spatial and cross-channel patterns. So, you can think of an inception module as an intermediate between a regular convolutional layer (which considers spatial patterns and cross-channel patterns jointly) and a separable convolutional layer (which considers them separately). In practice, it seems that separable convolutional layers often perform better. 

> 19 François Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions”, arXiv preprint arXiv:1610.02357 (2016). 

> 20 This name can sometimes be ambiguous, since spatially separable convolutions are often called “separable convolutions” as well. 

**CNN Architectures | 509** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 



<!-- Start of picture text -->
Feature map 1 Regular convolutional<br>layer with depthwise-<br>only filters (1x1)<br>Map 2<br>e<br>e<br>e<br>7 /,I, (1S p atial-onlyer input channel) filters<br>ae<br>1S Filter1 Filter2. Filter 3 mh<br><!-- End of picture text -->







<!-- Start of picture text -->
=Featuremaps | { seblock 10 featureRecalibrated maps<br>1.0<br>03<br>1.0<br>0.2<br><!-- End of picture text -->



<!-- Start of picture text -->
(ene ReLUTt<br>Global average pool<br><!-- End of picture text -->

###### _DenseNet_<sup>_23_</sup> 

A DenseNet is composed of several dense blocks, each made up of a few densely connected convolutional layers. This architecture achieved excellent accuracy while using comparatively few parameters. What does “densely connected” mean? The output of each layer is fed as input to every layer after it within the same block. For example, layer 4 in a block takes as input the depthwise concatenation of the outputs of layers 1, 2, and 3 in that block. Dense blocks are separated by a few transition layers. 

###### _MobileNet_<sup>_24_</sup> 

MobileNets are streamlined models designed to be lightweight and fast, making them popular in mobile and web applications. They are based on depthwise sepa‐ rable convolutional layers, like Xception. The authors proposed several variants, trading a bit of accuracy for faster and smaller models. 

###### _CSPNet_<sup>_25_</sup> 

A Cross Stage Partial Network (CSPNet) is similar to a DenseNet, but part of each dense block’s input is concatenated directly to that block’s output, without going through the block. 

###### _EfficientNet_<sup>_26_</sup> 

EfficientNet is arguably the most important model in this list. The authors proposed a method to scale any CNN efficiently, by jointly increasing the depth (number of layers), width (number of filters per layer), and resolution (size of the input image) in a principled way. This is called _compound scaling_ . They used neural architecture search to find a good architecture for a scaled-down version of ImageNet (with smaller and fewer images), and then used compound scaling to create larger and larger versions of this architecture. When EfficientNet mod‐ els came out, they vastly outperformed all existing models, across all compute budgets, and they remain among the best models out there today. 

Understanding EfficientNet’s compound scaling method is helpful to gain a deeper understanding of CNNs, especially if you ever need to scale a CNN architecture. It is based on a logarithmic measure of the compute budget, noted _ϕ_ : if your compute budget doubles, then _ϕ_ increases by 1. In other words, the number of floating-point operations available for training is proportional to 2<sup>_ϕ_</sup> . Your CNN architecture’s depth, 

> 23 Gao Huang et al., “Densely Connected Convolutional Networks”, arXiv preprint arXiv:1608.06993 (2016). 

- 24 Andrew G. Howard et al., “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applica‐ tions”, arXiv preprint arxiv:1704.04861 (2017). 

- 25 Chien-Yao Wang et al., “CSPNet: A New Backbone That Can Enhance Learning Capability of CNN”, arXiv preprint arXiv:1911.11929 (2019). 

- 26 Mingxing Tan and Quoc V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks”, arXiv preprint arXiv:1905.11946 (2019). 

**CNN Architectures | 513** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

width, and resolution should scale as _α_<sup>_ϕ_</sup> , _β_<sup>_ϕ_</sup> , and _γ_<sup>_ϕ_</sup> , respectively. The factors _α_ , _β_ , and _γ_ must be greater than 1, and _αβ_<sup>2</sup> _γ_<sup>2</sup> should be close to 2. The optimal values for these factors depend on the CNN’s architecture. To find the optimal values for the EfficientNet architecture, the authors started with a small baseline model (EfficientNetB0), fixed _ϕ_ = 1, and simply ran a grid search: they found α = 1.2, β = 1.1, and γ = 1.1. They then used these factors to create several larger architectures, named EfficientNetB1 to EfficientNetB7, for increasing values of _ϕ_ . 

#### **Choosing the Right CNN Architecture** 

With so many CNN architectures, how do you choose which one is best for your project? Well, it depends on what matters most to you: Accuracy? Model size (e.g., for deployment to a mobile device)? Inference speed on CPU? On GPU? Table 14-3 lists the best pretrained models currently available in Keras (you’ll see how to use them later in this chapter), sorted by model size. You can find the full list at _https:// keras.io/api/applications_ . For each model, the table shows the Keras class name to use (in the `tf.keras.applications` package), the model’s size in MB, the top-1 and top-5 validation accuracy on the ImageNet dataset, the number of parameters (millions), and the inference time on CPU and GPU in ms, using batches of 32 images on reasonably powerful hardware.<sup>27</sup> For each column, the best value is high‐ lighted. As you can see, larger models are generally more accurate, but not always; for example, EfficientNetB2 outperforms InceptionV3 both in size and accuracy. I only kept InceptionV3 in the list because it is almost twice as fast as EfficientNetB2 on a CPU. Similarly, InceptionResNetV2 is fast on a CPU, and ResNet50V2 and ResNet101V2 are blazingly fast on a GPU. 

_Table 14-3. Pretrained models available in Keras_ 

|**Class name**|**Size (MB)**|**Top-1 acc**|**Top-5 acc**|**Params**|**CPU (ms)**|**GPU (ms)**|
|---|---|---|---|---|---|---|
|MobileNetV2|**14**|71.3%|90.1%|**3.5M**|25.9|3.8|
|MobileNet|16|70.4%|89.5%|4.3M|**22.6**|**3.4**|
|NASNetMobile|23|74.4%|91.9%|5.3M|27.0|6.7|
|EfcientNetB0|29|77.1%|93.3%|5.3M|46.0|4.9|
|EfcientNetB1|31|79.1%|94.4%|7.9M|60.2|5.6|
|EfcientNetB2|36|80.1%|94.9%|9.2M|80.8|6.5|
|EfcientNetB3|48|81.6%|95.7%|12.3M|140.0|8.8|
|EfcientNetB4|75|82.9%|96.4%|19.5M|308.3|15.1|
|InceptionV3|92|77.9%|93.7%|23.9M|42.2|6.9|
|ResNet50V2|98|76.0%|93.0%|25.6M|45.6|4.4|



27 A 92-core AMD EPYC CPU with IBPB, 1.7 TB of RAM, and an Nvidia Tesla A100 GPU. 

###### **514 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

|**Class name**|**Size (MB)**|**Top-1 acc**|**Top-5 acc**|**Params**|**CPU (ms)**|**GPU (ms)**|
|---|---|---|---|---|---|---|
|EfcientNetB5|118|83.6%|96.7%|30.6M|579.2|25.3|
|EfcientNetB6|166|84.0%|96.8%|43.3M|958.1|40.4|
|ResNet101V2|171|77.2%|93.8%|44.7M|72.7|5.4|
|InceptionResNetV2|215|80.3%|95.3%|55.9M|130.2|10.0|
|EfcientNetB7|256|**84.3%**|**97.0%**|66.7M|1578.9|61.6|



I hope you enjoyed this deep dive into the main CNN architectures! Now let’s see how to implement one of them using Keras. 

## **Implementing a ResNet-34 CNN Using Keras** 

Most CNN architectures described so far can be implemented pretty naturally using Keras (although generally you would load a pretrained network instead, as you will see). To illustrate the process, let’s implement a ResNet-34 from scratch with Keras. First, we’ll create a `ResidualUnit` layer: 

```
DefaultConv2D=partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,
padding="same", kernel_initializer="he_normal",
use_bias=False)
classResidualUnit(tf.keras.layers.Layer):
def __init__(self, filters, strides=1, activation="relu", **kwargs):
super().__init__(**kwargs)
self.activation=tf.keras.activations.get(activation)
self.main_layers= [
DefaultConv2D(filters, strides=strides),
tf.keras.layers.BatchNormalization(),
self.activation,
DefaultConv2D(filters),
tf.keras.layers.BatchNormalization()
        ]
self.skip_layers= []
ifstrides>1:
self.skip_layers= [
DefaultConv2D(filters, kernel_size=1, strides=strides),
tf.keras.layers.BatchNormalization()
            ]
defcall(self, inputs):
Z=inputs
forlayerinself.main_layers:
Z=layer(Z)
skip_Z=inputs
forlayerinself.skip_layers:
skip_Z=layer(skip_Z)
returnself.activation(Z+skip_Z)
```

**Implementing a ResNet-34 CNN Using Keras | 515** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

As you can see, this code matches Figure 14-19 pretty closely. In the constructor, we create all the layers we will need: the main layers are the ones on the right side of the diagram, and the skip layers are the ones on the left (only needed if the stride is greater than 1). Then in the `call()` method, we make the inputs go through the main layers and the skip layers (if any), and we add both outputs and apply the activation function. 

Now we can build a ResNet-34 using a `Sequential` model, since it’s really just a long sequence of layers—we can treat each residual unit as a single layer now that we have the `ResidualUnit` class. The code closely matches Figure 14-18: 

```
model=tf.keras.Sequential([
DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 3]),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation("relu"),
tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
])
prev_filters=64
forfiltersin [64] *3+ [128] *4+ [256] *6+ [512] *3:
strides=1iffilters==prev_filterselse2
model.add(ResidualUnit(filters, strides=strides))
prev_filters=filters
```

```
model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))
```

The only tricky part in this code is the loop that adds the `ResidualUnit` layers to the model: as explained earlier, the first 3 RUs have 64 filters, then the next 4 RUs have 128 filters, and so on. At each iteration, we must set the stride to 1 when the number of filters is the same as in the previous RU, or else we set it to 2; then we add the `ResidualUnit` , and finally we update `prev_filters` . 

It is amazing that in about 40 lines of code, we can build the model that won the ILSVRC 2015 challenge! This demonstrates both the elegance of the ResNet model and the expressiveness of the Keras API. Implementing the other CNN architectures is a bit longer, but not much harder. However, Keras comes with several of these architectures built in, so why not use them instead? 

## **Using Pretrained Models from Keras** 

In general, you won’t have to implement standard models like GoogLeNet or ResNet manually, since pretrained networks are readily available with a single line of code in the `tf.keras.applications` package. 

**516 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

For example, you can load the ResNet-50 model, pretrained on ImageNet, with the following line of code: 

```
model=tf.keras.applications.ResNet50(weights="imagenet")
```

That’s all! This will create a ResNet-50 model and download weights pretrained on the ImageNet dataset. To use it, you first need to ensure that the images have the right size. A ResNet-50 model expects 224 × 224–pixel images (other models may expect other sizes, such as 299 × 299), so let’s use Keras’s `Resizing` layer (introduced in Chapter 13) to resize two sample images (after cropping them to the target aspect ratio): 

```
images=load_sample_images()["images"]
images_resized=tf.keras.layers.Resizing(height=224, width=224,
crop_to_aspect_ratio=True)(images)
```

The pretrained models assume that the images are preprocessed in a specific way. In some cases they may expect the inputs to be scaled from 0 to 1, or from –1 to 1, and so on. Each model provides a `preprocess_input()` function that you can use to preprocess your images. These functions assume that the original pixel values range from 0 to 255, which is the case here: 

```
inputs=tf.keras.applications.resnet50.preprocess_input(images_resized)
```

Now we can use the pretrained model to make predictions: 

```
>>> Y_proba=model.predict(inputs)
>>> Y_proba.shape
(2, 1000)
```

As usual, the output `Y_proba` is a matrix with one row per image and one column per class (in this case, there are 1,000 classes). If you want to display the top _K_ predictions, including the class name and the estimated probability of each predicted class, use the `decode_predictions()` function. For each image, it returns an array containing the top _K_ predictions, where each prediction is represented as an array containing the class identifier,<sup>28</sup> its name, and the corresponding confidence score: 

```
top_K=tf.keras.applications.resnet50.decode_predictions(Y_proba, top=3)
forimage_indexinrange(len(images)):
print(f"Image #{image_index}")
forclass_id, name, y_probaintop_K[image_index]:
print(f"  {class_id} - {name:12s}{y_proba:.2%}")
```

The output looks like this: 

> 28 In the ImageNet dataset, each image is mapped to a word in the WordNet dataset: the class ID is just a WordNet ID. 

**Using Pretrained Models from Keras | 517** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

```
Image #0
  n03877845 - palace       54.69%
  n03781244 - monastery    24.72%
  n02825657 - bell_cote    18.55%
Image #1
  n04522168 - vase         32.66%
  n11939491 - daisy        17.81%
  n03530642 - honeycomb    12.06%
```

The correct classes are palace and dahlia, so the model is correct for the first image but wrong for the second. However, that’s because dahlia is not one of the 1,000 ImageNet classes. With that in mind, vase is a reasonable guess (perhaps the flower is in a vase?), and daisy is not a bad choice either, since dahlias and daisies are both from the same Compositae family. 

As you can see, it is very easy to create a pretty good image classifier using a pretrained model. As you saw in Table 14-3, many other vision models are available in `tf.keras.applications` , from lightweight and fast models to large and accurate ones. 

But what if you want to use an image classifier for classes of images that are not part of ImageNet? In that case, you may still benefit from the pretrained models by using them to perform transfer learning. 

## **Pretrained Models for Transfer Learning** 

If you want to build an image classifier but you do not have enough data to train it from scratch, then it is often a good idea to reuse the lower layers of a pretrained model, as we discussed in Chapter 11. For example, let’s train a model to classify pictures of flowers, reusing a pretrained Xception model. First, we’ll load the flowers dataset using TensorFlow Datasets (introduced in Chapter 13): 

```
importtensorflow_datasetsastfds
```

```
dataset, info=tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size=info.splits["train"].num_examples# 3670
class_names=info.features["label"].names# ["dandelion", "daisy", ...]
n_classes=info.features["label"].num_classes# 5
```

Note that you can get information about the dataset by setting `with_info=True` . Here, we get the dataset size and the names of the classes. Unfortunately, there is only a `"train"` dataset, no test set or validation set, so we need to split the training set. Let’s call `tfds.load()` again, but this time taking the first 10% of the dataset for testing, the next 15% for validation, and the remaining 75% for training: 

```
test_set_raw, valid_set_raw, train_set_raw=tfds.load(
"tf_flowers",
split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
as_supervised=True)
```

###### **518 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

All three datasets contain individual images. We need to batch them, but first we need to ensure they all have the same size, or batching will fail. We can use a `Resizing` layer for this. We must also call the `tf.keras.applications. xception.preprocess_input()` function to preprocess the images appropriately for the Xception model. Lastly, we’ll also shuffle the training set and use prefetching: 

```
batch_size=32
preprocess=tf.keras.Sequential([
tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])
train_set=train_set_raw.map(lambdaX, y: (preprocess(X), y))
train_set=train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)
valid_set=valid_set_raw.map(lambdaX, y: (preprocess(X), y)).batch(batch_size)
test_set=test_set_raw.map(lambdaX, y: (preprocess(X), y)).batch(batch_size)
```

Now each batch contains 32 images, all of them 224 × 224 pixels, with pixel values ranging from –1 to 1. Perfect! 

Since the dataset is not very large, a bit of data augmentation will certainly help. Let’s create a data augmentation model that we will embed in our final model. During training, it will randomly flip the images horizontally, rotate them a little bit, and tweak the contrast: 

```
data_augmentation=tf.keras.Sequential([
tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
tf.keras.layers.RandomRotation(factor=0.05, seed=42),
tf.keras.layers.RandomContrast(factor=0.2, seed=42)
```

```
])
```



The `tf.keras.preprocessing.image.ImageDataGenerator` class makes it easy to load images from disk and augment them in various ways: you can shift each image, rotate it, rescale it, flip it horizontally or vertically, shear it, or apply any transformation function you want to it. This is very convenient for simple projects. However, a tf.data pipeline is not much more complicated, and it’s generally faster. Moreover, if you have a GPU and you include the preprocessing or data augmentation layers inside your model, they will benefit from GPU acceleration during training. 

Next let’s load an Xception model, pretrained on ImageNet. We exclude the top of the network by setting `include_top=False` . This excludes the global average pooling layer and the dense output layer. We then add our own global average pooling layer (feeding it the output of the base model), followed by a dense output layer with one unit per class, using the softmax activation function. Finally, we wrap all this in a Keras `Model` : 

**Pretrained Models for Transfer Learning | 519** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

```
base_model=tf.keras.applications.xception.Xception(weights="imagenet",
include_top=False)
avg=tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output=tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model=tf.keras.Model(inputs=base_model.input, outputs=output)
```

As explained in Chapter 11, it’s usually a good idea to freeze the weights of the pretrained layers, at least at the beginning of training: 

```
forlayerinbase_model.layers:
layer.trainable=False
```



Since our model uses the base model’s layers directly, rather than the `base_model` object itself, setting `base_model.trainable=False` would have no effect. 

Finally, we can compile the model and start training: 

```
optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
metrics=["accuracy"])
history=model.fit(train_set, validation_data=valid_set, epochs=3)
```



If you are running in Colab, make sure the runtime is using a GPU: select Runtime → “Change runtime type”, choose “GPU” in the “Hardware accelerator” drop-down menu, then click Save. It’s possible to train the model without a GPU, but it will be terribly slow (minutes per epoch, as opposed to seconds). 

After training the model for a few epochs, its validation accuracy should reach a bit over 80% and then stop improving. This means that the top layers are now pretty well trained, and we are ready to unfreeze some of the base model’s top layers, then continue training. For example, let’s unfreeze layers 56 and above (that’s the start of residual unit 7 out of 14, as you can see if you list the layer names): 

```
forlayerinbase_model.layers[56:]:
layer.trainable=True
```

Don’t forget to compile the model whenever you freeze or unfreeze layers. Also make sure to use a much lower learning rate to avoid damaging the pretrained weights: 

```
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
metrics=["accuracy"])
history=model.fit(train_set, validation_data=valid_set, epochs=10)
```

This model should reach around 92% accuracy on the test set, in just a few minutes of training (with a GPU). If you tune the hyperparameters, lower the learning rate, 

###### **520 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

and train for quite a bit longer, you should be able to reach 95% to 97%. With that, you can start training amazing image classifiers on your own images and classes! But there’s more to computer vision than just classification. For example, what if you also want to know _where_ the flower is in a picture? Let’s look at this now. 

## **Classification and Localization** 

Localizing an object in a picture can be expressed as a regression task, as discussed in Chapter 10: to predict a bounding box around the object, a common approach is to predict the horizontal and vertical coordinates of the object’s center, as well as its height and width. This means we have four numbers to predict. It does not require much change to the model; we just need to add a second dense output layer with four units (typically on top of the global average pooling layer), and it can be trained using the MSE loss: 

```
base_model=tf.keras.applications.xception.Xception(weights="imagenet",
include_top=False)
avg=tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
class_output=tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
loc_output=tf.keras.layers.Dense(4)(avg)
model=tf.keras.Model(inputs=base_model.input,
outputs=[class_output, loc_output])
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss=["sparse_categorical_crossentropy", "mse"],
loss_weights=[0.8, 0.2],  # depends on what you care most about
optimizer=optimizer, metrics=["accuracy"])
```

But now we have a problem: the flowers dataset does not have bounding boxes around the flowers. So, we need to add them ourselves. This is often one of the hardest and most costly parts of a machine learning project: getting the labels. It’s a good idea to spend time looking for the right tools. To annotate images with bounding boxes, you may want to use an open source image labeling tool like VGG Image Annotator, LabelImg, OpenLabeler, or ImgLab, or perhaps a commercial tool like LabelBox or Supervisely. You may also want to consider crowdsourcing platforms such as Amazon Mechanical Turk if you have a very large number of images to annotate. However, it is quite a lot of work to set up a crowdsourcing platform, prepare the form to be sent to the workers, supervise them, and ensure that the quality of the bounding boxes they produce is good, so make sure it is worth the effort. Adriana Kovashka et al. wrote a very practical paper<sup>29</sup> about crowdsourcing in computer vision. I recommend you check it out, even if you do not plan to use crowdsourcing. If there are just a few hundred or a even a couple thousand images to label, and you don’t plan to do this frequently, it may be preferable to do it 

> 29 Adriana Kovashka et al., “Crowdsourcing in Computer Vision”, _Foundations and Trends in Computer Graphics and Vision_ 10, no. 3 (2014): 177–243. 

**Classification and Localization | 521** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 





<!-- Start of picture text -->
Label NO DB ! 3<br>Prediction Gt, a | Intersection |<br>Tf) 4 1 '<br>SO el<br><!-- End of picture text -->

## **Object Detection** 

The task of classifying and localizing multiple objects in an image is called _object detection_ . Until a few years ago, a common approach was to take a CNN that was trained to classify and locate a single object roughly centered in the image, then slide this CNN across the image and make predictions at each step. The CNN was generally trained to predict not only class probabilities and a bounding box, but also an _objectness score_ : this is the estimated probability that the image does indeed contain an object centered near the middle. This is a binary classification output; it can be produced by a dense output layer with a single unit, using the sigmoid activation function and trained using the binary cross-entropy loss. 



Instead of an objectness score, a “no-object” class was sometimes added, but in general this did not work as well: the questions “Is an object present?” and “What type of object is it?” are best answered separately. 

This sliding-CNN approach is illustrated in Figure 14-25. In this example, the image was chopped into a 5 × 7 grid, and we see a CNN—the thick black rectangle—sliding across all 3 × 3 regions and making predictions at each step. 



_Figure 14-25. Detecting multiple objects by sliding a CNN across the image_ 

**Object Detection | 523** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

In this figure, the CNN has already made predictions for three of these 3 × 3 regions: 

- When looking at the top-left 3 × 3 region (centered on the red-shaded grid cell located in the second row and second column), it detected the leftmost rose. Notice that the predicted bounding box exceeds the boundary of this 3 × 3 region. That’s absolutely fine: even though the CNN could not see the bottom part of the rose, it was able to make a reasonable guess as to where it might be. It also predicted class probabilities, giving a high probability to the “rose” class. Lastly, it predicted a fairly high objectness score, since the center of the bounding box lies within the central grid cell (in this figure, the objectness score is represented by the thickness of the bounding box). 

- When looking at the next 3 × 3 region, one grid cell to the right (centered on the shaded blue square), it did not detect any flower centered in that region, so it predicted a very low objectness score; therefore, the predicted bounding box and class probabilities can safely be ignored. You can see that the predicted bounding box was no good anyway. 

- finally, when looking at the next 3 × 3 region, again one grid cell to the right (centered on the shaded green cell), it detected the rose at the top, although not perfectly: this rose is not well centered within this region, so the predicted objectness score was not very high. 

You can imagine how sliding the CNN across the whole image would give you a total of 15 predicted bounding boxes, organized in a 3 × 5 grid, with each bounding box accompanied by its estimated class probabilities and objectness score. Since objects can have varying sizes, you may then want to slide the CNN again across larger 4 × 4 regions as well, to get even more bounding boxes. 

This technique is fairly straightforward, but as you can see it will often detect the same object multiple times, at slightly different positions. Some postprocessing is needed to get rid of all the unnecessary bounding boxes. A common approach for this is called _non-max suppression_ . Here’s how it works: 

**1.** First, get rid of all the bounding boxes for which the objectness score is below some threshold: since the CNN believes there’s no object at that location, the bounding box is useless. 

**2.** Find the remaining bounding box with the highest objectness score, and get rid of all the other remaining bounding boxes that overlap a lot with it (e.g., with an IoU greater than 60%). For example, in Figure 14-25, the bounding box with the max objectness score is the thick bounding box over the leftmost rose. The other bounding box that touches this same rose overlaps a lot with the max bounding box, so we will get rid of it (although in this example it would already have been removed in the previous step). 

###### **524 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

**3.** Repeat step 2 until there are no more bounding boxes to get rid of. 

This simple approach to object detection works pretty well, but it requires running the CNN many times (15 times in this example), so it is quite slow. Fortunately, there is a much faster way to slide a CNN across an image: using a _fully convolutional network_ (FCN). 

#### **Fully Convolutional Networks** 

The idea of FCNs was first introduced in a 2015 paper<sup>30</sup> by Jonathan Long et al., for semantic segmentation (the task of classifying every pixel in an image according to the class of the object it belongs to). The authors pointed out that you could replace the dense layers at the top of a CNN with convolutional layers. To understand this, let’s look at an example: suppose a dense layer with 200 neurons sits on top of a convolutional layer that outputs 100 feature maps, each of size 7 × 7 (this is the feature map size, not the kernel size). Each neuron will compute a weighted sum of all 100 × 7 × 7 activations from the convolutional layer (plus a bias term). Now let’s see what happens if we replace the dense layer with a convolutional layer using 200 filters, each of size 7 × 7, and with `"valid"` padding. This layer will output 200 feature maps, each 1 × 1 (since the kernel is exactly the size of the input feature maps and we are using `"valid"` padding). In other words, it will output 200 numbers, just like the dense layer did; and if you look closely at the computations performed by a convolutional layer, you will notice that these numbers will be precisely the same as those the dense layer produced. The only difference is that the dense layer’s output was a tensor of shape [ _batch size_ , 200], while the convolutional layer will output a tensor of shape [ _batch size_ , 1, 1, 200]. 



To convert a dense layer to a convolutional layer, the number of filters in the convolutional layer must be equal to the number of units in the dense layer, the filter size must be equal to the size of the input feature maps, and you must use `"valid"` padding. The stride may be set to 1 or more, as you will see shortly. 

Why is this important? Well, while a dense layer expects a specific input size (since it has one weight per input feature), a convolutional layer will happily process images of any size<sup>31</sup> (however, it does expect its inputs to have a specific number of channels, since each kernel contains a different set of weights for each input channel). Since 

> 30 Jonathan Long et al., “Fully Convolutional Networks for Semantic Segmentation”, _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (2015): 3431–3440. 

> 31 There is one small exception: a convolutional layer using `"valid"` padding will complain if the input size is smaller than the kernel size. 

**Object Detection | 525** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

an FCN contains only convolutional layers (and pooling layers, which have the same property), it can be trained and executed on images of any size! 

For example, suppose we’d already trained a CNN for flower classification and locali‐ zation. It was trained on 224 × 224 images, and it outputs 10 numbers: 

- Outputs 0 to 4 are sent through the softmax activation function, and this gives the class probabilities (one per class). 

- Output 5 is sent through the sigmoid activation function, and this gives the objectness score. 

- Outputs 6 and 7 represent the bounding box’s center coordinates; they also go through a sigmoid activation function to ensure they range from 0 to 1. 

- Lastly, outputs 8 and 9 represent the bounding box’s height and width; they do not go through any activation function to allow the bounding boxes to extend beyond the borders of the image. 

We can now convert the CNN’s dense layers to convolutional layers. In fact, we don’t even need to retrain it; we can just copy the weights from the dense layers to the convolutional layers! Alternatively, we could have converted the CNN into an FCN before training. 

Now suppose the last convolutional layer before the output layer (also called the bottleneck layer) outputs 7 × 7 feature maps when the network is fed a 224 × 224 image (see the left side of Figure 14-26). If we feed the FCN a 448 × 448 image (see the right side of Figure 14-26), the bottleneck layer will now output 14 × 14 feature maps.<sup>32</sup> Since the dense output layer was replaced by a convolutional layer using 10 filters of size 7 × 7, with `"valid"` padding and stride 1, the output will be composed of 10 features maps, each of size 8 × 8 (since 14 – 7 + 1 = 8). In other words, the FCN will process the whole image only once, and it will output an 8 × 8 grid where each cell contains 10 numbers (5 class probabilities, 1 objectness score, and 4 bounding box coordinates). It’s exactly like taking the original CNN and sliding it across the image using 8 steps per row and 8 steps per column. To visualize this, imagine chopping the original image into a 14 × 14 grid, then sliding a 7 × 7 window across this grid; there will be 8 × 8 = 64 possible locations for the window, hence 8 × 8 predictions. However, the FCN approach is _much_ more efficient, since the network only looks at the image once. In fact, _You Only Look Once_ (YOLO) is the name of a very popular object detection architecture, which we’ll look at next. 

> 32 This assumes we used only `"same"` padding in the network: `"valid"` padding would reduce the size of the feature maps. Moreover, 448 can be neatly divided by 2 several times until we reach 7, without any rounding error. If any layer uses a different stride than 1 or 2, then there may be some rounding error, so again the feature maps may end up being smaller. 

**526 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 



<!-- Start of picture text -->
Feature maps —[Y_ Feature maps<br>Convolution Convolution<br>10, 7*7+1(V) 10, 7*7+1(V)<br>ff 7x7 / if 14x14<br>——/ Feature maps 7 Feature maps<br>SD a<br>CS J)<br>P P ’<br>a Pi<br>a my 224x224 ‘<br>a * 7” | mage E — _<br>; ; —<br>= A SS ee<br>» Behe) mage<br><!-- End of picture text -->

- It outputs two bounding boxes for each grid cell (instead of just one), which allows the model to handle cases where two objects are so close to each other that their bounding box centers lie within the same cell. Each bounding box also comes with its own objectness score. 

- YOLO also outputs a class probability distribution for each grid cell, predicting 20 class probabilities per grid cell since YOLO was trained on the PASCAL VOC dataset, which contains 20 classes. This produces a coarse _class probability map_ . Note that the model predicts one class probability distribution per grid cell, not per bounding box. However, it’s possible to estimate class probabilities for each bounding box during postprocessing, by measuring how well each bounding box matches each class in the class probability map. For example, imagine a picture of a person standing in front of a car. There will be two bounding boxes: one large horizontal one for the car, and a smaller vertical one for the person. These bounding boxes may have their centers within the same grid cell. So how can we tell which class should be assigned to each bounding box? Well, the class probability map will contain a large region where the “car” class is dominant, and inside it there will be a smaller region where the “person” class is dominant. Hopefully, the car’s bounding box will roughly match the “car” region, while the person’s bounding box will roughly match the “person” region: this will allow the correct class to be assigned to each bounding box. 

YOLO was originally developed using Darknet, an open source deep learning frame‐ work initially developed in C by Joseph Redmon, but it was soon ported to Tensor‐ Flow, Keras, PyTorch, and more. It was continuously improved over the years, with YOLOv2, YOLOv3, and YOLO9000 (again by Joseph Redmon et al.), YOLOv4 (by Alexey Bochkovskiy et al.), YOLOv5 (by Glenn Jocher), and PP-YOLO (by Xiang Long et al.). 

Each version brought some impressive improvements in speed and accuracy, using a variety of techniques; for example, YOLOv3 boosted accuracy in part thanks to _anchor priors_ , exploiting the fact that some bounding box shapes are more likely than others, depending on the class (e.g., people tend to have vertical bounding boxes, while cars usually don’t). They also increased the number of bounding boxes per grid cell, they trained on different datasets with many more classes (up to 9,000 classes organized in a hierarchy in the case of YOLO9000), they added skip connections to recover some of the spatial resolution that is lost in the CNN (we will discuss this shortly, when we look at semantic segmentation), and much more. There are many variants of these models too, such as YOLOv4-tiny, which is optimized to be trained on less powerful machines and which can run extremely fast (at over 1,000 frames per second!), but with a slightly lower _mean average precision_ (mAP). 

**528 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

##### **Mean Average Precision** 

A very common metric used in object detection tasks is the mean average precision. “Mean average” sounds a bit redundant, doesn’t it? To understand this metric, let’s go back to two classification metrics we discussed in Chapter 3: precision and recall. Remember the trade-off: the higher the recall, the lower the precision. You can visualize this in a precision/recall curve (see Figure 3-6). To summarize this curve into a single number, we could compute its area under the curve (AUC). But note that the precision/recall curve may contain a few sections where precision actually goes up when recall increases, especially at low recall values (you can see this at the top left of Figure 3-6). This is one of the motivations for the mAP metric. 

Suppose the classifier has 90% precision at 10% recall, but 96% precision at 20% recall. There’s really no trade-off here: it simply makes more sense to use the classifier at 20% recall rather than at 10% recall, as you will get both higher recall and higher precision. So instead of looking at the precision _at_ 10% recall, we should really be looking at the _maximum_ precision that the classifier can offer with _at least_ 10% recall. It would be 96%, not 90%. Therefore, one way to get a fair idea of the model’s performance is to compute the maximum precision you can get with at least 0% recall, then 10% recall, 20%, and so on up to 100%, and then calculate the mean of these maximum precisions. This is called the _average precision_ (AP) metric. Now when there are more than two classes, we can compute the AP for each class, and then compute the mean AP (mAP). That’s it! 

In an object detection system, there is an additional level of complexity: what if the system detected the correct class, but at the wrong location (i.e., the bounding box is completely off)? Surely we should not count this as a positive prediction. One approach is to define an IoU threshold: for example, we may consider that a prediction is correct only if the IoU is greater than, say, 0.5, and the predicted class is correct. The corresponding mAP is generally noted mAP@0.5 (or mAP@50%, or sometimes just AP50). In some competitions (such as the PASCAL VOC challenge), this is what is done. In others (such as the COCO competition), the mAP is computed for different IoU thresholds (0.50, 0.55, 0.60, …, 0.95), and the final metric is the mean of all these mAPs (noted mAP@[.50:.95] or mAP@[.50:0.05:.95]). Yes, that’s a mean mean average. 

**Object Detection | 529** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

Many object detection models are available on TensorFlow Hub, often with pre‐ trained weights, such as YOLOv5,<sup>34</sup> SSD,<sup>35</sup> Faster R-CNN,<sup>36</sup> and EfficentDet.<sup>37</sup> 

SSD and EfficientDet are “look once” detection models, similar to YOLO. Efficient‐ Det is based on the EfficientNet convolutional architecture. Faster R-CNN is more complex: the image first goes through a CNN, then the output is passed to a _region proposal network_ (RPN) that proposes bounding boxes that are most likely to contain an object; a classifier is then run for each bounding box, based on the cropped output of the CNN. The best place to start using these models is TensorFlow Hub’s excellent object detection tutorial. 

So far, we’ve only considered detecting objects in single images. But what about videos? Objects must not only be detected in each frame, they must also be tracked over time. Let’s take a quick look at object tracking now. 

## **Object Tracking** 

Object tracking is a challenging task: objects move, they may grow or shrink as they get closer to or further away from the camera, their appearance may change as they turn around or move to different lighting conditions or backgrounds, they may be temporarily occluded by other objects, and so on. 

One of the most popular object tracking systems is DeepSORT.<sup>38</sup> It is based on a combination of classical algorithms and deep learning: 

- It uses _Kalman filters_ to estimate the most likely current position of an object given prior detections, and assuming that objects tend to move at a constant speed. 

- It uses a deep learning model to measure the resemblance between new detec‐ tions and existing tracked objects. 

> 34 You can find YOLOv3, YOLOv4, and their tiny variants in the TensorFlow Models project at _https://homl.info/ yolotf_ . 

- 35 Wei Liu et al., “SSD: Single Shot Multibox Detector”, _Proceedings of the 14th European Conference on Computer Vision_ 1 (2016): 21–37. 

> 36 Shaoqing Ren et al., “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks”, _Proceedings of the 28th International Conference on Neural Information Processing Systems_ 1 (2015): 91–99. 

- 37 Mingxing Tan et al., “EfficientDet: Scalable and Efficient Object Detection”, arXiv preprint arXiv:1911.09070 (2019). 

- 38 Nicolai Wojke et al., “Simple Online and Realtime Tracking with a Deep Association Metric”, arXiv preprint arXiv:1703.07402 (2017). 

###### **530 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

- Lastly, it uses the _Hungarian algorithm_ to map new detections to existing tracked objects (or to new tracked objects): this algorithm efficiently finds the combination of mappings that minimizes the distance between the detections and the predicted positions of tracked objects, while also minimizing the appear‐ ance discrepancy. 

For example, imagine a red ball that just bounced off a blue ball traveling in the opposite direction. Based on the previous positions of the balls, the Kalman filter will predict that the balls will go through each other: indeed, it assumes that objects move at a constant speed, so it will not expect the bounce. If the Hungarian algorithm only considered positions, then it would happily map the new detections to the wrong balls, as if they had just gone through each other and swapped colors. But thanks to the resemblance measure, the Hungarian algorithm will notice the problem. Assuming the balls are not too similar, the algorithm will map the new detections to the correct balls. 



There are a few DeepSORT implementations available on GitHub, including a TensorFlow implementation of YOLOv4 + DeepSORT: _https://github.com/theAIGuysCode/yolov4-deepsort_ . 

So far we have located objects using bounding boxes. This is often sufficient, but sometimes you need to locate objects with much more precision—for example, to remove the background behind a person during a videoconference call. Let’s see how to go down to the pixel level. 

## **Semantic Segmentation** 

In _semantic segmentation_ , each pixel is classified according to the class of the object it belongs to (e.g., road, car, pedestrian, building, etc.), as shown in Figure 14-27. Note that different objects of the same class are _not_ distinguished. For example, all the bicycles on the right side of the segmented image end up as one big lump of pixels. The main difficulty in this task is that when images go through a regular CNN, they gradually lose their spatial resolution (due to the layers with strides greater than 1); so, a regular CNN may end up knowing that there’s a person somewhere in the bottom left of the image, but it will not be much more precise than that. 

**Semantic Segmentation | 531** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 



<!-- Start of picture text -->
Sin " PF) || | | Buitaings<br>r s Lee: f ay A People ars<br>(2<br> 4 { Pog<br>LiAys ‘Seslt LE Bicycles<br>ge eee ee Road<br><!-- End of picture text -->





<!-- Start of picture text -->
[—7_Z_Z._Z_Z_Z_/<br>17 _7 _Z _Z _Z_Z<br>LITT 10<br>/ 77 _7 _Z _Z_Z J utput<br>/ 77-7<br>i travStride=2<br>KYT B® T_Z_Z_Z_Z_Z_Z_7_7 7 /<br>NL ZZ _2Z_2Z_2Z_Z_Z_/<br>YTV \ 7 7 SZ _Z_Z_2_Z<br>117 J<br>1 nhC 7 _7_2_/7_2Z/Za _Zy _Z/<br>V77 VY 7 77 /7/<br>UY,Kernel size=3 4<br><!-- End of picture text -->



<!-- Start of picture text -->
Feature map Skip connection PEE Se:+ #3 =<br>EERE BEER CO +<br>EERE BEEBE EEE Eee eee<br>EERE EERE BEER BERR Ree<br>wees LLL... ef EEeSEEEe COO eer<br>EERE EEREE EERE BERR<br>Coo =2 Oooo*2 oo x16 GOCCCCeee<br>tttTT ttt «}« 4 LLL 7 eee<br>LtteT itt ty ! '' BEEBE '' EEREPET TTTeeeTTTeee<br>Downsampling ‘------- Upsampling -------! (34959545395755<br><!-- End of picture text -->

can recognize an object after it has seen it just once), predicting the next frames in a video, combining text and image tasks, and more. 

Now on to the next chapter, where we will look at how to process sequential data such as time series using recurrent neural networks and convolutional neural networks. 

## **Exercises** 

**1.** What are the advantages of a CNN over a fully connected DNN for image classification? 

**2.** Consider a CNN composed of three convolutional layers, each with 3 × 3 kernels, a stride of 2, and `"same"` padding. The lowest layer outputs 100 feature maps, the middle one outputs 200, and the top one outputs 400. The input images are RGB images of 200 × 300 pixels: 

   - **a.** What is the total number of parameters in the CNN? 

   - **b.** If we are using 32-bit floats, at least how much RAM will this network require when making a prediction for a single instance? 

   - **c.** What about when training on a mini-batch of 50 images? 

**3.** If your GPU runs out of memory while training a CNN, what are five things you could try to solve the problem? 

**4.** Why would you want to add a max pooling layer rather than a convolutional layer with the same stride? 

**5.** When would you want to add a local response normalization layer? 

**6.** Can you name the main innovations in AlexNet, as compared to LeNet-5? What about the main innovations in GoogLeNet, ResNet, SENet, Xception, and EfficientNet? 

**7.** What is a fully convolutional network? How can you convert a dense layer into a convolutional layer? 

**8.** What is the main technical difficulty of semantic segmentation? 

**9.** Build your own CNN from scratch and try to achieve the highest possible accuracy on MNIST. 

**10.** Use transfer learning for large image classification, going through these steps: 

   - **a.** Create a training set containing at least 100 images per class. For example, you could classify your own pictures based on the location (beach, mountain, city, etc.), or alternatively you can use an existing dataset (e.g., from TensorFlow Datasets). 

   - **b.** Split it into a training set, a validation set, and a test set. 

**Exercises | 535** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

   - **c.** Build the input pipeline, apply the appropriate preprocessing operations, and optionally add data augmentation. 

   - **d.** Fine-tune a pretrained model on this dataset. 

**11.** Go through TensorFlow’s Style Transfer tutorial. This is a fun way to generate art using deep learning. 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

**536 | Chapter 14: Deep Computer Vision Using Convolutional Neural Networks** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:09. 

