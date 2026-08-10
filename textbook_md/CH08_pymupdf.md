# **CHAPTER 8 Dimensionality Reduction** 

Many machine learning problems involve thousands or even millions of features for each training instance. Not only do all these features make training extremely slow, but they can also make it much harder to find a good solution, as you will see. This problem is often referred to as the _curse of dimensionality_ . 

Fortunately, in real-world problems, it is often possible to reduce the number of features considerably, turning an intractable problem into a tractable one. For exam‐ ple, consider the MNIST images (introduced in Chapter 3): the pixels on the image borders are almost always white, so you could completely drop these pixels from the training set without losing much information. As we saw in the previous chapter, (Figure 7-6) confirms that these pixels are utterly unimportant for the classification task. Additionally, two neighboring pixels are often highly correlated: if you merge them into a single pixel (e.g., by taking the mean of the two pixel intensities), you will not lose much information. 



Reducing dimensionality does cause some information loss, just like compressing an image to JPEG can degrade its quality, so even though it will speed up training, it may make your system perform slightly worse. It also makes your pipelines a bit more complex and thus harder to maintain. Therefore, I recommend you first try to train your system with the original data before considering using dimensionality reduction. In some cases, reducing the dimension‐ ality of the training data may filter out some noise and unnecessary details and thus result in higher performance, but in general it won’t; it will just speed up training. 

**237** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 



10,000-dimensional unit hypercube, this probability is greater than 99.999999%. Most points in a high-dimensional hypercube are very close to the border.<sup>3</sup> 

Here is a more troublesome difference: if you pick two points randomly in a unit square, the distance between these two points will be, on average, roughly 0.52. If you pick two random points in a 3D unit cube, the average distance will be roughly 0.66. But what about two points picked randomly in a 1,000,000-dimensional unit hypercube? The average distance, believe it or not, will be about 408.25 (roughly 

1, 000, 000 6)! This is counterintuitive: how can two points be so far apart when they both lie within the same unit hypercube? Well, there’s just plenty of space in high dimensions. As a result, high-dimensional datasets are at risk of being very sparse: most training instances are likely to be far away from each other. This also means that a new instance will likely be far away from any training instance, making predictions much less reliable than in lower dimensions, since they will be based on much larger extrapolations. In short, the more dimensions the training set has, the greater the risk of overfitting it. 

In theory, one solution to the curse of dimensionality could be to increase the size of the training set to reach a sufficient density of training instances. Unfortunately, in practice, the number of training instances required to reach a given density grows exponentially with the number of dimensions. With just 100 features—signifi‐ cantly fewer than in the MNIST problem—all ranging from 0 to 1, you would need more training instances than atoms in the observable universe in order for training instances to be within 0.1 of each other on average, assuming they were spread out uniformly across all dimensions. 

## **Main Approaches for Dimensionality Reduction** 

Before we dive into specific dimensionality reduction algorithms, let’s take a look at the two main approaches to reducing dimensionality: projection and manifold learning. 

### **Projection** 

In most real-world problems, training instances are _not_ spread out uniformly across all dimensions. Many features are almost constant, while others are highly correlated (as discussed earlier for MNIST). As a result, all training instances lie within (or close to) a much lower-dimensional _subspace_ of the high-dimensional space. This sounds very abstract, so let’s look at an example. In Figure 8-2 you can see a 3D dataset represented by small spheres. 

> 3 Fun fact: anyone you know is probably an extremist in at least one dimension (e.g., how much sugar they put in their coffee), if you consider enough dimensions. 

**Main Approaches for Dimensionality Reduction | 239** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 



<!-- Start of picture text -->
pa.<br>yy<br><!-- End of picture text -->



<!-- Start of picture text -->
Too<br>OseoD<br>ofoeyy| pb<br><!-- End of picture text -->



<!-- Start of picture text -->
aSCoots yee]: x<br>ree [ il —5<br>- - x 5 10 0 x2<br><!-- End of picture text -->



The Swiss roll is an example of a 2D _manifold_ . Put simply, a 2D manifold is a 2D shape that can be bent and twisted in a higher-dimensional space. More generally, a _d_ -dimensional manifold is a part of an _n_ -dimensional space (where _d_ < _n_ ) that locally resembles a _d_ -dimensional hyperplane. In the case of the Swiss roll, _d_ = 2 and _n_ = 3: it locally resembles a 2D plane, but it is rolled in the third dimension. 

Many dimensionality reduction algorithms work by modeling the manifold on which the training instances lie; this is called _manifold learning_ . It relies on the _manifold assumption_ , also called the _manifold hypothesis_ , which holds that most real-world high-dimensional datasets lie close to a much lower-dimensional manifold. This assumption is very often empirically observed. 

Once again, think about the MNIST dataset: all handwritten digit images have some similarities. They are made of connected lines, the borders are white, and they are more or less centered. If you randomly generated images, only a ridiculously tiny fraction of them would look like handwritten digits. In other words, the degrees of freedom available to you if you try to create a digit image are dramatically lower than the degrees of freedom you have if you are allowed to generate any image you want. These constraints tend to squeeze the dataset into a lower-dimensional manifold. 

The manifold assumption is often accompanied by another implicit assumption: that the task at hand (e.g., classification or regression) will be simpler if expressed in the lower-dimensional space of the manifold. For example, in the top row of Figure 8-6 the Swiss roll is split into two classes: in the 3D space (on the left) the decision boundary would be fairly complex, but in the 2D unrolled manifold space (on the right) the decision boundary is a straight line. 

However, this implicit assumption does not always hold. For example, in the bottom row of Figure 8-6, the decision boundary is located at _x_ 1 = 5. This decision boundary looks very simple in the original 3D space (a vertical plane), but it looks more complex in the unrolled manifold (a collection of four independent line segments). 

In short, reducing the dimensionality of your training set before training a model will usually speed up training, but it may not always lead to a better or simpler solution; it all depends on the dataset. 

Hopefully you now have a good sense of what the curse of dimensionality is and how dimensionality reduction algorithms can fight it, especially when the manifold assumption holds. The rest of this chapter will go through some of the most popular algorithms for dimensionality reduction. 

**242 | Chapter 8: Dimensionality Reduction** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 



<!-- Start of picture text -->
15<br>* | spss sty att | ate deal<br>“atlLU ang~w 10<br>reer | Folin 10 7 sig eae<br>Cia a<br>ox 5 ol1 0 Peer yt<br>20 ra<br>ee Yolk val ate 10<br>f<br>ae Bec Be<br>oy<br>er a<br>0 x ° —<br>21 i<br><!-- End of picture text -->



<!-- Start of picture text -->
- Sen aoe<br>~ .<br>x: 7 : _ Fa | | |<br>-0.5 Ae<br>- P| | | | Pee Pery<br>-10 05 00 O5 10 -2.0 -1.5 -1.0 -05 00 05 10 15 2.0<br>X1 21<br><!-- End of picture text -->



For each principal component, PCA finds a zero-centered unit vector pointing in the direction of the PC. Since two opposing unit vectors lie on the same axis, the direction of the unit vectors returned by PCA is not stable: if you perturb the training set slightly and run PCA again, the unit vectors may point in the oppo‐ site direction as the original vectors. However, they will generally still lie on the same axes. In some cases, a pair of unit vectors may even rotate or swap (if the variances along these two axes are very close), but the plane they define will generally remain the same. 

So how can you find the principal components of a training set? Luckily, there is a standard matrix factorization technique called _singular value decomposition_ (SVD) that can decompose the training set matrix **X** into the matrix multiplication of three matrices **U Σ V**<sup>⊺</sup> , where **V** contains the unit vectors that define all the principal components that you are looking for, in the correct order, as shown in Equation 8-1. 

_Equation 8-1. Principal components matrix_ 



The following Python code uses NumPy’s `svd()` function to obtain all the principal components of the 3D training set represented in Figure 8-2, then it extracts the two unit vectors that define the first two PCs: 

```
importnumpyasnp
```

```
X= [...]  # create a small 3D dataset
X_centered=X-X.mean(axis=0)
U, s, Vt=np.linalg.svd(X_centered)
c1=Vt[0]
c2=Vt[1]
```



PCA assumes that the dataset is centered around the origin. As you will see, Scikit-Learn’s PCA classes take care of centering the data for you. If you implement PCA yourself (as in the preceding example), or if you use other libraries, don’t forget to center the data first. 

### **Projecting Down to d Dimensions** 

Once you have identified all the principal components, you can reduce the dimen‐ sionality of the dataset down to _d_ dimensions by projecting it onto the hyperplane defined by the first _d_ principal components. Selecting this hyperplane ensures that 

**PCA | 245** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 

the projection will preserve as much variance as possible. For example, in Figure 8-2 the 3D dataset is projected down to the 2D plane defined by the first two principal components, preserving a large part of the dataset’s variance. As a result, the 2D projection looks very much like the original 3D dataset. 

To project the training set onto the hyperplane and obtain a reduced dataset **X** _d_ -proj of dimensionality _d_ , compute the matrix multiplication of the training set matrix **X** by the matrix **W** _d_ , defined as the matrix containing the first _d_ columns of **V** , as shown in Equation 8-2. 

_Equation 8-2. Projecting the training set down to d dimensions_ 

Xd ‐proj = XWd 

The following Python code projects the training set onto the plane defined by the first two principal components: 

```
W2=Vt[:2].T
X2D=X_centered@W2
```

There you have it! You now know how to reduce the dimensionality of any dataset by projecting it down to any number of dimensions, while preserving as much variance as possible. 

### **Using Scikit-Learn** 

Scikit-Learn’s `PCA` class uses SVD to implement PCA, just like we did earlier in this chapter. The following code applies PCA to reduce the dimensionality of the dataset down to two dimensions (note that it automatically takes care of centering the data): 

```
fromsklearn.decompositionimportPCA
```

```
pca=PCA(n_components=2)
X2D=pca.fit_transform(X)
```

After fitting the `PCA` transformer to the dataset, its `components_` attribute holds the transpose of **W** _d_ : it contains one row for each of the first _d_ principal components. 

### **Explained Variance Ratio** 

Another useful piece of information is the _explained variance ratio_ of each princi‐ pal component, available via the `explained_variance_ratio_` variable. The ratio indicates the proportion of the dataset’s variance that lies along each principal com‐ ponent. For example, let’s look at the explained variance ratios of the first two components of the 3D dataset represented in Figure 8-2: 

```
>>> pca.explained_variance_ratio_
array([0.7578477 , 0.15186921])
```

**246 | Chapter 8: Dimensionality Reduction** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 

This output tells us that about 76% of the dataset’s variance lies along the first PC, and about 15% lies along the second PC. This leaves about 9% for the third PC, so it is reasonable to assume that the third PC probably carries little information. 

### **Choosing the Right Number of Dimensions** 

Instead of arbitrarily choosing the number of dimensions to reduce down to, it is simpler to choose the number of dimensions that add up to a sufficiently large portion of the variance—say, 95% (An exception to this rule, of course, is if you are reducing dimensionality for data visualization, in which case you will want to reduce the dimensionality down to 2 or 3). 

The following code loads and splits the MNIST dataset (introduced in Chapter 3) and performs PCA without reducing dimensionality, then computes the minimum number of dimensions required to preserve 95% of the training set’s variance: 

```
fromsklearn.datasetsimportfetch_openml
```

```
mnist=fetch_openml('mnist_784', as_frame=False)
X_train, y_train=mnist.data[:60_000], mnist.target[:60_000]
X_test, y_test=mnist.data[60_000:], mnist.target[60_000:]
```

```
pca=PCA()
pca.fit(X_train)
cumsum=np.cumsum(pca.explained_variance_ratio_)
d=np.argmax(cumsum>=0.95) +1# d equals 154
```

You could then set `n_components=d` and run PCA again, but there’s a better option. Instead of specifying the number of principal components you want to preserve, you can set `n_components` to be a float between 0.0 and 1.0, indicating the ratio of variance you wish to preserve: 

```
pca=PCA(n_components=0.95)
X_reduced=pca.fit_transform(X_train)
```

The actual number of components is determined during training, and it is stored in the `n_components_` attribute: 

```
>>> pca.n_components_
154
```

Yet another option is to plot the explained variance as a function of the number of dimensions (simply plot `cumsum` ; see Figure 8-8). There will usually be an elbow in the curve, where the explained variance stops growing fast. In this case, you can see that reducing the dimensionality down to about 100 dimensions wouldn’t lose too much explained variance. 

**PCA | 247** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 



<!-- Start of picture text -->
1.0<br>w 08 \<br>< Elbow :<br>& :<br>pao 0.6 ::<br>><br>oe} :<br>v :<br>‘S£Q0.4 *:<br>9.2x::<br>0.0 i<br>0 50 100 150 200 250 300 350 400<br>Dimensions<br><!-- End of picture text -->



<!-- Start of picture text -->
YAQG3/ YAQ3B/<br>5TIAPZ 5TIAP<br>1947031491703<br>OFXIYOFAIA<br>Servei seers<br><!-- End of picture text -->

The equation for the inverse transformation is shown in Equation 8-3. 

_Equation 8-3. PCA inverse transformation, back to the original number of dimensions_ 

X recovered = Xd ‐proj Wd⊺ 

### **Randomized PCA** 

If you set the `svd_solver` hyperparameter to `"randomized"` , Scikit-Learn uses a stochastic algorithm called _randomized PCA_ that quickly finds an approximation of the first _d_ principal components. Its computational complexity is _O_ ( _m_ × _d_<sup>2</sup> ) + _O_ ( _d_<sup>3</sup> ), instead of _O_ ( _m_ × _n_<sup>2</sup> ) + _O_ ( _n_<sup>3</sup> ) for the full SVD approach, so it is dramatically faster than full SVD when _d_ is much smaller than _n_ : 

```
rnd_pca=PCA(n_components=154, svd_solver="randomized", random_state=42)
X_reduced=rnd_pca.fit_transform(X_train)
```



By default, `svd_solver` is actually set to `"auto"` : Scikit-Learn auto‐ matically uses the randomized PCA algorithm if max( _m_ , _n_ ) > 500 and `n_components` is an integer smaller than 80% of min( _m_ , _n_ ), or else it uses the full SVD approach. So the preceding code would use the randomized PCA algorithm even if you removed the `svd_solver="randomized"` argument, since 154 < 0.8 × 784. If you want to force Scikit-Learn to use full SVD for a slightly more precise result, you can set the `svd_solver` hyperparameter to `"full"` . 

### **Incremental PCA** 

One problem with the preceding implementations of PCA is that they require the whole training set to fit in memory in order for the algorithm to run. Fortunately, _incremental PCA_ (IPCA) algorithms have been developed that allow you to split the training set into mini-batches and feed these in one mini-batch at a time. This is useful for large training sets and for applying PCA online (i.e., on the fly, as new instances arrive). 

The following code splits the MNIST training set into 100 mini-batches (using Num‐ Py’s `array_split()` function) and feeds them to Scikit-Learn’s `IncrementalPCA` class<sup>5</sup> to reduce the dimensionality of the MNIST dataset down to 154 dimensions, just like 

> 5 Scikit-Learn uses the algorithm described in David A. Ross et al., “Incremental Learning for Robust Visual Tracking”, _International Journal of Computer Vision_ 77, no. 1–3 (2008): 125–141. 

**250 | Chapter 8: Dimensionality Reduction** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 

before. Note that you must call the `partial_fit()` method with each mini-batch, rather than the `fit()` method with the whole training set: 

```
fromsklearn.decompositionimportIncrementalPCA
```

```
n_batches=100
inc_pca=IncrementalPCA(n_components=154)
forX_batchinnp.array_split(X_train, n_batches):
inc_pca.partial_fit(X_batch)
```

```
X_reduced=inc_pca.transform(X_train)
```

Alternatively, you can use NumPy’s `memmap` class, which allows you to manipulate a large array stored in a binary file on disk as if it were entirely in memory; the class loads only the data it needs in memory, when it needs it. To demonstrate this, let’s first create a memory-mapped (memmap) file and copy the MNIST training set to it, then call `flush()` to ensure that any data still in the cache gets saved to disk. In real life, `X_train` would typically not fit in memory, so you would load it chunk by chunk and save each chunk to the right part of the memmap array: 

```
filename="my_mnist.mmap"
```

```
X_mmap=np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)
X_mmap[:] =X_train# could be a loop instead, saving the data chunk by chunk
X_mmap.flush()
```

Next, we can load the memmap file and use it like a regular NumPy array. Let’s use the `IncrementalPCA` class to reduce its dimensionality. Since this algorithm uses only a small part of the array at any given time, memory usage remains under control. This makes it possible to call the usual `fit()` method instead of `partial_fit()` , which is quite convenient: 

```
X_mmap=np.memmap(filename, dtype="float32", mode="readonly").reshape(-1, 784)
batch_size=X_mmap.shape[0] //n_batches
inc_pca=IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mmap)
```



Only the raw binary data is saved to disk, so you need to specify the data type and shape of the array when you load it. If you omit the shape, `np.memmap()` returns a 1D array. 

For very high-dimensional datasets, PCA can be too slow. As you saw earlier, even if you use randomized PCA its computational complexity is still _O_ ( _m_ × _d_<sup>2</sup> ) + _O_ ( _d_<sup>3</sup> ), so the target number of dimensions _d_ must not be too large. If you are dealing with a dataset with tens of thousands of features or more (e.g., images), then training may become much too slow: in this case, you should consider using random projection instead. 

**PCA | 251** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 

## **Random Projection** 

As its name suggests, the random projection algorithm projects the data to a lowerdimensional space using a random linear projection. This may sound crazy, but it turns out that such a random projection is actually very likely to preserve distances fairly well, as was demonstrated mathematically by William B. Johnson and Joram Lindenstrauss in a famous lemma. So, two similar instances will remain similar after the projection, and two very different instances will remain very different. 

Obviously, the more dimensions you drop, the more information is lost, and the more distances get distorted. So how can you choose the optimal number of dimen‐ sions? Well, Johnson and Lindenstrauss came up with an equation that determines the minimum number of dimensions to preserve in order to ensure—with high prob‐ ability—that distances won’t change by more than a given tolerance. For example, if you have a dataset containing _m_ = 5,000 instances with _n_ = 20,000 features each, and you don’t want the squared distance between any two instances to change by more than _ε_ = 10%,<sup>6</sup> then you should project the data down to _d_ dimensions, with _d_ ≥ 4 log( _m_ ) / (½ _ε_ ² - ⅓ _ε_ ³), which is 7,300 dimensions. That’s quite a significant dimensionality reduction! Notice that the equation does not use _n_ , it only relies on _m_ and _ε_ . This equation is implemented by the `johnson_lindenstrauss_min_dim()` function: 

```
>>> fromsklearn.random_projectionimportjohnson_lindenstrauss_min_dim
>>> m, ε=5_000, 0.1
>>> d=johnson_lindenstrauss_min_dim(m, eps=ε)
>>> d
7300
```

Now we can just generate a random matrix **P** of shape [ _d_ , _n_ ], where each item is sampled randomly from a Gaussian distribution with mean 0 and variance 1 / _d_ , and use it to project a dataset from _n_ dimensions down to _d_ : 

```
n=20_000
np.random.seed(42)
P=np.random.randn(d, n) /np.sqrt(d)  # std dev = square root of variance
```

```
X=np.random.randn(m, n)  # generate a fake dataset
X_reduced=X@P.T
```

That’s all there is to it! It’s simple and efficient, and no training is required: the only thing the algorithm needs to create the random matrix is the dataset’s shape. The data itself is not used at all. 

Scikit-Learn offers a `GaussianRandomProjection` class to do exactly what we just did: when you call its `fit()` method, it uses `johnson_lindenstrauss_min_dim()` to 

> 6 _ε_ is the Greek letter epsilon, often used for tiny values. 

**252 | Chapter 8: Dimensionality Reduction** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 

determine the output dimensionality, then it generates a random matrix, which it stores in the `components_` attribute. Then when you call `transform()` , it uses this matrix to perform the projection. When creating the transformer, you can set `eps` if you want to tweak _ε_ (it defaults to 0.1), and `n_components` if you want to force a specific target dimensionality _d_ . The following code example gives the same result as the preceding code (you can also verify that `gaussian_rnd_proj.components_` is equal to `P` ): 

```
fromsklearn.random_projectionimportGaussianRandomProjection
```

```
gaussian_rnd_proj=GaussianRandomProjection(eps=ε, random_state=42)
X_reduced=gaussian_rnd_proj.fit_transform(X)  # same result as above
```

Scikit-Learn also provides a second random projection transformer, known as `SparseRandomProjection` . It determines the target dimensionality in the same way, generates a random matrix of the same shape, and performs the projection identi‐ cally. The main difference is that the random matrix is sparse. This means it uses much less memory: about 25 MB instead of almost 1.2 GB in the preceding exam‐ ple! And it’s also much faster, both to generate the random matrix and to reduce dimensionality: about 50% faster in this case. Moreover, if the input is sparse, the transformation keeps it sparse (unless you set `dense_output=True` ). Lastly, it enjoys the same distance-preserving property as the previous approach, and the quality of the dimensionality reduction is comparable. In short, it’s usually preferable to use this transformer instead of the first one, especially for large or sparse datasets. 

The ratio _r_ of nonzero items in the sparse random matrix is called its _density_ . By default, it is equal to 1/ n . With 20,000 features, this means that only 1 in ~141 cells in the random matrix is nonzero: that’s quite sparse! You can set the `density` hyperparameter to another value if you prefer. Each cell in the sparse random matrix has a probability _r_ of being nonzero, and each nonzero value is either – _v_ or + _v_ (both equally likely), where _v_ = 1/ dr . 

If you want to perform the inverse transform, you first need to compute the pseudoinverse of the components matrix using SciPy’s `pinv()` function, then multiply the reduced data by the transpose of the pseudo-inverse: 

```
components_pinv=np.linalg.pinv(gaussian_rnd_proj.components_)
X_recovered=X_reduced@components_pinv.T
```



Computing the pseudo-inverse may take a very long time if the components matrix is large, as the computational complexity of `pinv()` is _O_ ( _dn_ ²) if _d_ < _n_ , or _O_ ( _nd_ ²) otherwise. 

**Random Projection | 253** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 

In summary, random projection is a simple, fast, memory-efficient, and surprisingly powerful dimensionality reduction algorithm that you should keep in mind, espe‐ cially when you deal with high-dimensional datasets. 



Random projection is not always used to reduce the dimensionality of large datasets. For example, a 2017 paper<sup>7</sup> by Sanjoy Dasgupta et al. showed that the brain of a fruit fly implements an analog of random projection to map dense low-dimensional olfactory inputs to sparse high-dimensional binary outputs: for each odor, only a small fraction of the output neurons get activated, but similar odors activate many of the same neurons. This is similar to a well-known algorithm called _locality sensitive hashing_ (LSH), which is typically used in search engines to group similar documents. 

## **LLE** 

_Locally linear embedding_ (LLE)<sup>8</sup> is a _nonlinear dimensionality reduction_ (NLDR) tech‐ nique. It is a manifold learning technique that does not rely on projections, unlike PCA and random projection. In a nutshell, LLE works by first measuring how each training instance linearly relates to its nearest neighbors, and then looking for a low-dimensional representation of the training set where these local relationships are best preserved (more details shortly). This approach makes it particularly good at unrolling twisted manifolds, especially when there is not too much noise. 

The following code makes a Swiss roll, then uses Scikit-Learn’s `LocallyLinearEmbed ding` class to unroll it: 

```
fromsklearn.datasetsimportmake_swiss_roll
fromsklearn.manifoldimportLocallyLinearEmbedding
```

```
X_swiss, t=make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
lle=LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_unrolled=lle.fit_transform(X_swiss)
```

The variable `t` is a 1D NumPy array containing the position of each instance along the rolled axis of the Swiss roll. We don’t use it in this example, but it can be used as a target for a nonlinear regression task. 

The resulting 2D dataset is shown in Figure 8-10. As you can see, the Swiss roll is completely unrolled, and the distances between instances are locally well preserved. 

> 7 Sanjoy Dasgupta et al., “A neural algorithm for a fundamental computing problem”, _Science_ 358, no. 6364 (2017): 793–796. 

> 8 Sam T. Roweis and Lawrence K. Saul, “Nonlinear Dimensionality Reduction by Locally Linear Embedding”, _Science_ 290, no. 5500 (2000): 2323–2326. 

**254 | Chapter 8: Dimensionality Reduction** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 



<!-- Start of picture text -->
0.08 a a ee<br>7, 0.02 ath0 Oa<br>es<br>-0.04 —0.02 0.00 0.02 0.04 0.06<br>Z1<br><!-- End of picture text -->

() 



<!-- Start of picture text -->
- | () )<br>| () Q)<br><!-- End of picture text -->

local relationships as much as possible. If **z**<sup>(</sup><sup>_i_)</sup> is the image of **x**<sup>(</sup><sup>_i_)</sup> in this _d_ -dimensional space, then we want the squared distance between **z**<sup>(</sup><sup>_i_)</sup> and ∑ mj = 1 wi , j<sup>z</sup> j to be as small as possible. This idea leads to the unconstrained optimization problem described in Equation 8-5. It looks very similar to the first step, but instead of keeping the instances fixed and finding the optimal weights, we are doing the reverse: keeping the weights fixed and finding the optimal position of the instances’ images in the low-dimensional space. Note that **Z** is the matrix containing all **z**<sup>(</sup><sup>_i_)</sup> . 

_Equation 8-5. LLE step 2: reducing dimensionality while preserving relationships_ 



Scikit-Learn’s LLE implementation has the following computational complexity: _O_ ( _m_ log( _m_ ) _n_ log( _k_ )) for finding the _k_ -nearest neighbors, _O_ ( _mnk_<sup>3</sup> ) for optimizing the weights, and _O_ ( _dm_<sup>2</sup> ) for constructing the low-dimensional representations. Unfortu‐ nately, the _m_<sup>2</sup> in the last term makes this algorithm scale poorly to very large datasets. 

As you can see, LLE is quite different from the projection techniques, and it’s sig‐ nificantly more complex, but it can also construct much better low-dimensional representations, especially if the data is nonlinear. 

## **Other Dimensionality Reduction Techniques** 

Before we conclude this chapter, let’s take a quick look at a few other popular dimensionality reduction techniques available in Scikit-Learn: 

#### `sklearn.manifold.MDS` 

_Multidimensional scaling_ (MDS) reduces dimensionality while trying to preserve the distances between the instances. Random projection does that for highdimensional data, but it doesn’t work well on low-dimensional data. 

#### `sklearn.manifold.Isomap` 

_Isomap_ creates a graph by connecting each instance to its nearest neighbors, then reduces dimensionality while trying to preserve the _geodesic distances_ between the instances. The geodesic distance between two nodes in a graph is the number of nodes on the shortest path between these nodes. 

#### `sklearn.manifold.TSNE` 

_t-distributed stochastic neighbor embedding_ (t-SNE) reduces dimensionality while trying to keep similar instances close and dissimilar instances apart. It is mostly used for visualization, in particular to visualize clusters of instances in highdimensional space. For example, in the exercises at the end of this chapter you will use t-SNE to visualize a 2D map of the MNIST images. 

##### **256 | Chapter 8: Dimensionality Reduction** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 



<!-- Start of picture text -->
rs = Ld Isomap Cone<br>Ts + ee ee ee<br>a yt. gus °<br>de US cp,<br>0 { " ° dl fF Gel | San a &<br>5 % 1 +S mo | 5 or 3 . wr<br>-15<br>-15 -30<br>~10 0 10 -40 -20 0 20 na z 7 ; + 1<br>21 » -<br><!-- End of picture text -->

**8.** Does it make any sense to chain two different dimensionality reduction algorithms? 

**9.** Load the MNIST dataset (introduced in Chapter 3) and split it into a training set and a test set (take the first 60,000 instances for training, and the remaining 10,000 for testing). Train a random forest classifier on the dataset and time how long it takes, then evaluate the resulting model on the test set. Next, use PCA to reduce the dataset’s dimensionality, with an explained variance ratio of 95%. Train a new random forest classifier on the reduced dataset and see how long it takes. Was training much faster? Next, evaluate the classifier on the test set. How does it compare to the previous classifier? Try again with an `SGDClassifier` . How much does PCA help now? 

**10.** Use t-SNE to reduce the first 5,000 images of the MNIST dataset down to 2 dimensions and plot the result using Matplotlib. You can use a scatterplot using 10 different colors to represent each image’s target class. Alternatively, you can replace each dot in the scatterplot with the corresponding instance’s class (a digit from 0 to 9), or even plot scaled-down versions of the digit images themselves (if you plot all digits the visualization will be too cluttered, so you should either draw a random sample or plot an instance only if no other instance has already been plotted at a close distance). You should get a nice visualization with wellseparated clusters of digits. Try using other dimensionality reduction algorithms, such as PCA, LLE, or MDS, and compare the resulting visualizations. 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

##### **258 | Chapter 8: Dimensionality Reduction** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:31. 

