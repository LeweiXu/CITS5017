# **CHAPTER 9 Unsupervised Learning Techniques** 

Although most of the applications of machine learning today are based on supervised learning (and as a result, this is where most of the investments go to), the vast majority of the available data is unlabeled: we have the input features **X** , but we do not have the labels **y** . The computer scientist Yann LeCun famously said that “if intelligence was a cake, unsupervised learning would be the cake, supervised learning would be the icing on the cake, and reinforcement learning would be the cherry on the cake.” In other words, there is a huge potential in unsupervised learning that we have only barely started to sink our teeth into. 

Say you want to create a system that will take a few pictures of each item on a manufacturing production line and detect which items are defective. You can fairly easily create a system that will take pictures automatically, and this might give you thousands of pictures every day. You can then build a reasonably large dataset in just a few weeks. But wait, there are no labels! If you want to train a regular binary classi‐ fier that will predict whether an item is defective or not, you will need to label every single picture as “defective” or “normal”. This will generally require human experts to sit down and manually go through all the pictures. This is a long, costly, and tedious task, so it will usually only be done on a small subset of the available pictures. As a result, the labeled dataset will be quite small, and the classifier’s performance will be disappointing. Moreover, every time the company makes any change to its products, the whole process will need to be started over from scratch. Wouldn’t it be great if the algorithm could just exploit the unlabeled data without needing humans to label every picture? Enter unsupervised learning. 

**259** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

In Chapter 8 we looked at the most common unsupervised learning task: dimension‐ ality reduction. In this chapter we will look at a few more unsupervised tasks: 

###### _Clustering_ 

The goal is to group similar instances together into _clusters_ . Clustering is a great tool for data analysis, customer segmentation, recommender systems, search engines, image segmentation, semi-supervised learning, dimensionality reduc‐ tion, and more. 

###### _Anomaly detection (also called outlier detection)_ 

The objective is to learn what “normal” data looks like, and then use that to detect abnormal instances. These instances are called _anomalies_ , or _outliers_ , while the normal instances are called _inliers_ . Anomaly detection is useful in a wide variety of applications, such as fraud detection, detecting defective products in manufacturing, identifying new trends in time series, or removing outliers from a dataset before training another model, which can significantly improve the performance of the resulting model. 

###### _Density estimation_ 

This is the task of estimating the _probability density function_ (PDF) of the random process that generated the dataset. Density estimation is commonly used for anomaly detection: instances located in very low-density regions are likely to be anomalies. It is also useful for data analysis and visualization. 

Ready for some cake? We will start with two clustering algorithms, _k_ -means and DBSCAN, then we’ll discuss Gaussian mixture models and see how they can be used for density estimation, clustering, and anomaly detection. 

## **Clustering Algorithms: k-means and DBSCAN** 

As you enjoy a hike in the mountains, you stumble upon a plant you have never seen before. You look around and you notice a few more. They are not identical, yet they are sufficiently similar for you to know that they most likely belong to the same species (or at least the same genus). You may need a botanist to tell you what species that is, but you certainly don’t need an expert to identify groups of similar-looking objects. This is called _clustering_ : it is the task of identifying similar instances and assigning them to _clusters_ , or groups of similar instances. 

Just like in classification, each instance gets assigned to a group. However, unlike classification, clustering is an unsupervised task. Consider Figure 9-1: on the left is the iris dataset (introduced in Chapter 4), where each instance’s species (i.e., its class) is represented with a different marker. It is a labeled dataset, for which classification algorithms such as logistic regression, SVMs, or random forest classifiers are well suited. On the right is the same dataset, but without the labels, so you cannot use a classification algorithm anymore. This is where clustering algorithms step in: many of 

**260 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 



<!-- Start of picture text -->
2.5 ; fe ote<br>sot, | Iris versicolor alk’ aA eevee<br><a) 4 IrisresvirginicaFar a ahaA akaA e coe“Seee coos.-°<br>Cis a o—see-ege_*<br>= : | my A +) © eee .<br>© Fs aun LS array<br>5 1.0 re] Sooo ee<br>a<br>0.5 e@@ e°<br>eae © ccoe<br>@ ecccee o<br>baC .XQ ecco<br>1 2 3 4 5 6 7 1 2 3 4 5 6 7<br>Petal length Petal length<br><!-- End of picture text -->

###### _Feature engineering_ 

The cluster affinities can often be useful as extra features. For example, we used _k_ -means in Chapter 2 to add geographic cluster affinity features to the California housing dataset, and they helped us get better performance. 

###### _Anomaly detection (also called outlier detection)_ 

Any instance that has a low affinity to all the clusters is likely to be an anomaly. For example, if you have clustered the users of your website based on their behavior, you can detect users with unusual behavior, such as an unusual number of requests per second. 

###### _Semi-supervised learning_ 

If you only have a few labels, you could perform clustering and propagate the labels to all the instances in the same cluster. This technique can greatly increase the number of labels available for a subsequent supervised learning algorithm, and thus improve its performance. 

###### _Search engines_ 

Some search engines let you search for images that are similar to a reference image. To build such a system, you would first apply a clustering algorithm to all the images in your database; similar images would end up in the same cluster. Then when a user provides a reference image, all you’d need to do is use the trained clustering model to find this image’s cluster, and you could then simply return all the images from this cluster. 

###### _Image segmentation_ 

By clustering pixels according to their color, then replacing each pixel’s color with the mean color of its cluster, it is possible to considerably reduce the number of different colors in an image. Image segmentation is used in many object detection and tracking systems, as it makes it easier to detect the contour of each object. 

There is no universal definition of what a cluster is: it really depends on the context, and different algorithms will capture different kinds of clusters. Some algorithms look for instances centered around a particular point, called a _centroid_ . Others look for continuous regions of densely packed instances: these clusters can take on any shape. Some algorithms are hierarchical, looking for clusters of clusters. And the list goes on. 

In this section, we will look at two popular clustering algorithms, _k_ -means and DBSCAN, and explore some of their applications, such as nonlinear dimensionality reduction, semi-supervised learning, and anomaly detection. 

###### **262 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 



<!-- Start of picture text -->
Lea . woh wlan .<br>2.0 i Sy ne rt gt<br>15 .* “* . .<br>it Daw<br>oY<br>ee:aa<br>1.0aa:<br>3-2 -1 0 1<br>X1<br><!-- End of picture text -->



<!-- Start of picture text -->
3.04 + ving l f.. 0° Lote ;<br>thee of a toad, er set paae . oo y re oe 3k re .. .<br>x2 CRISS hkDea woe ne ne<br>age CRE | EEE ES<br>1.5 atgs . 50 tee. 3 lal . .<br>1.0<br>-3 -2 -1 0 1<br>X1<br><!-- End of picture text -->





<!-- Start of picture text -->
Update the centroids (initially randomly) Label the instances<br>3.0] =i ea wy. . ooo AG<br>xop vege eS, Saoatar Peay<br>PRR | EEREE mais x,<br>“4 an ois ‘. . RRS . a x<br>1.0 “F .<br>3.0 , 500. 0 «Spa: we ety<br>22.0 F & “EE ; et Rb ee<br>bs " BSS<br>a3 3 Peik<br>15 Be Soe<br>Sp tsi . .<br>(x) (x]<br>1.0<br>3.04 «ging. BOO (O25 Seas y Oe<br>2.0 sag st Soe! ing sta<br>Sj Te RY <P TE Boas<br>(x) &3<br>1.0<br>-3 -2 -1 0 1 -3 -2 -1 0 1<br>x1 X1<br><!-- End of picture text -->



<!-- Start of picture text -->
Solution 1 Solution 2 (with a different random init)<br>2.5 g wgeid oe a) Ny SS frssichg eee Raa at nie.<br>2.04 Qspite . \stseeoF A roe oy 7 ‘X) 2 < aeane ie rom oa ns MCE. " madSo UGS . .<br>101 ** .<br>-3 -2 -1 0 1 -3 -2 -1 0 1<br>x1 X1<br><!-- End of picture text -->

##### **Centroid initialization methods** 

If you happen to know approximately where the centroids should be (e.g., if you ran another clustering algorithm earlier), then you can set the `init` hyperparameter to a NumPy array containing the list of centroids, and set `n_init` to `1` : 

```
good_init=np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans=KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
kmeans.fit(X)
```

Another solution is to run the algorithm multiple times with different random initializations and keep the best solution. The number of random initializations is controlled by the `n_init` hyperparameter: by default it is equal to `10` , which means that the whole algorithm described earlier runs 10 times when you call `fit()` , and Scikit-Learn keeps the best solution. But how exactly does it know which solution is the best? It uses a performance metric! That metric is called the model’s _inertia_ , which is the sum of the squared distances between the instances and their closest centroids. It is roughly equal to 219.4 for the model on the left in Figure 9-5, 258.6 for the model on the right in Figure 9-5, and only 211.6 for the model in Figure 9-3. The `KMeans` class runs the algorithm `n_init` times and keeps the model with the lowest inertia. In this example, the model in Figure 9-3 will be selected (unless we are very unlucky with `n_init` consecutive random initializations). If you are curious, a model’s inertia is accessible via the `inertia_` instance variable: 

```
>>> kmeans.inertia_
211.59853725816836
```

The `score()` method returns the negative inertia (it’s negative because a predictor’s `score()` method must always respect Scikit-Learn’s “greater is better” rule: if a predic‐ tor is better than another, its `score()` method should return a greater score): 

```
>>> kmeans.score(X)
-211.5985372581684
```

An important improvement to the _k_ -means algorithm, _k-means++_ , was proposed in a 2006 paper by David Arthur and Sergei Vassilvitskii.<sup>2</sup> They introduced a smarter initialization step that tends to select centroids that are distant from one another, and this improvement makes the _k_ -means algorithm much less likely to converge to a suboptimal solution. The paper showed that the additional computation required for the smarter initialization step is well worth it because it makes it possible to drastically reduce the number of times the algorithm needs to be run to find the optimal solution. The _k_ -means++ initialization algorithm works like this: 

**1.** Take one centroid **c**<sup>(1)</sup> , chosen uniformly at random from the dataset. 

> 2 David Arthur and Sergei Vassilvitskii, “k-Means++: The Advantages of Careful Seeding”, _Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms_ (2007): 1027–1035. 

**Clustering Algorithms: k-means and DBSCAN | 267** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

- i 2 

- **2.** Take a new centroid **c**<sup>(</sup><sup>_i_)</sup> , choosing an instance **x**<sup>(</sup><sup>_i_)</sup> with probability D x / ∑ mj = 1 D x j 2, where D( **x** ( _i_ )) is the distance between the instance **x** ( _i_ ) and the closest centroid that was already chosen. This probability distribution ensures that instances farther away from already chosen centroids are much more likely to be selected as centroids. 

**3.** Repeat the previous step until all _k_ centroids have been chosen. 

The `KMeans` class uses this initialization method by default. 

##### **Accelerated k-means and mini-batch k-means** 

Another improvement to the _k_ -means algorithm was proposed in a 2003 paper by Charles Elkan.<sup>3</sup> On some large datasets with many clusters, the algorithm can be accelerated by avoiding many unnecessary distance calculations. Elkan achieved this by exploiting the triangle inequality (i.e., that a straight line is always the shortest dis‐ tance between two points<sup>4</sup> ) and by keeping track of lower and upper bounds for dis‐ tances between instances and centroids. However, Elkan’s algorithm does not always accelerate training, and sometimes it can even slow down training significantly; it depends on the dataset. Still, if you want to give it a try, set `algorithm="elkan"` . 

Yet another important variant of the _k_ -means algorithm was proposed in a 2010 paper by David Sculley.<sup>5</sup> Instead of using the full dataset at each iteration, the algo‐ rithm is capable of using mini-batches, moving the centroids just slightly at each iteration. This speeds up the algorithm (typically by a factor of three to four) and makes it possible to cluster huge datasets that do not fit in memory. Scikit-Learn implements this algorithm in the `MiniBatchKMeans` class, which you can use just like the `KMeans` class: 

```
fromsklearn.clusterimportMiniBatchKMeans
```

```
minibatch_kmeans=MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)
```

If the dataset does not fit in memory, the simplest option is to use the `memmap` class, as we did for incremental PCA in Chapter 8. Alternatively, you can pass one mini-batch at a time to the `partial_fit()` method, but this will require much more work, since you will need to perform multiple initializations and select the best one yourself. 

> 3 Charles Elkan, “Using the Triangle Inequality to Accelerate k-Means”, _Proceedings of the 20th International Conference on Machine Learning_ (2003): 147–153. 

> 4 The triangle inequality is AC ≤ AB + BC, where A, B and C are three points and AB, AC, and BC are the distances between these points. 

- 5 David Sculley, “Web-Scale K-Means Clustering”, _Proceedings of the 19th International Conference on World Wide Web_ (2010): 1177–1178. 

###### **268 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 



<!-- Start of picture text -->
10080 ——----InertiaKMeansMini-batch KMeans 4.035. Training time (seconds) we 3i 4><br>\ 3.0 r re<br>AvaN art<br>60 : 2.5 po<br>os<br>2.0 ot<br>40 pant<br>: Ls at<br>oe<br>1.0 prt<br>20 " . a<br>0.5 44<br>os<br>0 0.0<br>20 40 60 80 100 20 40 60 80 100<br>k k<br><!-- End of picture text -->



<!-- Start of picture text -->
k=3 k=8<br>3.0 * Pay Bo RP | Ps. “ys: Eo ores)<br>25 XRT 2 ay rn iis 5 Aeese Ly aie<br>2.04 sepsis: SY Ua tetR Df” en Shares OT eG? Ee<br>, Rune a Jee hey ey Mh Rs 2a ON<br>1.0 ae' : .. no: | : te<br>-3 -2 -1 0) 1 -3 -2 -1 0 1<br>X1 X1<br>1200<br>1000<br>® 800<br>2<br>© 600<br>£<br>400<br>200<br>(e)<br>12345678<br>k<br><!-- End of picture text -->



<!-- Start of picture text -->
0.700<br>© 0.675<br>fo}<br>0.650<br>2$co) 0.625<br>—]<br>3 0.600<br>£<br>DH 0.575<br>0.550<br> 2345678<br>k<br><!-- End of picture text -->



<!-- Start of picture text -->
k=3 k=4<br>3<br>2<br>2<br>ra I<br>vo<br>a<br>9 11<br>1<br>'<br>'<br>'<br>1<br>0H<br>1!0<br>i<br>k=5 k=6<br>‘ | iS'<br>3 4 | ____e—eiws 1H<br>1<br>. 3 H<br>£ I<br>8 2<br>v 2<br>01 | iP<br>as —i1<br>-0.1 0.0 0.2 0.4 0.6 0.8 1.0 -0.1 0.0 0.2 0.4 0.6 0.8 1.0<br>Silhouette Coefficient Silhouette Coefficient<br><!-- End of picture text -->



<!-- Start of picture text -->
Inertia = 2242.6 Inertia = 2179.5<br>4 wane .<br>Sate” oan<br>x2" Rs a x magehaaes yt<br>0 “ aes oes a “re es ee: . fet at .<br>. o © . + ool es, SE je oneSe + 2 ty, MtsSe APs 5 . a . oe<br>27, Tem ot . Tepe<br>-2 0 2 4 6 -2 0 2 4 6<br>X1 x1<br><!-- End of picture text -->



The state of the art in semantic or instance segmentation today is achieved using complex architectures based on convolutional neural networks (see Chapter 14). In this chapter we are going to focus on the (much simpler) color segmentation task, using _k_ -means. 

We’ll start by importing the Pillow package (successor to the Python Imaging Library, PIL), which we’ll then use to load the _ladybug.png_ image (see the upper-left image in Figure 9-12), assuming it’s located at `filepath` : 

```
>>> importPIL
>>> image=np.asarray(PIL.Image.open(filepath))
>>> image.shape
(533, 800, 3)
```

The image is represented as a 3D array. The first dimension’s size is the height; the second is the width; and the third is the number of color channels, in this case red, green, and blue (RGB). In other words, for each pixel there is a 3D vector containing the intensities of red, green, and blue as unsigned 8-bit integers between 0 and 255. Some images may have fewer channels (such as grayscale images, which only have one), and some images may have more channels (such as images with an additional _alpha channel_ for transparency, or satellite images, which often contain channels for additional light frequencies (like infrared). 

The following code reshapes the array to get a long list of RGB colors, then it clusters these colors using _k_ -means with eight clusters. It creates a `segmented_img` array containing the nearest cluster center for each pixel (i.e., the mean color of each pixel’s cluster), and lastly it reshapes this array to the original image shape. The third line uses advanced NumPy indexing; for example, if the first 10 labels in `kmeans_.labels_` are equal to 1, then the first 10 colors in `segmented_img` are equal to `kmeans.cluster_centers_[1]` : 

```
X=image.reshape(-1, 3)
kmeans=KMeans(n_clusters=8, random_state=42).fit(X)
segmented_img=kmeans.cluster_centers_[kmeans.labels_]
segmented_img=segmented_img.reshape(image.shape)
```

This outputs the image shown in the upper right of Figure 9-12. You can experiment with various numbers of clusters, as shown in the figure. When you use fewer than eight clusters, notice that the ladybug’s flashy red color fails to get a cluster of its own: it gets merged with colors from the environment. This is because _k_ -means prefers clusters of similar sizes. The ladybug is small—much smaller than the rest of the image—so even though its color is flashy, _k_ -means fails to dedicate a cluster to it. 

**274 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 



<!-- Start of picture text -->
Original image 10 colors 8 colors<br>° Vesa <x<br>yi WZP) ’ P) P)<br>=~ ‘ . ~<br>6 colors 4 colors 2 colors<br>2 . ®.<br>- a un! i % . mern<br>‘\<br>—— , ¢ ; .<br>Pm P OY :<br>ee ae<br><!-- End of picture text -->



<!-- Start of picture text -->
{3640#2 97 2 4 8 9<br>§ © *+ 4 2 &@ € @ S$ &<br>443 4 & # 265 6 §<br>i ¢ 0 6 # 3 £§ &© # 2<br>© io 7? 5 t §$ 4 3 7<br><!-- End of picture text -->

{3640#2 97 2 4 8 9 § © *+ 4 2 &@ € @ S$ & 443 4 & # 265 6 § i ¢ 0 6 # 3 £§ &© # 2 © io 7? 5 t §$ 4 3 7 

But perhaps we can go one step further: what if we propagated the labels to all the other instances in the same cluster? This is called _label propagation_ : 

```
y_train_propagated=np.empty(len(X_train), dtype=np.int64)
foriinrange(k):
y_train_propagated[kmeans.labels_==i] =y_representative_digits[i]
```

Now let’s train the model again and look at its performance: 

```
>>> log_reg=LogisticRegression()
>>> log_reg.fit(X_train, y_train_propagated)
>>> log_reg.score(X_test, y_test)
0.8942065491183879
```

We got another significant accuracy boost! Let’s see if we can do even better by ignoring the 1% of instances that are farthest from their cluster center: this should eliminate some outliers. The following code first computes the distance from each instance to its closest cluster center, then for each cluster it sets the 1% largest distances to –1. Lastly, it creates a set without these instances marked with a –1 distance: 

```
percentile_closest=99
```

```
X_cluster_dist=X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
foriinrange(k):
in_cluster= (kmeans.labels_==i)
cluster_dist=X_cluster_dist[in_cluster]
cutoff_distance=np.percentile(cluster_dist, percentile_closest)
above_cutoff= (X_cluster_dist>cutoff_distance)
X_cluster_dist[in_cluster&above_cutoff] =-1
```

```
partially_propagated= (X_cluster_dist!=-1)
X_train_partially_propagated=X_train[partially_propagated]
y_train_partially_propagated=y_train_propagated[partially_propagated]
```

Now let’s train the model again on this partially propagated dataset and see what accuracy we get: 

```
>>> log_reg=LogisticRegression(max_iter=10_000)
>>> log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
>>> log_reg.score(X_test, y_test)
0.9093198992443325
```

Nice! With just 50 labeled instances (only 5 examples per class on average!) we got 90.9% accuracy, which is actually slightly higher than the performance we got on the fully labeled digits dataset (90.7%). This is partly thanks to the fact that we dropped some outliers, and partly because the propagated labels are actually pretty good—their accuracy is about 97.5%, as the following code shows: 

```
>>> (y_train_partially_propagated==y_train[partially_propagated]).mean()
0.9755555555555555
```

**Clustering Algorithms: k-means and DBSCAN | 277** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 



Scikit-Learn also offers two classes that can propagate labels automatically: `LabelSpreading` and `LabelPropagation` in the `sklearn.semi_supervised` package. Both classes construct a simi‐ larity matrix between all the instances, and iteratively propagate labels from labeled instances to similar unlabeled instances. There’s also a very different class called `SelfTrainingClassifier` in the same package: you give it a base classifier (such as a `RandomForest Classifier` ) and it trains it on the labeled instances, then uses it to predict labels for the unlabeled samples. It then updates the train‐ ing set with the labels it is most confident about, and repeats this process of training and labeling until it cannot add labels anymore. These techniques are not magic bullets, but they can occasionally give your model a little boost. 

#### **Active Learning** 

To continue improving your model and your training set, the next step could be to do a few rounds of _active learning_ , which is when a human expert interacts with the learning algorithm, providing labels for specific instances when the algorithm requests them. There are many different strategies for active learning, but one of the most common ones is called _uncertainty sampling_ . Here is how it works: 

**1.** The model is trained on the labeled instances gathered so far, and this model is used to make predictions on all the unlabeled instances. 

**2.** The instances for which the model is most uncertain (i.e., where its estimated probability is lowest) are given to the expert for labeling. 

**3.** You iterate this process until the performance improvement stops being worth the labeling effort. 

Other active learning strategies include labeling the instances that would result in the largest model change or the largest drop in the model’s validation error, or the instances that different models disagree on (e.g., an SVM and a random forest). 

Before we move on to Gaussian mixture models, let’s take a look at DBSCAN, another popular clustering algorithm that illustrates a very different approach based on local density estimation. This approach allows the algorithm to identify clusters of arbitrary shapes. 

###### **278 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

### **DBSCAN** 

The _density-based spatial clustering of applications with noise_ (DBSCAN) algorithm defines clusters as continuous regions of high density. Here is how it works: 

- For each instance, the algorithm counts how many instances are located within a small distance ε (epsilon) from it. This region is called the instance’s _ε-neighborhood_ . 

- If an instance has at least `min_samples` instances in its ε-neighborhood (includ‐ ing itself), then it is considered a _core instance_ . In other words, core instances are those that are located in dense regions. 

- All instances in the neighborhood of a core instance belong to the same cluster. This neighborhood may include other core instances; therefore, a long sequence of neighboring core instances forms a single cluster. 

- Any instance that is not a core instance and does not have one in its neighbor‐ hood is considered an anomaly. 

This algorithm works well if all the clusters are well separated by low-density regions. The `DBSCAN` class in Scikit-Learn is as simple to use as you might expect. Let’s test it on the moons dataset, introduced in Chapter 5: 

```
fromsklearn.clusterimportDBSCAN
fromsklearn.datasetsimportmake_moons
X, y=make_moons(n_samples=1000, noise=0.05)
dbscan=DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
```

The labels of all the instances are now available in the `labels_` instance variable: 

```
>>> dbscan.labels_
array([ 0,  2, -1, -1,  1,  0,  0,  0,  2,  5, [...], 3,  3,  4,  2,  6,  3])
```

Notice that some instances have a cluster index equal to –1, which means that they are considered as anomalies by the algorithm. The indices of the core instances are available in the `core_sample_indices_` instance variable, and the core instances themselves are available in the `components_` instance variable: 

```
>>> dbscan.core_sample_indices_
array([  0,   4,   5,   6,   7,   8,  10,  11, [...], 993, 995, 997, 998, 999])
>>> dbscan.components_
array([[-0.02137124,  0.40618608],
       [-0.84192557,  0.53058695],
       [...],
       [ 0.79419406,  0.60777171]])
```

**Clustering Algorithms: k-means and DBSCAN | 279** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 



<!-- Start of picture text -->
eps=0.05, min_samples=5 eps=0.20, min_samples=5<br>0.5 , Xa i ws a Je et<br>i xy 7 Fi<br>‘<br>-0.5aSOa AN |<br>-10 -05 00 O05 10 15 20 -10 -05 00 O05 10 15 20<br>XI X1<br><!-- End of picture text -->



<!-- Start of picture text -->
1.0 +<br>0.5 - be wae<br>0.0 #S, “a<br>ae ee<br>-0.5 TE eRrpeReee<br>-10 -05 00 O05 10 215 20<br>X1<br><!-- End of picture text -->



### **Other Clustering Algorithms** 

Scikit-Learn implements several more clustering algorithms that you should take a look at. I cannot cover them all in detail here, but here is a brief overview: 

###### _Agglomerative clustering_ 

A hierarchy of clusters is built from the bottom up. Think of many tiny bubbles floating on water and gradually attaching to each other until there’s one big group of bubbles. Similarly, at each iteration, agglomerative clustering connects the nearest pair of clusters (starting with individual instances). If you drew a tree with a branch for every pair of clusters that merged, you would get a binary tree of clusters, where the leaves are the individual instances. This approach can capture clusters of various shapes; it also produces a flexible and informative cluster tree instead of forcing you to choose a particular cluster scale, and it can be used with any pairwise distance. It can scale nicely to large numbers of instances if you provide a connectivity matrix, which is a sparse _m_ × _m_ matrix that indicates which pairs of instances are neighbors (e.g., returned by `sklearn.neighbors.kneighbors_graph()` ). Without a connectivity matrix, the algorithm does not scale well to large datasets. 

###### _BIRCH_ 

The balanced iterative reducing and clustering using hierarchies (BIRCH) algo‐ rithm was designed specifically for very large datasets, and it can be faster than batch _k_ -means, with similar results, as long as the number of features is not too large (<20). During training, it builds a tree structure containing just enough information to quickly assign each new instance to a cluster, without having to store all the instances in the tree: this approach allows it to use limited memory while handling huge datasets. 

###### _Mean-shift_ 

This algorithm starts by placing a circle centered on each instance; then for each circle it computes the mean of all the instances located within it, and it shifts the circle so that it is centered on the mean. Next, it iterates this mean-shifting step until all the circles stop moving (i.e., until each of them is centered on the mean of the instances it contains). Mean-shift shifts the circles in the direction of higher density, until each of them has found a local density maximum. Finally, all the instances whose circles have settled in the same place (or close enough) are assigned to the same cluster. Mean-shift has some of the same features as DBSCAN, like how it can find any number of clusters of any shape, it has very few hyperparameters (just one—the radius of the circles, called the _bandwidth_ ), and it relies on local density estimation. But unlike DBSCAN, mean-shift tends to chop clusters into pieces when they have internal density variations. Unfortu‐ nately, its computational complexity is _O_ ( _m_<sup>2</sup> _n_ ), so it is not suited for large datasets. 

###### **282 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

###### _Affinity propagation_ 

In this algorithm, instances repeatedly exchange messages between one another until every instance has elected another instance (or itself) to represent it. These elected instances are called _exemplars_ . Each exemplar and all the instances that elected it form one cluster. In real-life politics, you typically want to vote for a candidate whose opinions are similar to yours, but you also want them to win the election, so you might choose a candidate you don’t fully agree with, but who is more popular. You typically evaluate popularity through polls. Affinity propagation works in a similar way, and it tends to choose exemplars located near the center of clusters, similar to _k_ -means. But unlike with _k_ -means, you don’t have to pick a number of clusters ahead of time: it is determined during training. Moreover, affinity propagation can deal nicely with clusters of different sizes. Sadly, this algorithm has a computational complexity of _O_ ( _m_<sup>2</sup> ), so it is not suited for large datasets. 

###### _Spectral clustering_ 

This algorithm takes a similarity matrix between the instances and creates a lowdimensional embedding from it (i.e., it reduces the matrix’s dimensionality), then it uses another clustering algorithm in this low-dimensional space (Scikit-Learn’s implementation uses _k_ -means). Spectral clustering can capture complex cluster structures, and it can also be used to cut graphs (e.g., to identify clusters of friends on a social network). It does not scale well to large numbers of instances, and it does not behave well when the clusters have very different sizes. 

Now let’s dive into Gaussian mixture models, which can be used for density estima‐ tion, clustering, and anomaly detection. 

## **Gaussian Mixtures** 

A _Gaussian mixture model_ (GMM) is a probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown. All the instances generated from a single Gaussian distri‐ bution form a cluster that typically looks like an ellipsoid. Each cluster can have a different ellipsoidal shape, size, density, and orientation, just like in Figure 9-11. When you observe an instance, you know it was generated from one of the Gaussian distributions, but you are not told which one, and you do not know what the parameters of these distributions are. 

There are several GMM variants. In the simplest variant, implemented in the `GaussianMixture` class, you must know in advance the number _k_ of Gaussian dis‐ tributions. The dataset **X** is assumed to have been generated through the following probabilistic process: 

**Gaussian Mixtures | 283** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

- For each instance, a cluster is picked randomly from among _k_ clusters. The probability of choosing the _j_<sup>th</sup> cluster is the cluster’s weight _ϕ_<sup>(</sup><sup>_j_)</sup> .<sup>6</sup> The index of the cluster chosen for the _i_<sup>th</sup> instance is noted _z_<sup>(</sup><sup>_i_)</sup> . 

- If the _i_<sup>th</sup> instance was assigned to the _j_<sup>th</sup> cluster (i.e., _z_<sup>(</sup><sup>_i_)</sup> = _j_ ), then the location **x**<sup>(</sup><sup>_i_)</sup> of this instance is sampled randomly from the Gaussian distribution with mean **μ**<sup>(</sup><sup>_j_)</sup> and covariance matrix **Σ**<sup>(</sup><sup>_j_)</sup> . This is noted **x**<sup>(</sup><sup>_i_)</sup> ~ N ( **μ**<sup>(</sup><sup>_j_)</sup> , **Σ**<sup>(</sup><sup>_j_)</sup> ). 

So what can you do with such a model? Well, given the dataset **X** , you typically want to start by estimating the weights **ϕ** and all the distribution parameters **μ**<sup>(1)</sup> to **μ**<sup>(</sup><sup>_k_)</sup> and **Σ**<sup>(1)</sup> to **Σ**<sup>(</sup><sup>_k_)</sup> . Scikit-Learn’s `GaussianMixture` class makes this super easy: 

```
fromsklearn.mixtureimportGaussianMixture
```

```
gm=GaussianMixture(n_components=3, n_init=10)
gm.fit(X)
```

Let’s look at the parameters that the algorithm estimated: 

```
>>> gm.weights_
array([0.39025715, 0.40007391, 0.20966893])
>>> gm.means_
array([[ 0.05131611,  0.07521837],
       [-1.40763156,  1.42708225],
       [ 3.39893794,  1.05928897]])
>>> gm.covariances_
array([[[ 0.68799922,  0.79606357],
        [ 0.79606357,  1.21236106]],
       [[ 0.63479409,  0.72970799],
        [ 0.72970799,  1.1610351 ]],
       [[ 1.14833585, -0.03256179],
        [-0.03256179,  0.95490931]]])
```

Great, it worked fine! Indeed, two of the three clusters were generated with 500 instances each, while the third cluster only contains 250 instances. So the true cluster weights are 0.4, 0.4, and 0.2, respectively, and that’s roughly what the algorithm found. Similarly, the true means and covariance matrices are quite close to those found by the algorithm. But how? This class relies on the _expectation-maximization_ (EM) algorithm, which has many similarities with the _k_ -means algorithm: it also ini‐ tializes the cluster parameters randomly, then it repeats two steps until convergence, first assigning instances to clusters (this is called the _expectation step_ ) and then updating the clusters (this is called the _maximization step_ ). Sounds familiar, right? In the context of clustering, you can think of EM as a generalization of _k_ -means that not only finds the cluster centers ( **μ**<sup>(1)</sup> to **μ**<sup>(</sup><sup>_k_)</sup> ), but also their size, shape, and 

> 6 Phi ( _ϕ_ or _φ_ ) is the 21st letter of the Greek alphabet. 

###### **284 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

orientation ( **Σ**<sup>(1)</sup> to **Σ**<sup>(</sup><sup>_k_)</sup> ), as well as their relative weights ( _ϕ_<sup>(1)</sup> to _ϕ_<sup>(</sup><sup>_k_)</sup> ). Unlike _k_ -means, though, EM uses soft cluster assignments, not hard assignments. For each instance, during the expectation step, the algorithm estimates the probability that it belongs to each cluster (based on the current cluster parameters). Then, during the maximi‐ zation step, each cluster is updated using _all_ the instances in the dataset, with each instance weighted by the estimated probability that it belongs to that cluster. These probabilities are called the _responsibilities_ of the clusters for the instances. During the maximization step, each cluster’s update will mostly be impacted by the instances it is most responsible for. 



Unfortunately, just like _k_ -means, EM can end up converging to poor solutions, so it needs to be run several times, keeping only the best solution. This is why we set `n_init` to 10. Be careful: by default `n_init` is set to 1. 

You can check whether or not the algorithm converged and how many iterations it took: 

```
>>> gm.converged_
True
>>> gm.n_iter_
4
```

Now that you have an estimate of the location, size, shape, orientation, and relative weight of each cluster, the model can easily assign each instance to the most likely cluster (hard clustering) or estimate the probability that it belongs to a particular cluster (soft clustering). Just use the `predict()` method for hard clustering, or the `predict_proba()` method for soft clustering: 

```
>>> gm.predict(X)
array([0, 0, 1, ..., 2, 2, 2])
>>> gm.predict_proba(X).round(3)
array([[0.977, 0.   , 0.023],
       [0.983, 0.001, 0.016],
       [0.   , 1.   , 0.   ],
       ...,
       [0.   , 0.   , 1.   ],
       [0.   , 0.   , 1.   ],
       [0.   , 0.   , 1.   ]])
```

A Gaussian mixture model is a _generative model_ , meaning you can sample new instances from it (note that they are ordered by cluster index): 

**Gaussian Mixtures | 285** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 



<!-- Start of picture text -->
VA5<br>3<br>2<br>X24 2 x)<br>0 x]<br>-1<br>-2<br>-2 ) 2 4 6<br>Xi<br><!-- End of picture text -->



<!-- Start of picture text -->
covariance_type="tied" covariance_type="spherical"<br>4<br>2<br>0 @ (x)<br>VA<br>-2 0 2 4 6 -2 0 2 4 6<br>X1 X1<br><!-- End of picture text -->



The computational complexity of training a `GaussianMixture` model depends on the number of instances _m_ , the number of dimensions _n_ , the number of clusters _k_ , and the constraints on the covariance matrices. If `covariance_type` is `"spherical"` or `"diag"` , it is _O_ ( _kmn_ ), assuming the data has a clustering structure. If `covariance_type` is `"tied"` or `"full"` , it is _O_ ( _kmn_<sup>2</sup> + _kn_<sup>3</sup> ), so it will not scale to large numbers of features. 

Gaussian mixture models can also be used for anomaly detection. We’ll see how in the next section. 

### **Using Gaussian Mixtures for Anomaly Detection** 

Using a Gaussian mixture model for anomaly detection is quite simple: any instance located in a low-density region can be considered an anomaly. You must define what density threshold you want to use. For example, in a manufacturing company that tries to detect defective products, the ratio of defective products is usually well known. Say it is equal to 2%. You then set the density threshold to be the value that results in having 2% of the instances located in areas below that threshold density. If you notice that you get too many false positives (i.e., perfectly good products that are flagged as defective), you can lower the threshold. Conversely, if you have too many false negatives (i.e., defective products that the system does not flag as defective), you can increase the threshold. This is the usual precision/recall trade-off (see Chapter 3). Here is how you would identify the outliers using the second percentile lowest density as the threshold (i.e., approximately 2% of the instances will be flagged as anomalies): 

```
densities=gm.score_samples(X)
density_threshold=np.percentile(densities, 2)
anomalies=X[densities<density_threshold]
```

###### Figure 9-18 represents these anomalies as stars. 

A closely related task is _novelty detection_ : it differs from anomaly detection in that the algorithm is assumed to be trained on a “clean” dataset, uncontaminated by outliers, whereas anomaly detection does not make this assumption. Indeed, outlier detection is often used to clean up a dataset. 



Gaussian mixture models try to fit all the data, including the outli‐ ers; if you have too many of them this will bias the model’s view of “normality”, and some outliers may wrongly be considered as normal. If this happens, you can try to fit the model once, use it to detect and remove the most extreme outliers, then fit the model again on the cleaned-up dataset. Another approach is to use robust covariance estimation methods (see the `EllipticEnvelope` class). 

###### **288 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 



<!-- Start of picture text -->
Ve5<br>3<br>2<br>X24 (x) x)<br>0 x]<br>-1<br>-2<br>-2 0 2 4 6<br>X1<br><!-- End of picture text -->



<!-- Start of picture text -->
() ()<br>()<br><!-- End of picture text -->



<!-- Start of picture text -->
500 Model f(x; @) 012 £L(O|x = 2.5) = f(x = 2.5; 8)<br>8 1.75 0.11 poe<br> + Cowl od<br>sos self | le<br>00 sel<br>-6 —4 —2 0 2 4 1.0 1.2 1.4 1.6 1.8 2.0<br>PDF f(x;@ = 1.3) 4 log £(O|x= 2.5)<br>. OF loge, O9F—<br>oo N ae<br>-6 -4 —2 0 2 4 1.0 1.2 1.4 1.6 1.8 2.0<br>x 0<br><!-- End of picture text -->

In short, the PDF is a function of _x_ (with _θ_ fixed), while the likelihood function is a function of _θ_ (with _x_ fixed). It is important to understand that the likelihood function is _not_ a probability distribution: if you integrate a probability distribution over all possible values of _x_ , you always get 1, but if you integrate the likelihood function over all possible values of _θ_ the result can be any positive value. 

Given a dataset **X** , a common task is to try to estimate the most likely values for the model parameters. To do this, you must find the values that maximize the likelihood function, given **X** . In this example, if you have observed a single instance _x_ =2.5, the _maximum likelihood estimate_ (MLE) of _θ_ is θ ≈1.66. If a prior probability distribution _g_ over _θ_ exists, it is possible to take it into account by maximizing ℒ ( _θ_ | _x_ )g( _θ_ ) rather than just maximizing ℒ ( _θ_ | _x_ ). This is called _maximum a-posteriori_ (MAP) estimation. Since MAP constrains the parameter values, you can think of it as a regularized version of MLE. 

Notice that maximizing the likelihood function is equivalent to maximizing its loga‐ rithm (represented in the lower-right plot in Figure 9-19). Indeed, the logarithm is a strictly increasing function, so if _θ_ maximizes the log likelihood, it also maximizes the likelihood. It turns out that it is generally easier to maximize the log likelihood. For example, if you observed several independent instances _x_<sup>(1)</sup> to _x_<sup>(</sup><sup>_m_)</sup> , you would need to find the value of _θ_ that maximizes the product of the individual likelihood functions. But it is equivalent, and much simpler, to maximize the sum (not the product) of the log likelihood functions, thanks to the magic of the logarithm which converts products into sums: log( _ab_ ) = log( _a_ ) + log( _b_ ). 

Once you have estimated θ , the value of _θ_ that maximizes the likelihood function, then you are ready to compute ℒ = ℒ θ , X , which is the value used to compute the AIC and BIC; you can think of it as a measure of how well the model fits the data. 

To compute the BIC and AIC, call the `bic()` and `aic()` methods: 

```
>>> gm.bic(X)
8189.747000497186
>>> gm.aic(X)
8102.521720382148
```

Figure 9-20 shows the BIC for different numbers of clusters _k_ . As you can see, both the BIC and the AIC are lowest when _k_ =3, so it is most likely the best choice. 

**Gaussian Mixtures | 291** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 



<!-- Start of picture text -->
Cc<br>© 9200 4S ° BIC<br>2 N<br>2 9000 ~ -e- AIC<br>=<br>YL ge00 \ =<br>5 \ Minimum<br>“5 8600 \<br>O<br>£<br>je}a sao0 \\\ J<br>‘= 8200 \<br>< ny nn een, tet Sti Lettie alee<br>.23456789<br>k<br><!-- End of picture text -->



<!-- Start of picture text -->
1.00 pedrai a, in<br>0.75 age? ne<br>0.50} 2 %. at Y<br>20.254 te Bs & “ge %<br>-0.250.00 4s + Be x e ea<br>cen oe Sic:<br>-0.50 Raga CEP<br>-1.0 -0.5 0.0 0.5 1.0 1.5 2.0 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0<br>X1 X1<br><!-- End of picture text -->

###### _One-class SVM_ 

This algorithm is better suited for novelty detection. Recall that a kernelized SVM classifier separates two classes by first (implicitly) mapping all the instances to a high-dimensional space, then separating the two classes using a linear SVM classifier within this high-dimensional space (see Chapter 5). Since we just have one class of instances, the one-class SVM algorithm instead tries to separate the instances in high-dimensional space from the origin. In the original space, this will correspond to finding a small region that encompasses all the instances. If a new instance does not fall within this region, it is an anomaly. There are a few hyperparameters to tweak: the usual ones for a kernelized SVM, plus a margin hyperparameter that corresponds to the probability of a new instance being mistakenly considered as novel when it is in fact normal. It works great, especially with high-dimensional datasets, but like all SVMs it does not scale to large datasets. 

_PCA and other dimensionality reduction techniques with an_ `inverse_transform()` _method_ 

If you compare the reconstruction error of a normal instance with the recon‐ struction error of an anomaly, the latter will usually be much larger. This is a simple and often quite efficient anomaly detection approach (see this chapter’s exercises for an example). 

## **Exercises** 

**1.** How would you define clustering? Can you name a few clustering algorithms? 

**2.** What are some of the main applications of clustering algorithms? 

**3.** Describe two techniques to select the right number of clusters when using _k_ -means. 

**4.** What is label propagation? Why would you implement it, and how? 

**5.** Can you name two clustering algorithms that can scale to large datasets? And two that look for regions of high density? 

**6.** Can you think of a use case where active learning would be useful? How would you implement it? 

**7.** What is the difference between anomaly detection and novelty detection? 

**8.** What is a Gaussian mixture? What tasks can you use it for? 

**9.** Can you name two techniques to find the right number of clusters when using a Gaussian mixture model? 

**10.** The classic Olivetti faces dataset contains 400 grayscale 64 × 64–pixel images of faces. Each image is flattened to a 1D vector of size 4,096. Forty different people were photographed (10 times each), and the usual task is to train a model that 

###### **294 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

can predict which person is represented in each picture. Load the dataset using the `sklearn.datasets.fetch_olivetti_faces()` function, then split it into a training set, a validation set, and a test set (note that the dataset is already scaled between 0 and 1). Since the dataset is quite small, you will probably want to use stratified sampling to ensure that there are the same number of images per person in each set. Next, cluster the images using _k_ -means, and ensure that you have a good number of clusters (using one of the techniques discussed in this chapter). Visualize the clusters: do you see similar faces in each cluster? 

**11.** Continuing with the Olivetti faces dataset, train a classifier to predict which person is represented in each picture, and evaluate it on the validation set. Next, use _k_ -means as a dimensionality reduction tool, and train a classifier on the reduced set. Search for the number of clusters that allows the classifier to get the best performance: what performance can you reach? What if you append the features from the reduced set to the original features (again, searching for the best number of clusters)? 

**12.** Train a Gaussian mixture model on the Olivetti faces dataset. To speed up the algorithm, you should probably reduce the dataset’s dimensionality (e.g., use PCA, preserving 99% of the variance). Use the model to generate some new faces (using the `sample()` method), and visualize them (if you used PCA, you will need to use its `inverse_transform()` method). Try to modify some images (e.g., rotate, flip, darken) and see if the model can detect the anomalies (i.e., compare the output of the `score_samples()` method for normal images and for anomalies). 

**13.** Some dimensionality reduction techniques can also be used for anomaly detec‐ tion. For example, take the Olivetti faces dataset and reduce it with PCA, preserv‐ ing 99% of the variance. Then compute the reconstruction error for each image. Next, take some of the modified images you built in the previous exercise and look at their reconstruction error: notice how much larger it is. If you plot a reconstructed image, you will see why: it tries to reconstruct a normal face. 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

**Exercises | 295** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

