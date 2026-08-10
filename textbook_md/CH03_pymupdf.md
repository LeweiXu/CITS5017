# **CHAPTER 3 Classification** 

In Chapter 1 I mentioned that the most common supervised learning tasks are regression (predicting values) and classification (predicting classes). In Chapter 2 we explored a regression task, predicting housing values, using various algorithms such as linear regression, decision trees, and random forests (which will be explained in further detail in later chapters). Now we will turn our attention to classification systems. 

## **MNIST** 

In this chapter we will be using the MNIST dataset, which is a set of 70,000 small images of digits handwritten by high school students and employees of the US Census Bureau. Each image is labeled with the digit it represents. This set has been studied so much that it is often called the “hello world” of machine learning: whenever people come up with a new classification algorithm they are curious to see how it will perform on MNIST, and anyone who learns machine learning tackles this dataset sooner or later. 

Scikit-Learn provides many helper functions to download popular datasets. MNIST is one of them. The following code fetches the MNIST dataset from OpenML.org:<sup>1</sup> 

```
fromsklearn.datasetsimportfetch_openml
```

```
mnist=fetch_openml('mnist_784', as_frame=False)
```

The `sklearn.datasets` package contains mostly three types of functions: `fetch_*` functions such as `fetch_openml()` to download real-life datasets, `load_*` functions 

- 1 By default Scikit-Learn caches downloaded datasets in a directory called _scikit_learn_data_ in your home directory. 

**103** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 

to load small toy datasets bundled with Scikit-Learn (so they don’t need to be down‐ loaded over the internet), and `make_*` functions to generate fake datasets, useful for tests. Generated datasets are usually returned as an `(X, y)` tuple containing the input data and the targets, both as NumPy arrays. Other datasets are returned as `sklearn.utils.Bunch` objects, which are dictionaries whose entries can also be accessed as attributes. They generally contain the following entries: 

##### `"DESCR"` 

A description of the dataset 

##### `"data"` 

The input data, usually as a 2D NumPy array 

```
"target"
```

The labels, usually as a 1D NumPy array 

The `fetch_openml()` function is a bit unusual since by default it returns the inputs as a Pandas DataFrame and the labels as a Pandas Series (unless the dataset is sparse). But the MNIST dataset contains images, and DataFrames aren’t ideal for that, so it’s preferable to set `as_frame=False` to get the data as NumPy arrays instead. Let’s look at these arrays: 

```
>>> X, y=mnist.data, mnist.target
>>> X
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
>>> X.shape
(70000, 784)
>>> y
array(['5', '0', '4', ..., '4', '5', '6'], dtype=object)
>>> y.shape
(70000,)
```

There are 70,000 images, and each image has 784 features. This is because each image is 28 × 28 pixels, and each feature simply represents one pixel’s intensity, from 0 (white) to 255 (black). Let’s take a peek at one digit from the dataset (Figure 3-1). All we need to do is grab an instance’s feature vector, reshape it to a 28 × 28 array, and display it using Matplotlib’s `imshow()` function. We use `cmap="binary"` to get a grayscale color map where 0 is white and 255 is black: 

**104 | Chapter 3: Classification** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 





<!-- Start of picture text -->
SOU /GTANZBIY<br>3s53GO\7327639<br>Hdvg/A2y327<br>SBF 67O0F690276<br>17973979 8S4N3<br>Z2O0O7FVTFOINS<br>Y¢6O@Ys6E/O08<br>17)6362/)7<br>SOQA6T7TBRXI970OY<br>6746390783)<br><!-- End of picture text -->

Now we can use it to detect images of the number 5: 

```
>>> sgd_clf.predict([some_digit])
array([ True])
```

The classifier guesses that this image represents a 5 ( `True` ). Looks like it guessed right in this particular case! Now, let’s evaluate this model’s performance. 

## **Performance Measures** 

Evaluating a classifier is often significantly trickier than evaluating a regressor, so we will spend a large part of this chapter on this topic. There are many performance measures available, so grab another coffee and get ready to learn a bunch of new concepts and acronyms! 

### **Measuring Accuracy Using Cross-Validation** 

A good way to evaluate a model is to use cross-validation, just as you did in Chapter 2. Let’s use the `cross_val_score()` function to evaluate our `SGDClassifier` model, using _k_ -fold cross-validation with three folds. Remember that _k_ -fold crossvalidation means splitting the training set into _k_ folds (in this case, three), then training the model _k_ times, holding out a different fold each time for evaluation (see Chapter 2): 

```
>>> fromsklearn.model_selectionimportcross_val_score
>>> cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
array([0.95035, 0.96035, 0.9604 ])
```

Wow! Above 95% accuracy (ratio of correct predictions) on all cross-validation folds? This looks amazing, doesn’t it? Well, before you get too excited, let’s look at a dummy classifier that just classifies every single image in the most frequent class, which in this case is the negative class (i.e., _non_ 5): 

```
fromsklearn.dummyimportDummyClassifier
```

```
dummy_clf=DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))  # prints False: no 5s detected
```

Can you guess this model’s accuracy? Let’s find out: 

```
>>> cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
array([0.90965, 0.90965, 0.90965])
```

That’s right, it has over 90% accuracy! This is simply because only about 10% of the images are 5s, so if you always guess that an image is _not_ a 5, you will be right about 90% of the time. Beats Nostradamus. 

This demonstrates why accuracy is generally not the preferred performance measure for classifiers, especially when you are dealing with _skewed datasets_ (i.e., when some 

**Performance Measures | 107** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 

classes are much more frequent than others). A much better way to evaluate the performance of a classifier is to look at the _confusion matrix_ (CM). 

#### **Implementing Cross-Validation** 

Occasionally you will need more control over the cross-validation process than what Scikit-Learn provides off the shelf. In these cases, you can implement crossvalidation yourself. The following code does roughly the same thing as Scikit-Learn’s `cross_val_score()` function, and it prints the same result: 

```
fromsklearn.model_selectionimportStratifiedKFold
fromsklearn.baseimportclone
skfolds=StratifiedKFold(n_splits=3)  # add shuffle=True if the dataset is
# not already shuffled
fortrain_index, test_indexinskfolds.split(X_train, y_train_5):
clone_clf=clone(sgd_clf)
X_train_folds=X_train[train_index]
y_train_folds=y_train_5[train_index]
X_test_fold=X_train[test_index]
y_test_fold=y_train_5[test_index]
clone_clf.fit(X_train_folds, y_train_folds)
y_pred=clone_clf.predict(X_test_fold)
n_correct=sum(y_pred==y_test_fold)
print(n_correct/len(y_pred))  # prints 0.95035, 0.96035, and 0.9604
```

The `StratifiedKFold` class performs stratified sampling (as explained in Chapter 2) to produce folds that contain a representative ratio of each class. At each iteration the code creates a clone of the classifier, trains that clone on the training folds, and makes predictions on the test fold. Then it counts the number of correct predictions and outputs the ratio of correct predictions. 

### **Confusion Matrices** 

The general idea of a confusion matrix is to count the number of times instances of class A are classified as class B, for all A/B pairs. For example, to know the number of times the classifier confused images of 8s with 0s, you would look at row #8, column #0 of the confusion matrix. 

To compute the confusion matrix, you first need to have a set of predictions so that they can be compared to the actual targets. You could make predictions on the test set, but it’s best to keep that untouched for now (remember that you want to use the test set only at the very end of your project, once you have a classifier that you are ready to launch). Instead, you can use the `cross_val_predict()` function: 

**108 | Chapter 3: Classification** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 

```
fromsklearn.model_selectionimportcross_val_predict
```

```
y_train_pred=cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

Just like the `cross_val_score()` function, `cross_val_predict()` performs _k_ -fold cross-validation, but instead of returning the evaluation scores, it returns the predic‐ tions made on each test fold. This means that you get a clean prediction for each instance in the training set (by “clean” I mean “out-of-sample”: the model makes predictions on data that it never saw during training). 

Now you are ready to get the confusion matrix using the `confusion_matrix()` function. Just pass it the target classes ( `y_train_5` ) and the predicted classes ( `y_train_pred` ): 

```
>>> fromsklearn.metricsimportconfusion_matrix
>>> cm=confusion_matrix(y_train_5, y_train_pred)
>>> cm
array([[53892,   687],
       [ 1891,  3530]])
```

Each row in a confusion matrix represents an _actual class_ , while each column rep‐ resents a _predicted class_ . The first row of this matrix considers non-5 images (the _negative class_ ): 53,892 of them were correctly classified as non-5s (they are called _true negatives_ ), while the remaining 687 were wrongly classified as 5s ( _false positives_ , also called _type I errors_ ). The second row considers the images of 5s (the _positive class_ ): 1,891 were wrongly classified as non-5s ( _false negatives_ , also called _type II errors_ ), while the remaining 3,530 were correctly classified as 5s ( _true positives_ ). A perfect classifier would only have true positives and true negatives, so its confusion matrix would have nonzero values only on its main diagonal (top left to bottom right): 

```
>>> y_train_perfect_predictions=y_train_5# pretend we reached perfection
>>> confusion_matrix(y_train_5, y_train_perfect_predictions)
array([[54579,     0],
       [    0,  5421]])
```

The confusion matrix gives you a lot of information, but sometimes you may prefer a more concise metric. An interesting one to look at is the accuracy of the positive predictions; this is called the _precision_ of the classifier (Equation 3-1). 

_Equation 3-1. Precision_ 



_TP_ is the number of true positives, and _FP_ is the number of false positives. 

A trivial way to have perfect precision is to create a classifier that always makes negative predictions, except for one single positive prediction on the instance it’s 

**Performance Measures | 109** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 



<!-- Start of picture text -->
Predicted<br>[TN | Negative Positive FP<br>Actual : Precision<br>ooFot%po ee (eg., 30ut of4)<br>Pe<br>FN cmRecall  iia<br>(e.g., 3 out of 5)<br><!-- End of picture text -->

Now our 5-detector does not look as shiny as it did when we looked at its accuracy. When it claims an image represents a 5, it is correct only 83.7% of the time. More‐ over, it only detects 65.1% of the 5s. 

It is often convenient to combine precision and recall into a single metric called the _F1 score_ , especially when you need a single metric to compare two classifiers. The F1 score is the _harmonic mean_ of precision and recall (Equation 3-3). Whereas the regular mean treats all values equally, the harmonic mean gives much more weight to low values. As a result, the classifier will only get a high F1 score if both recall and precision are high. 

_Equation 3-3. F1 score_ 



To compute the F1 score, simply call the `f1_score()` function: 

```
>>> fromsklearn.metricsimportf1_score
>>> f1_score(y_train_5, y_train_pred)
0.7325171197343846
```

The F1 score favors classifiers that have similar precision and recall. This is not always what you want: in some contexts you mostly care about precision, and in other contexts you really care about recall. For example, if you trained a classifier to detect videos that are safe for kids, you would probably prefer a classifier that rejects many good videos (low recall) but keeps only safe ones (high precision), rather than a classifier that has a much higher recall but lets a few really bad videos show up in your product (in such cases, you may even want to add a human pipeline to check the classifier’s video selection). On the other hand, suppose you train a classifier to detect shoplifters in surveillance images: it is probably fine if your classifier only has 30% precision as long as it has 99% recall (sure, the security guards will get a few false alerts, but almost all shoplifters will get caught). 

Unfortunately, you can’t have it both ways: increasing precision reduces recall, and vice versa. This is called the . _precision/recall trade-off_ 

### **The Precision/Recall Trade-off** 

To understand this trade-off, let’s look at how the `SGDClassifier` makes its classifica‐ tion decisions. For each instance, it computes a score based on a _decision function_ . If that score is greater than a threshold, it assigns the instance to the positive class; otherwise it assigns it to the negative class. Figure 3-4 shows a few digits positioned from the lowest score on the left to the highest score on the right. Suppose the _decision threshold_ is positioned at the central arrow (between the two 5s): you will 

**Performance Measures | 111** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 



<!-- Start of picture text -->
Precision: 6/8=75% 4/5=80% 3/3=100%<br>Recall: 6/6=100% 4/6=67% 3/6=50%<br>5 , all<br> +3 AS 2 Ste cls s ie<br>a Sy a core<br>Negative - I .?<br> predictions ae | ,? Positive predictions<br>— Various~  thresholdswv —)<br><!-- End of picture text -->



<!-- Start of picture text -->
1.0 — T<br>08 iw aoe ~Sapa!wyWsMAI<br>/<br>°-8 7 ; --- Precision<br>/<br>0.4 se threshold<br>7<br>4<br>A<br>0.2 ¢<br>0.0 —_—_—<br>—40000 20000 0 20000 40000<br>Threshold<br><!-- End of picture text -->





<!-- Start of picture text -->
: thresholdHigher \<br>c 0.6<br>md}<br>U<br>v<br>:7 PN<br>0.2PY— Precision/Recall curve: \<br>e §=Point at threshold 3,000<br>0.0 :<br>0.0 0.2 0.4 0.6 0.8 1.0<br>Recall<br><!-- End of picture text -->

To make predictions (on the training set for now), instead of calling the classifier’s `predict()` method, you can run this code: 

```
y_train_pred_90= (y_scores>=threshold_for_90_precision)
```

Let’s check these predictions’ precision and recall: 

```
>>> precision_score(y_train_5, y_train_pred_90)
0.9000345901072293
>>> recall_at_90_precision=recall_score(y_train_5, y_train_pred_90)
>>> recall_at_90_precision
0.4799852425751706
```

Great, you have a 90% precision classifier! As you can see, it is fairly easy to create a classifier with virtually any precision you want: just set a high enough threshold, and you’re done. But wait, not so fast–a high-precision classifier is not very useful if its recall is too low! For many applications, 48% recall wouldn’t be great at all. 



If someone says, “Let’s reach 99% precision”, you should ask, “At what recall?” 

### **The ROC Curve** 

The _receiver operating characteristic_ (ROC) curve is another common tool used with binary classifiers. It is very similar to the precision/recall curve, but instead of plot‐ ting precision versus recall, the ROC curve plots the _true positive rate_ (another name for recall) against the _false positive rate_ (FPR). The FPR (also called the _fall-out_ ) is the ratio of negative instances that are incorrectly classified as positive. It is equal to 1 – the _true negative rate_ (TNR), which is the ratio of negative instances that are correctly classified as negative. The TNR is also called _specificity_ . Hence, the ROC curve plots _sensitivity_ (recall) versus 1 – _specificity_ . 

To plot the ROC curve, you first use the `roc_curve()` function to compute the TPR and FPR for various threshold values: 

```
fromsklearn.metricsimportroc_curve
```

```
fpr, tpr, thresholds=roc_curve(y_train_5, y_scores)
```

Then you can plot the FPR against the TPR using Matplotlib. The following code produces the plot in Figure 3-7. To find the point that corresponds to 90% precision, we need to look for the index of the desired threshold. Since thresholds are listed in decreasing order in this case, we use `<=` instead of `>=` on the first line: 

**Performance Measures | 115** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 



<!-- Start of picture text -->
| / ea<br>208 Higher<br>S threshold<br>ao<br>x<br>@ 0.6<br>~<br>©<br>3<br>2 oO ee<br>= 04<br>fe)<br>F2 0.2 _ -“. =———_ ROC. curve<br>“+ Random classifier's ROC curve<br>e Threshold for 90% precision<br>0.0<br>0.0 0.2 0.4 0.6 0.8 1.0<br>False Positive Rate (Fall-Out)<br><!-- End of picture text -->



Since the ROC curve is so similar to the precision/recall (PR) curve, you may wonder how to decide which one to use. As a rule of thumb, you should prefer the PR curve whenever the positive class is rare or when you care more about the false positives than the false negatives. Otherwise, use the ROC curve. For example, looking at the previous ROC curve (and the ROC AUC score), you may think that the classifier is really good. But this is mostly because there are few positives (5s) compared to the negatives (non-5s). In contrast, the PR curve makes it clear that the classifier has room for improvement: the curve could really be closer to the top-right corner (see Figure 3-6 again). 

Let’s now create a `RandomForestClassifier` , whose PR curve and F1 score we can compare to those of the `SGDClassifier` : 

```
fromsklearn.ensembleimportRandomForestClassifier
```

```
forest_clf=RandomForestClassifier(random_state=42)
```

The `precision_recall_curve()` function expects labels and scores for each instance, so we need to train the random forest classifier and make it assign a score to each instance. But the `RandomForestClassifier` class does not have a `decision_function()` method, due to the way it works (we will cover this in Chapter 7). Luckily, it has a `predict_proba()` method that returns estimated class probabilities for each instance, and we can just use the probability of the positive class as a score, so `precision_recall_curve()` will work.<sup>4</sup> We can call the `cross_val_pre dict()` function to train the `RandomForestClassifier` using cross-validation and make it predict class probabilities for every image as follows: 

```
y_probas_forest=cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
method="predict_proba")
```

Let’s look at the estimated class probabilities for the first two images in the training set: 

```
>>> y_probas_forest[:2]
array([[0.11, 0.89],
       [0.99, 0.01]])
```

The model predicts that the first image is positive with 89% probability, and it predicts that the second image is negative with 99% probability. Since each image is either positive or negative, the estimated probabilities in each row add up to 100%. 

> 4 Scikit-Learn classifiers always have either a `decision_function()` method or a `predict_proba()` method, or sometimes both. 

**Performance Measures | 117** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 





<!-- Start of picture text -->
1.0<br>oAttanaee<br>I<br>eeer,<br>0.8 wae<br>Sy.<br>aN<br>x,x<br>\<br>Cc 0.6<br>2 ‘<br>ay \<br>U \<br>oO AY<br>pas \<br>2 04<br>,<br>\<br>\\\<br>0.2<br>— Random Forest<br>--- SGD<br>0.0<br>0.0 0.2 0.4 0.6 0.8 1.0<br>Recall<br><!-- End of picture text -->

As you can see in Figure 3-8, the `RandomForestClassifier` ’s PR curve looks much better than the `SGDClassifier` ’s: it comes much closer to the top-right corner. Its F1 score and ROC AUC score are also significantly better: 

```
>>> y_train_pred_forest=y_probas_forest[:, 1] >=0.5# positive proba ≥ 50%
>>> f1_score(y_train_5, y_train_pred_forest)
0.9242275142688446
>>> roc_auc_score(y_train_5, y_scores_forest)
0.9983436731328145
```

Try measuring the precision and recall scores: you should find about 99.1% precision and 86.6% recall. Not too bad! 

You now know how to train binary classifiers, choose the appropriate metric for your task, evaluate your classifiers using cross-validation, select the precision/recall trade-off that fits your needs, and use several metrics and curves to compare various models. You’re ready to try to detect more than just the 5s. 

## **Multiclass Classification** 

Whereas binary classifiers distinguish between two classes, _multiclass classifiers_ (also called _multinomial classifiers_ ) can distinguish between more than two classes. 

Some Scikit-Learn classifiers (e.g., `LogisticRegression` , `RandomForestClassifier` , and `GaussianNB` ) are capable of handling multiple classes natively. Others are strictly binary classifiers (e.g., `SGDClassifier` and `SVC` ). However, there are various strategies that you can use to perform multiclass classification with multiple binary classifiers. 

One way to create a system that can classify the digit images into 10 classes (from 0 to 9) is to train 10 binary classifiers, one for each digit (a 0-detector, a 1-detector, a 2-detector, and so on). Then when you want to classify an image, you get the decision score from each classifier for that image and you select the class whose classifier outputs the highest score. This is called the _one-versus-the-rest_ (OvR) strategy, or sometimes _one-versus-all_ (OvA). 

Another strategy is to train a binary classifier for every pair of digits: one to distin‐ guish 0s and 1s, another to distinguish 0s and 2s, another for 1s and 2s, and so on. This is called the _one-versus-one_ (OvO) strategy. If there are _N_ classes, you need to train _N_ × ( _N_ – 1) / 2 classifiers. For the MNIST problem, this means training 45 binary classifiers! When you want to classify an image, you have to run the image through all 45 classifiers and see which class wins the most duels. The main advantage of OvO is that each classifier only needs to be trained on the part of the training set containing the two classes that it must distinguish. 

Some algorithms (such as support vector machine classifiers) scale poorly with the size of the training set. For these algorithms OvO is preferred because it is faster 

**Multiclass Classification | 119** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 

to train many classifiers on small training sets than to train few classifiers on large training sets. For most binary classification algorithms, however, OvR is preferred. 

Scikit-Learn detects when you try to use a binary classification algorithm for a multiclass classification task, and it automatically runs OvR or OvO, depending on the algorithm. Let’s try this with a support vector machine classifier using the `sklearn.svm.SVC` class (see Chapter 5). We’ll only train on the first 2,000 images, or else it will take a very long time: 

```
fromsklearn.svmimportSVC
```

```
svm_clf=SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])  # y_train, not y_train_5
```

That was easy! We trained the `SVC` using the original target classes from 0 to 9 ( `y_train` ), instead of the 5-versus-the-rest target classes ( `y_train_5` ). Since there are 10 classes (i.e., more than 2), Scikit-Learn used the OvO strategy and trained 45 binary classifiers. Now let’s make a prediction on an image: 

```
>>> svm_clf.predict([some_digit])
array(['5'], dtype=object)
```

That’s correct! This code actually made 45 predictions—one per pair of classes—and it selected the class that won the most duels.<sup>5</sup> If you call the `decision_function()` method, you will see that it returns 10 scores per instance: one per class. Each class gets a score equal to the number of won duels plus or minus a small tweak (max ±0.33) to break ties, based on the classifier scores: 

```
>>> some_digit_scores=svm_clf.decision_function([some_digit])
>>> some_digit_scores.round(2)
array([[ 3.79,  0.73,  6.06,  8.3 , -0.29,  9.3 ,  1.75,  2.77,  7.21,
         4.82]])
```

The highest score is 9.3, and it’s indeed the one corresponding to class 5: 

```
>>> class_id=some_digit_scores.argmax()
>>> class_id
5
```

When a classifier is trained, it stores the list of target classes in its `classes_` attribute, ordered by value. In the case of MNIST, the index of each class in the `classes_` array conveniently matches the class itself (e.g., the class at index 5 happens to be class `'5'` ), but in general you won’t be so lucky; you will need to look up the class label like this: 

```
>>> svm_clf.classes_
array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype=object)
```

> 5 In case of a tie, the first class is selected, unless you set the `break_ties` hyperparameters to `True` , in which case ties are broken using the output of the `decision_function()` . 

###### **120 | Chapter 3: Classification** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 

```
>>> svm_clf.classes_[class_id]
'5'
```

If you want to force Scikit-Learn to use one-versus-one or one-versus-the-rest, you can use the `OneVsOneClassifier` or `OneVsRestClassifier` classes. Simply create an instance and pass a classifier to its constructor (it doesn’t even have to be a binary classifier). For example, this code creates a multiclass classifier using the OvR strategy, based on an `SVC` : 

```
fromsklearn.multiclassimportOneVsRestClassifier
```

```
ovr_clf=OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
```

Let’s make a prediction, and check the number of trained classifiers: 

```
>>> ovr_clf.predict([some_digit])
array(['5'], dtype='<U1')
>>> len(ovr_clf.estimators_)
10
```

Training an `SGDClassifier` on a multiclass dataset and using it to make predictions is just as easy: 

```
>>> sgd_clf=SGDClassifier(random_state=42)
>>> sgd_clf.fit(X_train, y_train)
>>> sgd_clf.predict([some_digit])
array(['3'], dtype='<U1')
```

Oops, that’s incorrect. Prediction errors do happen! This time Scikit-Learn used the OvR strategy under the hood: since there are 10 classes, it trained 10 binary classifiers. The `decision_function()` method now returns one value per class. Let’s look at the scores that the SGD classifier assigned to each class: 

```
>>> sgd_clf.decision_function([some_digit]).round()
array([[-31893., -34420.,  -9531.,   1824., -22320.,  -1386., -26189.,
        -16148.,  -4604., -12051.]])
```

You can see that the classifier is not very confident about its prediction: almost all scores are very negative, while class 3 has a score of +1,824, and class 5 is not too far behind at –1,386. Of course, you’ll want to evaluate this classifier on more than one image. Since there are roughly the same number of images in each class, the accuracy metric is fine. As usual, you can use the `cross_val_score()` function to evaluate the model: 

```
>>> cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
array([0.87365, 0.85835, 0.8689 ])
```

It gets over 85.8% on all test folds. If you used a random classifier, you would get 10% accuracy, so this is not such a bad score, but you can still do much better. Simply scaling the inputs (as discussed in Chapter 2) increases accuracy above 89.1%: 

**Multiclass Classification | 121** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 

```
>>> fromsklearn.preprocessingimportStandardScaler
```

```
>>> scaler=StandardScaler()
```

```
>>> X_train_scaled=scaler.fit_transform(X_train.astype("float64"))
>>> cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
array([0.8983, 0.891 , 0.9018])
```

## **Error Analysis** 

If this were a real project, you would now follow the steps in your machine learning project checklist (see Appendix A). You’d explore data preparation options, try out multiple models, shortlist the best ones, fine-tune their hyperparameters using `Grid SearchCV` , and automate as much as possible. Here, we will assume that you have found a promising model and you want to find ways to improve it. One way to do this is to analyze the types of errors it makes. 

First, look at the confusion matrix. For this, you first need to make predictions using the `cross_val_predict()` function; then you can pass the labels and predictions to the `confusion_matrix()` function, just like you did earlier. However, since there are now 10 classes instead of 2, the confusion matrix will contain quite a lot of numbers, and it may be hard to read. 

A colored diagram of the confusion matrix is much easier to analyze. To plot such a diagram, use the `ConfusionMatrixDisplay.from_predictions()` function like this: 

```
fromsklearn.metricsimportConfusionMatrixDisplay
```

```
y_train_pred=cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()
```

This produces the left diagram in Figure 3-9. This confusion matrix looks pretty good: most images are on the main diagonal, which means that they were classified correctly. Notice that the cell on the diagonal in row #5 and column #5 looks slightly darker than the other digits. This could be because the model made more errors on 5s, or because there are fewer 5s in the dataset than the other digits. That’s why it’s important to normalize the confusion matrix by dividing each value by the total number of images in the corresponding (true) class (i.e., divide by the row’s sum). This can be done simply by setting `normalize="true"` . We can also specify the `val ues_format=".0%"` argument to show percentages with no decimals. The following code produces the diagram on the right in Figure 3-9: 

```
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
normalize="true", values_format=".0%")
plt.show()
```

Now we can easily see that only 82% of the images of 5s were classified correctly. The most common error the model made with images of 5s was to misclassify them as 8s: this happened for 10% of all 5s. But only 2% of 8s got misclassified as 5s; confusion 

**122 | Chapter 3: Classification** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 



<!-- Start of picture text -->
Confusion matrix CM normalized by row<br>0 0 22 °5 8 43 36 6 225 1 6000 0 ey 0% 0% 0% 0% 1% 1% 0% 4% 0% 0.8<br>mo 0 b 409 37 24 4 44 4 7 212 10 5000 1 0% Egg 1% 0% 0% 1% 0% 0% 3% 0%<br>Yee 27 27 5224 92 73 27 67 36 378 11 ye 0% 0% fia 2% 1% 0% 1% 1% 6% 0%<br>73 5227 2 203 27 40 403 73 4000 3 a 2% BR 0% 3% 0% 1% 7% 1% 0.6<br>PE 12 14 41 «9 12 34 27 347 164 BME 0% 0% 1% 0% gg 0% 1% 0% 6% 3%<br>v i 27 15 30 168 53 75 14 535 60 3000 v EH 0% 0% 1% 3% 1% ay 1% 0% 10% 1% 0.4<br>- (me 30 15 42 3 44 97 3 131 1 FE (me 1% 0% 1% 0% 1% 2% EIR 0% 2% 0%<br>21 10 51 30 49 12 3 bode 210 2000 TH 0% 0% 1% 0% 1% 0% 0% BER 3% 3%<br>Peem 1725 6318 4830 8664 1183 12636 251 #17910 E10744 1000 PHrm 0%0% 0%1% 1%1% 1%1% 2%0% 2%1% 0%0% 0%3% BE6% a1% 0.2<br>0123 45 678 9 0 0123 45 67 8 9 0.0<br>Predicted label Predicted label<br><!-- End of picture text -->



<!-- Start of picture text -->
Errors normalized by row Errors normalized by column<br>WE 0% 0% 6% 1% 2% 12%10%2% (ak 0% 0.6 We 0% 0% 5% 1% 2% 7% 13% 2% 8% 0% 0.5<br>mm 0% 0% 11% 7% 1% 13% 1% 2% CHAS 3% sme 0% 0% 9% 5% 1% 7% 1% 2% 8% 2%<br>0.5<br>PH 4% 4% 0% 12%10% 4% 9% 5% wey 1% FH 15%15% 0% 19%21% 4% 25%11%14% 2% 0.4<br>Fry 2% 2% 13% 0% 0% 22% 3% 4% 8% 04 @ > ee 0% 1% BYE310%12%14%13%<br>BBP 2% 2% 6% 1% 0% 2% 5% 4% bEWA25% SMM 7% 8% 10% 2% 0% 2% 12% 8% 12% 0.3<br>REE 3% 2% 3% 17% 5% 0% 8% 1% BEE 6% MOREE 15% 8% 7% 15% 0% 2B% 4% 19%10%<br>LEEsae 8% 4% 11% 1% 12%27% 0% 1% 0% lHse 17% 8% 10% 1% 12%16% 0% 1% 5% 0% 0.2<br>wae A% 2% 9% 5% 8% 2% 1% 0% 0.2 wae 12% 6% 12% 6% 14% 2% 1% 0% 7%<br>ime 4% 15%11%20% 1% BO% 6% 2% 0% 10% 8 11%18% 1% 21% 9% 3% 0% 8% 0.1<br>RE 3% 2% 4% 8% 14% 4% 0% 2IMUEA 0% 0.1 14% 10% 7% 13% 6% 0% Fee 13% 0% :<br>012 3 45 67 8 9 0.0 0 12 3 4 567 8 9 0.0<br>Predicted label Predicted label<br><!-- End of picture text -->

Now you can see much more clearly the kinds of errors the classifier makes. The column for class 8 is now really bright, which confirms that many images got misclas‐ sified as 8s. In fact this is the most common misclassification for almost all classes. But be careful how you interpret the percentages in this diagram: remember that we’ve excluded the correct predictions. For example, the 36% in row #7, column #9 does _not_ mean that 36% of all images of 7s were misclassified as 9s. It means that 36% of the _errors_ the model made on images of 7s were misclassifications as 9s. In reality, only 3% of images of 7s were misclassified as 9s, as you can see in the diagram on the right in Figure 3-9. 

It is also possible to normalize the confusion matrix by column rather than by row: if you set `normalize="pred"` , you get the diagram on the right in Figure 3-10. For example, you can see that 56% of misclassified 7s are actually 9s. 

Analyzing the confusion matrix often gives you insights into ways to improve your classifier. Looking at these plots, it seems that your efforts should be spent on reduc‐ ing the false 8s. For example, you could try to gather more training data for digits that look like 8s (but are not) so that the classifier can learn to distinguish them from real 8s. Or you could engineer new features that would help the classifier—for example, writing an algorithm to count the number of closed loops (e.g., 8 has two, 6 has one, 5 has none). Or you could preprocess the images (e.g., using Scikit-Image, Pillow, or OpenCV) to make some patterns, such as closed loops, stand out more. 

Analyzing individual errors can also be a good way to gain insights into what your classifier is doing and why it is failing. For example, let’s plot examples of 3s and 5s in a confusion matrix style (Figure 3-11): 

```
cl_a, cl_b='3', '5'
X_aa=X_train[(y_train==cl_a) & (y_train_pred==cl_a)]
X_ab=X_train[(y_train==cl_a) & (y_train_pred==cl_b)]
X_ba=X_train[(y_train==cl_b) & (y_train_pred==cl_a)]
X_bb=X_train[(y_train==cl_b) & (y_train_pred==cl_b)]
[...]  # plot all images in X_aa, X_ab, X_ba, X_bb in a confusion matrix style
```

**124 | Chapter 3: Classification** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 



<!-- Start of picture text -->
33333 35335<br>3332233733535<br>13333333885<br>133333949333<br>2/33 33% 33553<br>S|5R55 §S 4 55555<br>S55S55 55655<br>HS SSESSASSSSS<br>SSSSS sFS eS<br>SISSSS FFT<br>° "<br>Predicted label<br><!-- End of picture text -->

## **Multilabel Classification** 

Until now, each instance has always been assigned to just one class. But in some cases you may want your classifier to output multiple classes for each instance. Consider a face-recognition classifier: what should it do if it recognizes several people in the same picture? It should attach one tag per person it recognizes. Say the classifier has been trained to recognize three faces: Alice, Bob, and Charlie. Then when the classifier is shown a picture of Alice and Charlie, it should output `[True, False, True]` (meaning “Alice yes, Bob no, Charlie yes”). Such a classification system that outputs multiple binary tags is called a _multilabel classification_ system. 

We won’t go into face recognition just yet, but let’s look at a simpler example, just for illustration purposes: 

```
importnumpyasnp
fromsklearn.neighborsimportKNeighborsClassifier
y_train_large= (y_train>='7')
y_train_odd= (y_train.astype('int8') %2==1)
y_multilabel=np.c_[y_train_large, y_train_odd]
```

```
knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```

This code creates a `y_multilabel` array containing two target labels for each digit image: the first indicates whether or not the digit is large (7, 8, or 9), and the second indicates whether or not it is odd. Then the code creates a `KNeighborsClassifier` instance, which supports multilabel classification (not all classifiers do), and trains this model using the multiple targets array. Now you can make a prediction, and notice that it outputs two labels: 

```
>>> knn_clf.predict([some_digit])
array([[False,  True]])
```

And it gets it right! The digit 5 is indeed not large ( `False` ) and odd ( `True` ). 

There are many ways to evaluate a multilabel classifier, and selecting the right metric really depends on your project. One approach is to measure the F1 score for each individual label (or any other binary classifier metric discussed earlier), then simply compute the average score. The following code computes the average F1 score across all labels: 

```
>>> y_train_knn_pred=cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
>>> f1_score(y_multilabel, y_train_knn_pred, average="macro")
0.976410265560605
```

This approach assumes that all labels are equally important, which may not be the case. In particular, if you have many more pictures of Alice than of Bob or Charlie, you may want to give more weight to the classifier’s score on pictures of Alice. One 

**126 | Chapter 3: Classification** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 

simple option is to give each label a weight equal to its _support_ (i.e., the number of instances with that target label). To do this, simply set `average="weighted"` when calling the `f1_score()` function.<sup>6</sup> 

If you wish to use a classifier that does not natively support multilabel classification, such as `SVC` , one possible strategy is to train one model per label. However, this strategy may have a hard time capturing the dependencies between the labels. For example, a large digit (7, 8, or 9) is twice more likely to be odd than even, but the classifier for the “odd” label does not know what the classifier for the “large” label predicted. To solve this issue, the models can be organized in a chain: when a model makes a prediction, it uses the input features plus all the predictions of the models that come before it in the chain. 

The good news is that Scikit-Learn has a class called `ClassifierChain` that does just that! By default it will use the true labels for training, feeding each model the appropriate labels depending on their position in the chain. But if you set the `cv` hyperparameter, it will use cross-validation to get “clean” (out-of-sample) predictions from each trained model for every instance in the training set, and these predictions will then be used to train all the models later in the chain. Here’s an example showing how to create and train a `ClassifierChain` using the cross-validation strategy. As earlier, we’ll just use the first 2,000 images in the training set to speed things up: 

```
fromsklearn.multioutputimportClassifierChain
```

```
chain_clf=ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])
```

Now we can use this `ClassifierChain` to make predictions: 

```
>>> chain_clf.predict([some_digit])
array([[0., 1.]])
```

## **Multioutput Classification** 

The last type of classification task we’ll discuss here is called _multioutput–multiclass classification_ (or just _multioutput classification_ ). It is a generalization of multilabel classification where each label can be multiclass (i.e., it can have more than two possible values). 

To illustrate this, let’s build a system that removes noise from images. It will take as input a noisy digit image, and it will (hopefully) output a clean digit image, represented as an array of pixel intensities, just like the MNIST images. Notice that the classifier’s output is multilabel (one label per pixel) and each label can have 

> 6 Scikit-Learn offers a few other averaging options and multilabel classifier metrics; see the documentation for more details. 

**Multioutput Classification | 127** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 







- **a.** Download examples of spam and ham from Apache SpamAssassin’s public datasets. 

- **b.** Unzip the datasets and familiarize yourself with the data format. 

- **c.** Split the data into a training set and a test set. 

- **d.** Write a data preparation pipeline to convert each email into a feature vector. Your preparation pipeline should transform an email into a (sparse) vector that indicates the presence or absence of each possible word. For example, if all emails only ever contain four words, “Hello”, “how”, “are”, “you”, then the email “Hello you Hello Hello you” would be converted into a vector [1, 0, 0, 1] (meaning [“Hello” is present, “how” is absent, “are” is absent, “you” is present]), or [3, 0, 0, 2] if you prefer to count the number of occurrences of each word. 

You may want to add hyperparameters to your preparation pipeline to control whether or not to strip off email headers, convert each email to lowercase, remove punctuation, replace all URLs with “URL”, replace all numbers with “NUMBER”, or even perform _stemming_ (i.e., trim off word endings; there are Python libraries available to do this). 

- **e.** Finally, try out several classifiers and see if you can build a great spam classi‐ fier, with both high recall and high precision. 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

**130 | Chapter 3: Classification** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:46. 

