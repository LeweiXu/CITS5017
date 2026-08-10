# **CHAPTER 7 Ensemble Learning and Random Forests** 

Suppose you pose a complex question to thousands of random people, then aggregate their answers. In many cases you will find that this aggregated answer is better than an expert’s answer. This is called the _wisdom of the crowd_ . Similarly, if you aggregate the predictions of a group of predictors (such as classifiers or regressors), you will often get better predictions than with the best individual predictor. A group of predictors is called an _ensemble_ ; thus, this technique is called _ensemble learning_ , and an ensemble learning algorithm is called an _ensemble method_ . 

As an example of an ensemble method, you can train a group of decision tree classifiers, each on a different random subset of the training set. You can then obtain the predictions of all the individual trees, and the class that gets the most votes is the ensemble’s prediction (see the last exercise in Chapter 6). Such an ensemble of decision trees is called a _random forest_ , and despite its simplicity, this is one of the most powerful machine learning algorithms available today. 

As discussed in Chapter 2, you will often use ensemble methods near the end of a project, once you have already built a few good predictors, to combine them into an even better predictor. In fact, the winning solutions in machine learning competitions often involve several ensemble methods—most famously in the Netflix Prize competition. 

In this chapter we will examine the most popular ensemble methods, including voting classifiers, bagging and pasting ensembles, random forests, and boosting, and stacking ensembles. 

**211** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 



<!-- Start of picture text -->
Logistic SVM Random Other...<br>regression classifier forest<br>classifier<br>, on on Diverse predictors<br>ee<br>[sages<br>Ensemble's prediction<br>(e.g., majority vote)<br>L 1 2 ) Predictions<br>—<br>“ New instance<br><!-- End of picture text -->



<!-- Start of picture text -->
0.564<br>fo) #B *<br>© Bo NG a I SSa |<br>© 0.48 4) vir<br>Sa 0.46 4" 1 A<br>pnd Wee828<br>0.42 a 70"<br>0 2000 4000 6000 8000 10000<br>Number of coin tosses<br><!-- End of picture text -->



Ensemble methods work best when the predictors are as independ‐ ent from one another as possible. One way to get diverse classifiers is to train them using very different algorithms. This increases the chance that they will make very different types of errors, improving the ensemble’s accuracy. 

Scikit-Learn provides a `VotingClassifier` class that’s quite easy to use: just give it a list of name/predictor pairs, and use it like a normal classifier. Let’s try it on the moons dataset (introduced in Chapter 5). We will load and split the moons dataset into a training set and a test set, then we’ll create and train a voting classifier composed of three diverse classifiers: 

```
fromsklearn.datasetsimportmake_moons
fromsklearn.ensembleimportRandomForestClassifier, VotingClassifier
fromsklearn.linear_modelimportLogisticRegression
fromsklearn.model_selectionimporttrain_test_split
fromsklearn.svmimportSVC
X, y=make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=42)
voting_clf=VotingClassifier(
estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)
```

When you fit a `VotingClassifier` , it clones every estimator and fits the clones. The original estimators are available via the `estimators` attribute, while the fitted clones are available via the `estimators_` attribute. If you prefer a dict rather than a list, you can use `named_estimators` or `named_estimators_` instead. To begin, let’s look at each fitted classifier’s accuracy on the test set: 

```
>>> forname, clfinvoting_clf.named_estimators_.items():
... print(name, "=", clf.score(X_test, y_test))
...
lr = 0.864
rf = 0.896
svc = 0.896
```

When you call the voting classifier’s `predict()` method, it performs hard voting. For example, the voting classifier predicts class 1 for the first instance of the test set, because two out of three classifiers predict that class: 

#### **214 | Chapter 7: Ensemble Learning and Random Forests** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 

```
>>> voting_clf.predict(X_test[:1])
array([1])
>>> [clf.predict(X_test[:1]) forclfinvoting_clf.estimators_]
[array([1]), array([1]), array([0])]
```

Now let’s look at the performance of the voting classifier on the test set: 

```
>>> voting_clf.score(X_test, y_test)
0.912
```

There you have it! The voting classifier outperforms all the individual classifiers. 

If all classifiers are able to estimate class probabilities (i.e., if they all have a `predict_proba()` method), then you can tell Scikit-Learn to predict the class with the highest class probability, averaged over all the individual classifiers. This is called _soft voting_ . It often achieves higher performance than hard voting because it gives more weight to highly confident votes. All you need to do is set the voting classi‐ fier’s `voting` hyperparameter to `"soft"` , and ensure that all classifiers can estimate class probabilities. This is not the case for the `SVC` class by default, so you need to set its `probability` hyperparameter to `True` (this will make the `SVC` class use cross-validation to estimate class probabilities, slowing down training, and it will add a `predict_proba()` method). Let’s try that: 

```
>>> voting_clf.voting="soft"
>>> voting_clf.named_estimators["svc"].probability=True
>>> voting_clf.fit(X_train, y_train)
>>> voting_clf.score(X_test, y_test)
0.92
```

We reach 92% accuracy simply by using soft voting—not bad! 

## **Bagging and Pasting** 

One way to get a diverse set of classifiers is to use very different training algorithms, as just discussed. Another approach is to use the same training algorithm for every predictor but train them on different random subsets of the training set. When sampling is performed _with_ replacement,<sup>1</sup> this method is called _bagging_<sup>2</sup> (short for _bootstrap aggregating_<sup>3</sup> ). When sampling is performed _without_ replacement, it is called _pasting_ .<sup>4</sup> 

> 1 Imagine picking a card randomly from a deck of cards, writing it down, then placing it back in the deck before picking the next card: the same card could be sampled multiple times. 

> 2 Leo Breiman, “Bagging Predictors”, _Machine Learning_ 24, no. 2 (1996): 123–140. 

> 3 In statistics, resampling with replacement is called _bootstrapping_ . 

> 4 Leo Breiman, “Pasting Small Votes for Classification in Large Databases and On-Line”, _Machine Learning_ 36, no. 1–2 (1999): 85–103. 

**Bagging and Pasting | 215** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 



<!-- Start of picture text -->
© 6 Predictors<br>(e.g., classifiers)<br>Training<br>° oe eote 8 oe, 0° 8o°<br>e e ee e<br>Random sampling<br>(with replacement=bootstrap)<br>or ge<br>fe) oe<br>eo ~ Training set<br><!-- End of picture text -->

### **Bagging and Pasting in Scikit-Learn** 

Scikit-Learn offers a simple API for both bagging and pasting: `BaggingClassifier` class (or `BaggingRegressor` for regression). The following code trains an ensemble of 500 decision tree classifiers:<sup>6</sup> each is trained on 100 training instances randomly sampled from the training set with replacement (this is an example of bagging, but if you want to use pasting instead, just set `bootstrap=False` ). The `n_jobs` parameter tells Scikit-Learn the number of CPU cores to use for training and predictions, and `–1` tells Scikit-Learn to use all available cores: 

```
fromsklearn.ensembleimportBaggingClassifier
fromsklearn.treeimportDecisionTreeClassifier
```

```
bag_clf=BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
max_samples=100, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
```



A `BaggingClassifier` automatically performs soft voting instead of hard voting if the base classifier can estimate class probabilities (i.e., if it has a `predict_proba()` method), which is the case with decision tree classifiers. 

Figure 7-5 compares the decision boundary of a single decision tree with the decision boundary of a bagging ensemble of 500 trees (from the preceding code), both trained on the moons dataset. As you can see, the ensemble’s predictions will likely generalize much better than the single decision tree’s predictions: the ensemble has a compara‐ ble bias but a smaller variance (it makes roughly the same number of errors on the training set, but the decision boundary is less irregular). 

Bagging introduces a bit more diversity in the subsets that each predictor is trained on, so bagging ends up with a slightly higher bias than pasting; but the extra diversity also means that the predictors end up being less correlated, so the ensemble’s variance is reduced. Overall, bagging often results in better models, which explains why it’s generally preferred. But if you have spare time and CPU power, you can use cross-validation to evaluate both bagging and pasting and select the one that works best. 

> 6 `max_samples` can alternatively be set to a float between 0.0 and 1.0, in which case the max number of sampled instances is equal to the size of the training set times `max_samples` . 

**Bagging and Pasting | 217** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 



<!-- Start of picture text -->
1.5 Decision Tree Decision Trees with Bagging<br>e r<br>ee e *e es°of in U ee e “e efof A,e O<br>DAs A eee a es SA a.<br>& > e 68h & e @el aa e. ° ae e e @ejyaa<br>an ee Pa ee A ae ee “ae ty cee fa ae<br>0.0 Fok 4 hte ary Soete e Le rs<br>2 eoh, 4 ma! ee nx nN 4a 2 ook, 4 mAa . os aa<br>ha = a n Aa 8 yp n<br>os rN Ya are A Agee res aa<br>e a SAA yer 4 e a fA i ye 4<br>“OTs 10-05 00 ‘05 a #210 ‘a #15a «20 -15 -10 -05 00 ‘O05 wa10 15a 2.0<br>X1 X1<br><!-- End of picture text -->

According to this OOB evaluation, this `BaggingClassifier` is likely to achieve about 89.6% accuracy on the test set. Let’s verify this: 

```
>>> fromsklearn.metricsimportaccuracy_score
>>> y_pred=bag_clf.predict(X_test)
>>> accuracy_score(y_test, y_pred)
0.92
```

We get 92% accuracy on the test. The OOB evaluation was a bit too pessimistic, just over 2% too low. 

The OOB decision function for each training instance is also available through the `oob_decision_function_` attribute. Since the base estimator has a `predict_proba()` method, the decision function returns the class probabilities for each training instance. For example, the OOB evaluation estimates that the first training instance has a 67.6% probability of belonging to the positive class and a 32.4% probability of belonging to the negative class: 

```
>>> bag_clf.oob_decision_function_[:3]  # probas for the first 3 instances
array([[0.32352941, 0.67647059],
       [0.3375    , 0.6625    ],
       [1.        , 0.        ]])
```

### **Random Patches and Random Subspaces** 

The `BaggingClassifier` class supports sampling the features as well. Sampling is controlled by two hyperparameters: `max_features` and `bootstrap_features` . They work the same way as `max_samples` and `bootstrap` , but for feature sampling instead of instance sampling. Thus, each predictor will be trained on a random subset of the input features. 

This technique is particularly useful when you are dealing with high-dimensional inputs (such as images), as it can considerably speed up training. Sampling both training instances and features is called the _random patches_ method.<sup>8</sup> Keeping all training instances (by setting `bootstrap=False` and `max_samples=1.0` ) but sampling features (by setting `bootstrap_features` to `True` and/or `max_features` to a value smaller than `1.0` ) is called the _random subspaces_ method.<sup>9</sup> 

Sampling features results in even more predictor diversity, trading a bit more bias for a lower variance. 

> 8 Gilles Louppe and Pierre Geurts, “Ensembles on Random Patches”, _Lecture Notes in Computer Science_ 7523 (2012): 346–361. 

> 9 Tin Kam Ho, “The Random Subspace Method for Constructing Decision Forests”, _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 20, no. 8 (1998): 832–844. 

**Bagging and Pasting | 219** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 

## **Random Forests** 

As we have discussed, a random forest<sup>10</sup> is an ensemble of decision trees, generally trained via the bagging method (or sometimes pasting), typically with `max_samples` set to the size of the training set. Instead of building a `BaggingClassifier` and passing it a `DecisionTreeClassifier` , you can use the `RandomForestClassifier` class, which is more convenient and optimized for decision trees<sup>11</sup> (similarly, there is a `RandomForestRegressor` class for regression tasks). The following code trains a random forest classifier with 500 trees, each limited to maximum 16 leaf nodes, using all available CPU cores: 

```
fromsklearn.ensembleimportRandomForestClassifier
```

```
rnd_clf=RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
n_jobs=-1, random_state=42)
```

```
rnd_clf.fit(X_train, y_train)
```

```
y_pred_rf=rnd_clf.predict(X_test)
```

With a few exceptions, a `RandomForestClassifier` has all the hyperparameters of a `DecisionTreeClassifier` (to control how trees are grown), plus all the hyperpara‐ meters of a `BaggingClassifier` to control the ensemble itself. 

The `RandomForestClassifier` class introduces extra randomness when growing trees: instead of searching for the very best feature when splitting a node (see Chap‐ ter 6), it searches for the best feature among a random subset of features. By default, it samples n features (where _n_ is the total number of features). The algorithm results in greater tree diversity, which (again) trades a higher bias for a lower variance, generally yielding an overall better model. So, the following `BaggingClassifier` is equivalent to the previous `RandomForestClassifier` : 

```
bag_clf=BaggingClassifier(
DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
n_estimators=500, n_jobs=-1, random_state=42)
```

### **Extra-Trees** 

When you are growing a tree in a random forest, at each node only a random subset of the features is considered for splitting (as discussed earlier). It is possible to make trees even more random by also using random thresholds for each feature rather than 

> 10 Tin Kam Ho, “Random Decision Forests”, _Proceedings of the Third International Conference on Document Analysis and Recognition_ 1 (1995): 278. 

> 11 The `BaggingClassifier` class remains useful if you want a bag of something other than decision trees. 

**220 | Chapter 7: Ensemble Learning and Random Forests** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 

searching for the best possible thresholds (like regular decision trees do). For this, simply set `splitter="random"` when creating a `DecisionTreeClassifier` . 

A forest of such extremely random trees is called an _extremely randomized trees_<sup>12</sup> (or _extra-trees_ for short) ensemble. Once again, this technique trades more bias for a lower variance. It also makes extra-trees classifiers much faster to train than regular random forests, because finding the best possible threshold for each feature at every node is one of the most time-consuming tasks of growing a tree. 

You can create an extra-trees classifier using Scikit-Learn’s `ExtraTreesClassifier` class. Its API is identical to the `RandomForestClassifier` class, except `bootstrap` defaults to `False` . Similarly, the `ExtraTreesRegressor` class has the same API as the `RandomForestRegressor` class, except `bootstrap` defaults to `False` . 



It is hard to tell in advance whether a `RandomForestClassifier` will perform better or worse than an `ExtraTreesClassifier` . Gen‐ erally, the only way to know is to try both and compare them using cross-validation. 

### **Feature Importance** 

Yet another great quality of random forests is that they make it easy to measure the relative importance of each feature. Scikit-Learn measures a feature’s importance by looking at how much the tree nodes that use that feature reduce impurity on average, across all trees in the forest. More precisely, it is a weighted average, where each node’s weight is equal to the number of training samples that are associated with it (see Chapter 6). 

Scikit-Learn computes this score automatically for each feature after training, then it scales the results so that the sum of all importances is equal to 1. You can access the result using the `feature_importances_` variable. For example, the following code trains a `RandomForestClassifier` on the iris dataset (introduced in Chapter 4) and outputs each feature’s importance. It seems that the most important features are the petal length (44%) and width (42%), while sepal length and width are rather unimportant in comparison (11% and 2%, respectively): 

```
>>> fromsklearn.datasetsimportload_iris
```

```
>>> iris=load_iris(as_frame=True)
```

```
>>> rnd_clf=RandomForestClassifier(n_estimators=500, random_state=42)
>>> rnd_clf.fit(iris.data, iris.target)
```

```
>>> forscore, nameinzip(rnd_clf.feature_importances_, iris.data.columns):
... print(round(score, 2), name)
```

```
...
```

12 Pierre Geurts et al., “Extremely Randomized Trees”, _Machine Learning_ 63, no. 1 (2006): 3–42. 

**Random Forests | 221** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 



<!-- Start of picture text -->
ra Not important:<br><!-- End of picture text -->



<!-- Start of picture text -->
1 iN. 0 OF OQ, Sf —— af<br>5 WY e‘.e<br>o (CO e @ o (lO e °@<br>© ° ° CG ° e 0 e e a ° e<br>| [oJe) oOo | ec eo | foe) ee ee eo<br><!-- End of picture text -->



<!-- Start of picture text -->
1s learning_rate =1 learning_rate = 0.5<br>, f ee *e ve A, e si<br>1.0 | waen Te e ? fe i)ey)<br>; y he seat 2.6 i<br>0.0 id e f/ 7 ieee<br>(<br>-0.5 {4 ' , e |<br>as L<br>-1.0-15 -10 -05 00 O05 1034) 15 20 -15 -10 J-05 00 O05 10 15 20<br>X1 X1<br><!-- End of picture text -->



() aC) A) () 

to zero. However, if it is most often wrong (i.e., less accurate than random guessing), then its weight will be negative. 

_Equation 7-2. Predictor weight_ 



Next, the AdaBoost algorithm updates the instance weights, using Equation 7-3, which boosts the weights of the misclassified instances. 

_Equation 7-3. Weight update rule_ 



m i Then all the instance weights are normalized (i.e., divided by ∑ i = 1 w ). 

Finally, a new predictor is trained using the updated weights, and the whole process is repeated: the new predictor’s weight is computed, the instance weights are updated, then another predictor is trained, and so on. The algorithm stops when the desired number of predictors is reached, or when a perfect predictor is found. 

To make predictions, AdaBoost simply computes the predictions of all the predictors and weighs them using the predictor weights _αj_ . The predicted class is the one that receives the majority of weighted votes (see Equation 7-4). 

_Equation 7-4. AdaBoost predictions_ 



Scikit-Learn uses a multiclass version of AdaBoost called _SAMME_<sup>16</sup> (which stands for _Stagewise Additive Modeling using a Multiclass Exponential loss function_ ). When there are just two classes, SAMME is equivalent to AdaBoost. If the predictors can estimate class probabilities (i.e., if they have a `predict_proba()` method), Scikit-Learn can use a variant of SAMME called _SAMME.R_ (the _R_ stands for “Real”), which relies on class probabilities rather than predictions and generally performs better. 

> 16 For more details, see Ji Zhu et al., “Multi-Class AdaBoost”, _Statistics and Its Interface_ 2, no. 3 (2009): 349–360. 

**Boosting | 225** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 

The following code trains an AdaBoost classifier based on 30 _decision stumps_ using Scikit-Learn’s `AdaBoostClassifier` class (as you might expect, there is also an `AdaBoostRegressor` class). A decision stump is a decision tree with `max_depth=1` —in other words, a tree composed of a single decision node plus two leaf nodes. This is the default base estimator for the `AdaBoostClassifier` class: 

```
fromsklearn.ensembleimportAdaBoostClassifier
```

```
ada_clf=AdaBoostClassifier(
DecisionTreeClassifier(max_depth=1), n_estimators=30,
learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
```



If your AdaBoost ensemble is overfitting the training set, you can try reducing the number of estimators or more strongly regulariz‐ ing the base estimator. 

### **Gradient Boosting** 

Another very popular boosting algorithm is _gradient boosting_ .<sup>17</sup> Just like AdaBoost, gradient boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the _residual errors_ made by the previous predictor. 

Let’s go through a simple regression example, using decision trees as the base predic‐ tors; this is called _gradient tree boosting_ , or _gradient boosted regression trees_ (GBRT). First, let’s generate a noisy quadratic dataset and fit a `DecisionTreeRegressor` to it: 

```
importnumpyasnp
fromsklearn.treeimportDecisionTreeRegressor
np.random.seed(42)
X=np.random.rand(100, 1) -0.5
y=3*X[:, 0] **2+0.05*np.random.randn(100)  # y = 3x² + Gaussian noise
```

```
tree_reg1=DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
```

- 17 Gradient boosting was first introduced in Leo Breiman’s 1997 paper “Arcing the Edge” and was further developed in the 1999 paper “Greedy Function Approximation: A Gradient Boosting Machine” by Jerome H. Friedman. 

#### **226 | Chapter 7: Ensemble Learning and Random Forests** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 

Next, we’ll train a second `DecisionTreeRegressor` on the residual errors made by the first predictor: 

```
y2=y-tree_reg1.predict(X)
tree_reg2=DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)
```

And then we’ll train a third regressor on the residual errors made by the second predictor: 

```
y3=y2-tree_reg2.predict(X)
tree_reg3=DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)
```

Now we have an ensemble containing three trees. It can make predictions on a new instance simply by adding up the predictions of all the trees: 

```
>>> X_new=np.array([[-0.4], [0.], [0.5]])
>>> sum(tree.predict(X_new) fortreein (tree_reg1, tree_reg2, tree_reg3))
array([0.49484029, 0.04021166, 0.75026781])
```

Figure 7-9 represents the predictions of these three trees in the left column, and the ensemble’s predictions in the right column. In the first row, the ensemble has just one tree, so its predictions are exactly the same as the first tree’s predictions. In the second row, a new tree is trained on the residual errors of the first tree. On the right you can see that the ensemble’s predictions are equal to the sum of the predictions of the first two trees. Similarly, in the third row another tree is trained on the residual errors of the second tree. You can see that the ensemble’s predictions gradually get better as trees are added to the ensemble. 

You can use Scikit-Learn’s `GradientBoostingRegressor` class to train GBRT ensem‐ bles more easily (there’s also a `GradientBoostingClassifier` class for classifica‐ tion). Much like the `RandomForestRegressor` class, it has hyperparameters to control the growth of decision trees (e.g., `max_depth` , `min_samples_leaf` ), as well as hyperparameters to control the ensemble training, such as the number of trees ( `n_estimators` ). The following code creates the same ensemble as the previous one: 

```
fromsklearn.ensembleimportGradientBoostingRegressor
```

```
gbrt=GradientBoostingRegressor(max_depth=2, n_estimators=3,
learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
```

**Boosting | 227** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 



<!-- Start of picture text -->
os Residuals and tree predictions os Ensemble predictions<br>* * Training set . ° * * Training set . .<br>06 . — h(x) | 06 . — h(xr) = hi(x1) .<br>. .<br>0.4 .° 0.4 .°<br>y oe : ee :<br>02 ar ee 02 aan es<br>. oc aot ; ot wt<br>0.0 0.0<br>0.2 -0.2<br>-0.4 -0.2 0.0 02 0.4 -0.4 -0.2 0.0 0.2 0.4<br>0.6 0.8<br>+ Residuals: y — hy(x1) ° — A(x) = hi (x1) + h2(x1) .<br>0.4 — ho) 0.6 . o<br>+<br>0.2 Ht + 1 oa .°<br>y + + + ee<br>0.0 4% + Ft7 + Ht at 0.2 =: . “og e<br>hoot, + ot 2 .<br>4 e e Pd *<br>-0.2 0.0<br>-0.4 -0.2<br>-0.4 -0.2 0.0 0.2 0.4 -0.4 -0.2 0.0 0.2 0.4<br>0.6 0.8<br>+ — Residuals: y — hy(x1)— h2(x1) SS — Alxy) = hi (x1) + ho(xr) + h3(x1)<br>0.4 — hs(x1) 0.6 . °<br>“<br>0.2 0.4 .°<br>y + + + rl oot<br>0.0 ot te yt+ +, + +og thy +# 0.2 ce %. -<br>a .<br>~~ ° e o +<br>-0.2 0.0<br>-0.4 -0.2<br>-0.4 -0.2 0.0 0.2 0.4 -0.4 -0.2 0.0 0.2 0.4<br>X1 X1<br><!-- End of picture text -->



<!-- Start of picture text -->
oa learning_rate=1.0, n_estimators=3 learning_rate=0.05, n_estimators=92<br>074% — Ensemble predictions<br>0.6 . e<br>*<br>0.5<br>y 0.4 .<br>See<br>0.3 C ‘|<br>0.2 o e : 0<br>o.1 *, ° . ... 4<br>0.0<br>-0.1<br>-0.4 -0.2 0.0 0.2 0.4 -0.4 -0.2 0.0 0.2 0.4<br>X1 X1<br><!-- End of picture text -->

The `GradientBoostingRegressor` class also supports a `subsample` hyperparameter, which specifies the fraction of training instances to be used for training each tree. For example, if `subsample=0.25` , then each tree is trained on 25% of the training instances, selected randomly. As you can probably guess by now, this technique trades a higher bias for a lower variance. It also speeds up training considerably. This is called _stochastic gradient boosting_ . 

### **Histogram-Based Gradient Boosting** 

Scikit-Learn also provides another GBRT implementation, optimized for large data‐ sets: _histogram-based gradient boosting_ (HGB). It works by binning the input features, replacing them with integers. The number of bins is controlled by the `max_bins` hyperparameter, which defaults to 255 and cannot be set any higher than this. Bin‐ ning can greatly reduce the number of possible thresholds that the training algorithm needs to evaluate. Moreover, working with integers makes it possible to use faster and more memory-efficient data structures. And the way the bins are built removes the need for sorting the features when training each tree. 

As a result, this implementation has a computational complexity of _O_ ( _b_ × _m_ ) instead of _O_ ( _n_ × _m_ ×log( _m_ )), where _b_ is the number of bins, _m_ is the number of training instances, and _n_ is the number of features. In practice, this means that HGB can train hundreds of times faster than regular GBRT on large datasets. However, binning causes a precision loss, which acts as a regularizer: depending on the dataset, this may help reduce overfitting, or it may cause underfitting. 

Scikit-Learn provides two classes for HGB: `HistGradientBoostingRegressor` and `HistGradientBoostingClassifier` . They’re similar to `GradientBoostingRegressor` and `GradientBoostingClassifier` , with a few notable differences: 

- Early stopping is automatically activated if the number of instances is greater than 10,000. You can turn early stopping always on or always off by setting the `early_stopping` hyperparameter to `True` or `False` . 

- Subsampling is not supported. 

- `n_estimators` is renamed to `max_iter` . 

- The only decision tree hyperparameters that can be tweaked are `max_leaf_nodes` , `min_samples_leaf` , and `max_depth` . 

The HGB classes also have two nice features: they support both categorical features and missing values. This simplifies preprocessing quite a bit. However, the categorical features must be represented as integers ranging from 0 to a number lower than `max_bins` . You can use an `OrdinalEncoder` for this. For example, here’s how to build and train a complete pipeline for the California housing dataset introduced in Chapter 2: 

**230 | Chapter 7: Ensemble Learning and Random Forests** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 

```
fromsklearn.pipelineimportmake_pipeline
fromsklearn.composeimportmake_column_transformer
fromsklearn.ensembleimportHistGradientBoostingRegressor
fromsklearn.preprocessingimportOrdinalEncoder
```

```
hgb_reg=make_pipeline(
make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
remainder="passthrough"),
HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
)
hgb_reg.fit(housing, housing_labels)
```

The whole pipeline is just as short as the imports! No need for an imputer, scaler, or a one-hot encoder, so it’s really convenient. Note that `categorical_features` must be set to the categorical column indices (or a Boolean array). Without any hyperparameter tuning, this model yields an RMSE of about 47,600, which is not too bad. 



Several other optimized implementations of gradient boosting are available in the Python ML ecosystem: in particular, XGBoost, Cat‐ Boost, and LightGBM. These libraries have been around for several years. They are all specialized for gradient boosting, their APIs are very similar to Scikit-Learn’s, and they provide many additional features, including GPU acceleration; you should definitely check them out! Moreover, the TensorFlow Random Forests library pro‐ vides optimized implementations of a variety of random forest algorithms, including plain random forests, extra-trees, GBRT, and several more. 

## **Stacking** 

The last ensemble method we will discuss in this chapter is called _stacking_ (short for _stacked generalization_ ).<sup>18</sup> It is based on a simple idea: instead of using trivial functions (such as hard voting) to aggregate the predictions of all predictors in an ensemble, why don’t we train a model to perform this aggregation? Figure 7-11 shows such an ensemble performing a regression task on a new instance. Each of the bottom three predictors predicts a different value (3.1, 2.7, and 2.9), and then the final predictor (called a _blender_ , or a _meta learner_ ) takes these predictions as inputs and makes the final prediction (3.0). 

> 18 David H. Wolpert, “Stacked Generalization”, _Neural Networks_ 5, no. 2 (1992): 241–259. 

**Stacking | 231** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 



<!-- Start of picture text -->
3.0<br>[oa Blending<br>(31) 2] (29) Predictions<br>eo)<br>On Predict<br>xX New instance<br><!-- End of picture text -->



<!-- Start of picture text -->
Blending<br>Train<br>to combine predictions<br>Z Blending training set «---------, '<br>{~~ —. f%& Ff] Cross-validation |<br>a ° a predictions !<br>'<br>Predictors<br>0° | Copy the targets}<br>=) 5g | Trainingset - ----------------<br><!-- End of picture text -->



<!-- Start of picture text -->
29<br>Layer3[ea<br>a1°]> Sas<br>EpX,<br>“ne New instance<br><!-- End of picture text -->

If you evaluate this stacking model on the test set, you will find 92.8% accuracy, which is a bit better than the voting classifier using soft voting, which got 92%. 

In conclusion, ensemble methods are versatile, powerful, and fairly simple to use. Random forests, AdaBoost, and GBRT are among the first models you should test for most machine learning tasks, and they particularly shine with heterogeneous tabular data. Moreover, as they require very little preprocessing, they’re great for getting a prototype up and running quickly. Lastly, ensemble methods like voting classifiers and stacking classifiers can help push your system’s performance to its limits. 

## **Exercises** 

**1.** If you have trained five different models on the exact same training data, and they all achieve 95% precision, is there any chance that you can combine these models to get better results? If so, how? If not, why? 

**2.** What is the difference between hard and soft voting classifiers? 

**3.** Is it possible to speed up training of a bagging ensemble by distributing it across multiple servers? What about pasting ensembles, boosting ensembles, random forests, or stacking ensembles? 

**4.** What is the benefit of out-of-bag evaluation? 

**5.** What makes extra-trees ensembles more random than regular random forests? How can this extra randomness help? Are extra-trees classifiers slower or faster than regular random forests? 

**6.** If your AdaBoost ensemble underfits the training data, which hyperparameters should you tweak, and how? 

**7.** If your gradient boosting ensemble overfits the training set, should you increase or decrease the learning rate? 

**8.** Load the MNIST dataset (introduced in Chapter 3), and split it into a training set, a validation set, and a test set (e.g., use 50,000 instances for training, 10,000 for validation, and 10,000 for testing). Then train various classifiers, such as a random forest classifier, an extra-trees classifier, and an SVM classifier. Next, try to combine them into an ensemble that outperforms each individual classifier on the validation set, using soft or hard voting. Once you have found one, try it on the test set. How much better does it perform compared to the individual classifiers? 

**9.** Run the individual classifiers from the previous exercise to make predictions on the validation set, and create a new training set with the resulting predictions: each training instance is a vector containing the set of predictions from all your classifiers for an image, and the target is the image’s class. Train a classifier on this new training set. Congratulations—you have just trained a blender, and 

**Exercises | 235** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 

together with the classifiers it forms a stacking ensemble! Now evaluate the ensemble on the test set. For each image in the test set, make predictions with all your classifiers, then feed the predictions to the blender to get the ensemble’s pre‐ dictions. How does it compare to the voting classifier you trained earlier? Now try again using a `StackingClassifier` instead. Do you get better performance? If so, why? 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

#### **236 | Chapter 7: Ensemble Learning and Random Forests** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:17. 

