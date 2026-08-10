# **CHAPTER 6 Decision Trees** 

_Decision trees_ are versatile machine learning algorithms that can perform both clas‐ sification and regression tasks, and even multioutput tasks. They are powerful algo‐ rithms, capable of fitting complex datasets. For example, in Chapter 2 you trained a `DecisionTreeRegressor` model on the California housing dataset, fitting it perfectly (actually, overfitting it). 

Decision trees are also the fundamental components of random forests (see Chap‐ ter 7), which are among the most powerful machine learning algorithms available today. 

In this chapter we will start by discussing how to train, visualize, and make predic‐ tions with decision trees. Then we will go through the CART training algorithm used by Scikit-Learn, and we will explore how to regularize trees and use them for regression tasks. Finally, we will discuss some of the limitations of decision trees. 

## **Training and Visualizing a Decision Tree** 

To understand decision trees, let’s build one and take a look at how it makes predic‐ tions. The following code trains a `DecisionTreeClassifier` on the iris dataset (see Chapter 4): 

```
fromsklearn.datasetsimportload_iris
fromsklearn.treeimportDecisionTreeClassifier
iris=load_iris(as_frame=True)
X_iris=iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris=iris.target
```

```
tree_clf=DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)
```

**195** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:10. 



<!-- Start of picture text -->
petal length (cm) <= 2.45<br>gini = 0.667 Root node<br>samples = 150 & split node<br>value = [50, 50, 50]<br>class = setosa<br>True False<br>petal width (cm) <= 1.75<br>Leaf gini = 0.5 Split<br>node samples = 100 node<br>value = [0, 50, 50]<br>class = versicolor<br>True False<br>Leaf Leaf<br>node node<br><!-- End of picture text -->



<!-- Start of picture text -->
|<br><!-- End of picture text -->

| 





<!-- Start of picture text -->
3.0 5<br>e Iris setosa :<br>2.5 m_ Iris versicolor : a AA<br>€ 4 Irisrervirginica :> Baas “naaA Fes A<br>5 2.0 AAA ~+4<br>= —— Reet ~ bb DADA<br>; rm. |<br>$ 1.0 Depth=0 a | :<br>a ;<br>05<br>; a ; :(Depth=2)<br>00 © :<br>0.0<br>0 1 2 3 4 5 6 7<br>Petal length (cm)<br><!-- End of picture text -->



### **Model Interpretation: White Box Versus Black Box** 

Decision trees are intuitive, and their decisions are easy to interpret. Such models are often called _white box models_ . In contrast, as you will see, random forests and neural networks are generally considered _black box models_ . They make great predictions, and you can easily check the calculations that they performed to make these predic‐ tions; nevertheless, it is usually hard to explain in simple terms why the predictions were made. For example, if a neural network says that a particular person appears in a picture, it is hard to know what contributed to this prediction: Did the model recog‐ nize that person’s eyes? Their mouth? Their nose? Their shoes? Or even the couch that they were sitting on? Conversely, decision trees provide nice, simple classification rules that can even be applied manually if need be (e.g., for flower classification). The field of _interpretable ML_ aims at creating ML systems that can explain their decisions in a way humans can understand. This is important in many domains—for example, to ensure the system does not make unfair decisions. 

## **Estimating Class Probabilities** 

A decision tree can also estimate the probability that an instance belongs to a partic‐ ular class _k_ . First it traverses the tree to find the leaf node for this instance, and then it returns the ratio of training instances of class _k_ in this node. For example, suppose you have found a flower whose petals are 5 cm long and 1.5 cm wide. The corresponding leaf node is the depth-2 left node, so the decision tree outputs the following probabilities: 0% for _Iris setosa_ (0/54), 90.7% for _Iris versicolor_ (49/54), and 9.3% for _Iris virginica_ (5/54). And if you ask it to predict the class, it outputs _Iris versicolor_ (class 1) because it has the highest probability. Let’s check this: 

```
>>> tree_clf.predict_proba([[5, 1.5]]).round(3)
array([[0.   , 0.907, 0.093]])
>>> tree_clf.predict([[5, 1.5]])
array([1])
```

Perfect! Notice that the estimated probabilities would be identical anywhere else in the bottom-right rectangle of Figure 6-2—for example, if the petals were 6 cm long and 1.5 cm wide (even though it seems obvious that it would most likely be an _Iris virginica_ in this case). 

## **The CART Training Algorithm** 

Scikit-Learn uses the _Classification and Regression Tree_ (CART) algorithm to train decision trees (also called “growing” trees). The algorithm works by first splitting the training set into two subsets using a single feature _k_ and a threshold _tk_ (e.g., “petal length ≤ 2.45 cm”). How does it choose _k_ and _tk_ ? It searches for the pair ( _k_ , _tk_ ) 

**The CART Training Algorithm | 199** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:10. 

that produces the purest subsets, weighted by their size. Equation 6-2 gives the cost function that the algorithm tries to minimize. 

_Equation 6-2. CART cost function for classification_ 



G left/right measures the impurity of the left/right subset where m left/right is the number of instances in the left/right subset m = m left + m right 

Once the CART algorithm has successfully split the training set in two, it splits the subsets using the same logic, then the sub-subsets, and so on, recursively. It stops recursing once it reaches the maximum depth (defined by the `max_depth` hyperparameter), or if it cannot find a split that will reduce impurity. A few other hyperparameters (described in a moment) control additional stopping con‐ ditions: `min_samples_split` , `min_samples_leaf` , `min_weight_fraction_leaf` , and `max_leaf_nodes` . 



As you can see, the CART algorithm is a _greedy algorithm_ : it greed‐ ily searches for an optimum split at the top level, then repeats the process at each subsequent level. It does not check whether or not the split will lead to the lowest possible impurity several levels down. A greedy algorithm often produces a solution that’s reasonably good but not guaranteed to be optimal. 

Unfortunately, finding the optimal tree is known to be an _NPcomplete_ problem.<sup>1</sup> It requires _O_ (exp( _m_ )) time, making the prob‐ lem intractable even for small training sets. This is why we must settle for a “reasonably good” solution when training decision trees. 

> 1 P is the set of problems that can be solved in _polynomial time_ (i.e., a polynomial of the dataset size). NP is the set of problems whose solutions can be verified in polynomial time. An NP-hard problem is a problem that can be reduced to a known NP-hard problem in polynomial time. An NP-complete problem is both NP and NP-hard. A major open mathematical question is whether or not P = NP. If P ≠ NP (which seems likely), then no polynomial algorithm will ever be found for any NP-complete problem (except perhaps one day on a quantum computer). 

##### **200 | Chapter 6: Decision Trees** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:10. 

## **Computational Complexity** 

Making predictions requires traversing the decision tree from the root to a leaf. Decision trees generally are approximately balanced, so traversing the decision tree requires going through roughly _O_ (log2( _m_ )) nodes, where log2( _m_ ) is the _binary loga‐ rithm_ of _m_ , equal to log( _m_ ) / log(2). Since each node only requires checking the value of one feature, the overall prediction complexity is _O_ (log2( _m_ )), independent of the number of features. So predictions are very fast, even when dealing with large training sets. 

The training algorithm compares all features (or less if `max_features` is set) on all samples at each node. Comparing all features on all samples at each node results in a training complexity of _O_ ( _n_ × _m_ log2( _m_ )). 

## **Gini Impurity or Entropy?** 

By default, the `DecisionTreeClassifier` class uses the Gini impurity measure, but you can select the _entropy_ impurity measure instead by setting the `criterion` hyper‐ parameter to `"entropy"` . The concept of entropy originated in thermodynamics as a measure of molecular disorder: entropy approaches zero when molecules are still and well ordered. Entropy later spread to a wide variety of domains, including in Shannon’s information theory, where it measures the average information content of a message, as we saw in Chapter 4. Entropy is zero when all messages are identical. In machine learning, entropy is frequently used as an impurity measure: a set’s entropy is zero when it contains instances of only one class. Equation 6-3 shows the definition of the entropy of the _i_<sup>th</sup> node. For example, the depth-2 left node in Figure 6-1 has an entropy equal to –(49/54) log2 (49/54) – (5/54) log2 (5/54) ≈ 0.445. 

_Equation 6-3. Entropy_ 



So, should you use Gini impurity or entropy? The truth is, most of the time it does not make a big difference: they lead to similar trees. Gini impurity is slightly faster to compute, so it is a good default. However, when they differ, Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees.<sup>2</sup> 

> 2 See Sebastian Raschka’s interesting analysis for more details. 

**Gini Impurity or Entropy? | 201** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:10. 

## **Regularization Hyperparameters** 

Decision trees make very few assumptions about the training data (as opposed to lin‐ ear models, which assume that the data is linear, for example). If left unconstrained, the tree structure will adapt itself to the training data, fitting it very closely—indeed, most likely overfitting it. Such a model is often called a _nonparametric model_ , not because it does not have any parameters (it often has a lot) but because the number of parameters is not determined prior to training, so the model structure is free to stick closely to the data. In contrast, a _parametric model_ , such as a linear model, has a predetermined number of parameters, so its degree of freedom is limited, reducing the risk of overfitting (but increasing the risk of underfitting). 

To avoid overfitting the training data, you need to restrict the decision tree’s freedom during training. As you know by now, this is called regularization. The regulariza‐ tion hyperparameters depend on the algorithm used, but generally you can at least restrict the maximum depth of the decision tree. In Scikit-Learn, this is controlled by the `max_depth` hyperparameter. The default value is `None` , which means unlimited. Reducing `max_depth` will regularize the model and thus reduce the risk of overfitting. 

The `DecisionTreeClassifier` class has a few other parameters that similarly restrict the shape of the decision tree: 

```
max_features
```

Maximum number of features that are evaluated for splitting at each node 

```
max_leaf_nodes
```

Maximum number of leaf nodes 

```
min_samples_split
```

Minimum number of samples a node must have before it can be split 

#### `min_samples_leaf` 

Minimum number of samples a leaf node must have to be created 

#### `min_weight_fraction_leaf` 

Same as `min_samples_leaf` but expressed as a fraction of the total number of weighted instances 

Increasing `min_*` hyperparameters or reducing `max_*` hyperparameters will regularize the model. 

**202 | Chapter 6: Decision Trees** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:10. 





<!-- Start of picture text -->
No restrictions min_samples_leaf = 5<br>1s<br>0.5 er ees) ee ee<br>x2<br>0.0 raT<br>-0.5<br>-1.0<br>-15 -10 -05 00 O05 10 15 20 -15 -10 -05 00 O05 10 15 20<br>X1 X1<br><!-- End of picture text -->



<!-- Start of picture text -->
x1 <= -0.303<br>squared_error = 0.006<br>samples = 200<br>value = 0.088<br>True False<br>x1 <= 0.272<br>squared_error = 0.005<br>samples = 156<br>value = 0.065<br>squared_error = 0.001<br>samples = 110<br>value = 0.028<br><!-- End of picture text -->



<!-- Start of picture text -->
025 max_depth=2 max_depth=3<br>0.207, * ¥ | wee . A oe<br>“td Depth=0P 1f Se *e i I °<br>0.15<br>4 73 _Depth=2 i > i<br>0.05 ! a coe , aay? ao,<br>0.00 7Depth=1) °° oo ee 4° 'Depth=1 pe: be we Pom fe 6° 1 i<br>J os 2c 0% ° & ! pode: pe$ 2 0% ° & I i<br>-0.05 +1 ; ¢<br>-0.4 -0.2 0.0 0.2 0.4 -0.4 -0.2 0.0 0.2 0.4<br>X1 X1<br><!-- End of picture text -->



<!-- Start of picture text -->
Co)<br>Q<br>.<br><!-- End of picture text -->



<!-- Start of picture text -->
025“f No restrictions min_samples_leaf=10<br>0.20 ok eons<br>i} .. .<br>y 0.150.10 | j | “Fye .°:ie *<br>j .<br>Pil{'] yg l t} ay. °res<br>0.00 i im || ot "e<br>om 8 ee Sf<br>e *,<br>—0.05<br>-0.4 -0.2 0.0 0.2 0.4 -0.4 -0.2 0.0 0.2 0.4<br>X1 X1<br>0.6<br>0.4<br>0.2 i<br>X2 0.0<br>-0.2<br>-0.4<br>-0.6<br>-06 -04 -02 0.0 0.2 0.4 0.6 -06 -04 -0.2 0.0 0.2 0.4 0.6<br>X1 X1<br><!-- End of picture text -->



<!-- Start of picture text -->
T<br>0.6 e Iris setosa | A<br>m Iris versicolor| | a<br>0.4 2 vee 4 a4<br>0.2: e °eei. 4 a: Iris virginicaP| I1 4 A nN aA a<br>0.0 ee ® a a A<br>S08 @ | os A A<br>% “e L .* - 4, I<br>-0.2 ° . a 1, 4" “.<br>1 . ! Aa a“ “<br>-0.4 : rI A 4<br>I A A a<br>Depth=0 Depth=1! ,<br>-0.6<br>2 -1 0 1 2<br>21<br><!-- End of picture text -->



<!-- Start of picture text -->
3.0<br>e Iris setosa<br>254, m Iris versicolor P Ke oe<br>e 4 Iris virginica Aas * re *<br>v5 2.0 Depth=1 atts Nea aA<br>To5 SS On An-~=—<br>$ 1.5 | . i | A<br>ri] a a nn oe oe<br>43 1.0 Depth=0 sae =<br>a -0.0gg<br>0 1 2 3 4 5 6 7<br>Petal length (cm)<br><!-- End of picture text -->

   - **a.** Use `make_moons(n_samples=10000, noise=0.4)` to generate a moons dataset. 

   - **b.** Use `train_test_split()` to split the dataset into a training set and a test set. 

   - **c.** Use grid search with cross-validation (with the help of the `GridSearchCV` class) to find good hyperparameter values for a `DecisionTreeClassifier` . Hint: try various values for `max_leaf_nodes` . 

   - **d.** Train it on the full training set using these hyperparameters, and measure your model’s performance on the test set. You should get roughly 85% to 87% accuracy. 

**8.** Grow a forest by following these steps: 

   - **a.** Continuing the previous exercise, generate 1,000 subsets of the training set, each containing 100 instances selected randomly. Hint: you can use ScikitLearn’s `ShuffleSplit` class for this. 

   - **b.** Train one decision tree on each subset, using the best hyperparameter values found in the previous exercise. Evaluate these 1,000 decision trees on the test set. Since they were trained on smaller sets, these decision trees will likely perform worse than the first decision tree, achieving only about 80% accuracy. 

   - **c.** Now comes the magic. For each test set instance, generate the predictions of the 1,000 decision trees, and keep only the most frequent prediction (you can use SciPy’s `mode()` function for this). This approach gives you _majority-vote predictions_ over the test set. 

   - **d.** Evaluate these predictions on the test set: you should obtain a slightly higher accuracy than your first model (about 0.5 to 1.5% higher). Congratulations, you have trained a random forest classifier! 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

**Exercises | 209** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:10. 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:10. 

