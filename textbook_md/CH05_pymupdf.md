# **CHAPTER 5 Support Vector Machines** 

A _support vector machine_ (SVM) is a powerful and versatile machine learning model, capable of performing linear or nonlinear classification, regression, and even novelty detection. SVMs shine with small to medium-sized nonlinear datasets (i.e., hundreds to thousands of instances), especially for classification tasks. However, they don’t scale very well to very large datasets, as you will see. 

This chapter will explain the core concepts of SVMs, how to use them, and how they work. Let’s jump right in! 

## **Linear SVM Classification** 

The fundamental idea behind SVMs is best explained with some visuals. Figure 5-1 shows part of the iris dataset that was introduced at the end of Chapter 4. The two classes can clearly be separated easily with a straight line (they are _linearly separable_ ). The left plot shows the decision boundaries of three possible linear classifiers. The model whose decision boundary is represented by the dashed line is so bad that it does not even separate the classes properly. The other two models work perfectly on this training set, but their decision boundaries come so close to the instances that these models will probably not perform as well on new instances. In contrast, the solid line in the plot on the right represents the decision boundary of an SVM classifier; this line not only separates the two classes but also stays as far away from the closest training instances as possible. You can think of an SVM classifier as fitting the widest possible street (represented by the parallel dashed lines) between the classes. This is called . _large margin classification_ 

**175** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:02. 



<!-- Start of picture text -->
2.0<br>J . if ba =<br>1.0 a = = Pa<br>=35 os aeSaaneee= ee eee°‘ XN‘<br>0.0 a s<br>0 1 2 3 4 5 0 1 2 3 4 5<br>Petal length Petal length<br><!-- End of picture text -->





<!-- Start of picture text -->
Unscaled Scaled<br>80 e 2Sare<br>1 Tal<br>X1 60 fot— -@- = = —, Pe<br>40 ---@----- += == XD 9 ~_ ~~ e-~<br>BLL — =s<br>20 See —__<br>0 a - 12 “SLL ~e-__ —<br>0 1 2 3 4 5 6 -2.0 -15 -10 -05 0.0 05 10 15 2.0<br>Xo x'9<br><!-- End of picture text -->



<!-- Start of picture text -->
2.0<br>G15 ‘ —~<br>=S.o{Lo I mpo ssi bl ee! spaae |  |g SSS ~ -,<br>8.Boe aie RSie: .TSS—<br>00 : : tlier ==<br>a) 1 2 3 4 5 0 1 2 3 4 5<br>Petal length Petal length<br><!-- End of picture text -->



<!-- Start of picture text -->
C=1 C=100<br>FSI 2.5 a Iris virgini r a<br>AS) 2.0 = acreIris versicolor , haaaaaaran arn TS-- Aaaaaaarawr<br>3© — — swe **@--@@-4p+ 7 iao Fee© ry wear<br>$15 _ 4 > A ~-- —, ~2. ioe<br>2 t~peei eepie A ———9 epee. platPe ---2_ak. =-~==<br>1.078 Sr = a 7<br>4.00 4.25 450 4.75 5.00 5.25 5.50 5.75 4.00 4.25 450 4.75 5.00 5.25 5.50 5.75<br>Petal length Petal length<br><!-- End of picture text -->



```
fromsklearn.datasetsimportload_iris
fromsklearn.pipelineimportmake_pipeline
fromsklearn.preprocessingimportStandardScaler
fromsklearn.svmimportLinearSVC
```

```
iris=load_iris(as_frame=True)
X=iris.data[["petal length (cm)", "petal width (cm)"]].values
y= (iris.target==2)  # Iris virginica
```

```
svm_clf=make_pipeline(StandardScaler(),
LinearSVC(C=1, random_state=42))
svm_clf.fit(X, y)
```

The resulting model is represented on the left in Figure 5-4. 

Then, as usual, you can use the model to make predictions: 

```
>>> X_new= [[5.5, 1.7], [5.0, 1.5]]
>>> svm_clf.predict(X_new)
array([ True, False])
```

The first plant is classified as an _Iris virginica_ , while the second is not. Let’s look at the scores that the SVM used to make these predictions. These measure the signed distance between each instance and the decision boundary: 

```
>>> svm_clf.decision_function(X_new)
array([ 0.66163411, -0.22036063])
```

Unlike `LogisticRegression` , `LinearSVC` doesn’t have a `predict_proba()` method to estimate the class probabilities. That said, if you use the `SVC` class (discussed shortly) instead of `LinearSVC` , and if you set its `probability` hyperparameter to `True` , then the model will fit an extra model at the end of training to map the SVM decision function scores to estimated probabilities. Under the hood, this requires using 5-fold cross-validation to generate out-of-sample predictions for every instance in the train‐ ing set, then training a `LogisticRegression` model, so it will slow down training considerably. After that, the `predict_proba()` and `predict_log_proba()` methods will be available. 

## **Nonlinear SVM Classification** 

Although linear SVM classifiers are efficient and often work surprisingly well, many datasets are not even close to being linearly separable. One approach to handling nonlinear datasets is to add more features, such as polynomial features (as we did in Chapter 4); in some cases this can result in a linearly separable dataset. Consider the lefthand plot in Figure 5-5: it represents a simple dataset with just one feature, _x_ 1. This dataset is not linearly separable, as you can see. But if you add a second feature _x_ 2 = ( _x_ 1)<sup>2</sup> , the resulting 2D dataset is perfectly linearly separable. 

**178 | Chapter 5: Support Vector Machines** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:02. 



<!-- Start of picture text -->
16 a |<br>” oe oe ee<br>0 SsInneCG<br>-4 -3 -2 -1 #0 1 2 3 4 -4 -3 -2 -1 #0 1 2 3 4<br>x1 x1<br><!-- End of picture text -->



<!-- Start of picture text -->
1.5<br>1.0<br>0.5<br>X2<br>0.0<br>-0.5<br>-1.0<br>-15 -10 -05 00 O5 10 15 20 25<br>X1<br><!-- End of picture text -->



<!-- Start of picture text -->
1s degree=3, coef0=1, C=5 degree=10, coef0=100, C=5<br>1.0<br>0.5<br>X2<br>0.0<br>-0.5<br>-1.0<br>-15 -10 -05 00 O05 10 215 20 -15 -10 -05 00 O05 10 415 20<br>Xi X1<br><!-- End of picture text -->





<!-- Start of picture text -->
1.00 aN, ay, 1.0 ry<br>/ X2 \ L X3 “<br>0.75 / \ i * 0.8<br>7 a rs * A<br>Pay / \ i * 0.6<br>8 f 5 x (x)<br>= 0.50 rf + oe<br>£ / EN ‘. 0.4 AL<br>a) y YoN 2 .<br>0.25 47 2 xs . 02 se<br>” SS *e, se<br>0.00 a —— - ee ee-——eee- 0.0 es ~<br>-4 -3 -2 -1 «0 1 2 3 4 0.0 0.2 0.4 0.6 0.8 1.0<br>X1 X2<br><!-- End of picture text -->



<!-- Start of picture text -->
1s gamma=0.1, C=0.001 gamma=0.1, C=1000<br>1.0<br>0.5<br>x2<br>0.0<br>-0.5<br>-1.0<br>1s gamma=5, C=0.001 gamma=5, C=1000<br>1.0<br>0.5<br>x2<br>0.0<br>-0.5<br>-1.0<br>-15 -10 -05 00 O05 10 15 20 -15 -10 -05 00 O05 10 15 20<br>X1 X1<br><!-- End of picture text -->



With so many kernels to choose from, how can you decide which one to use? As a rule of thumb, you should always try the linear kernel first. The `LinearSVC` class is much faster than `SVC(kernel="linear")` , especially if the training set is very large. If it is not too large, you should also try kernelized SVMs, starting with the Gaussian RBF kernel; it often works really well. Then, if you have spare time and computing power, you can experiment with a few other kernels using hyperparameter search. If there are kernels specialized for your training set’s data structure, make sure to give them a try too. 

### **SVM Classes and Computational Complexity** 

The `LinearSVC` class is based on the `liblinear` library, which implements an opti‐ mized algorithm for linear SVMs.<sup>1</sup> It does not support the kernel trick, but it scales almost linearly with the number of training instances and the number of features. Its training time complexity is roughly _O_ ( _m_ × _n_ ). The algorithm takes longer if you require very high precision. This is controlled by the tolerance hyperparameter ϵ (called `tol` in Scikit-Learn). In most classification tasks, the default tolerance is fine. 

The `SVC` class is based on the `libsvm` library, which implements an algorithm that supports the kernel trick.<sup>2</sup> The training time complexity is usually between _O_ ( _m_<sup>2</sup> × _n_ ) and _O_ ( _m_<sup>3</sup> × _n_ ). Unfortunately, this means that it gets dreadfully slow when the number of training instances gets large (e.g., hundreds of thousands of instances), so this algorithm is best for small or medium-sized nonlinear training sets. It scales well with the number of features, especially with sparse features (i.e., when each instance has few nonzero features). In this case, the algorithm scales roughly with the average number of nonzero features per instance. 

The `SGDClassifier` class also performs large margin classification by default, and its hyperparameters–especially the regularization hyperparameters ( `alpha` and `penalty` ) and the `learning_rate` –can be adjusted to produce similar results as the linear SVMs. For training it uses stochastic gradient descent (see Chapter 4), which allows incremental learning and uses little memory, so you can use it to train a model on a large dataset that does not fit in RAM (i.e., for out-of-core learning). Moreover, it scales very well, as its computational complexity is _O_ ( _m_ × _n_ ). Table 5-1 compares Scikit-Learn’s SVM classification classes. 

> 1 Chih-Jen Lin et al., “A Dual Coordinate Descent Method for Large-Scale Linear SVM”, _Proceedings of the 25th International Conference on Machine Learning_ (2008): 408–415. 

> 2 John Platt, “Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines” (Microsoft Research technical report, April 21, 1998). 

**Nonlinear SVM Classification | 183** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:02. 



<!-- Start of picture text -->
W epsilon=0.5 epsilon=1.2<br>10} — Y o ©? + —~y o © <<br>®-- o-<br>9 cae 2 oom e<br>8 “oeoa ato*od uote rae28 =<br>© ov 2-@ A 2<br>Vy o eaeOse” e e -€eo. os 7 ed@<br>6 O- e oo A en<br>e nae ‘ 52'8e e- to<br>,tAe@54 Sncfy--AV@ee77 e ee-" eu@ eeaAo-o5----GEv-<br>é e eu e<br>3 e<br>0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00<br>X1 X1<br><!-- End of picture text -->



<!-- Start of picture text -->
Lo degree=2, C=0.01, epsilon=0.1 degree=2, C=100, epsilon=0.1<br>yA e y“ eA<br>0.8 ;<br>e e 4<br>0.6 ,” NS ifs<br>7@ ri \ ca v7<br>y NS.~s see. o7 aSeAv»' Y“@ . N \@» e “4be)oy e<br>SS % a" Per \ %. @--. “<br>0.2 ~oS--—* ee o~*o*”- Optoe s--« @ ee4<br>o.----8| € a, fe<br>e @-----"<br>0.0<br>=1.00 -0.75 -0.50 -0.25 0.00 0.25 0.50 0.75 1.00 —1.00 -0.75 —0.50 -0.25 0.00 0.25 0.50 0.75 1.00<br>X1 X1<br><!-- End of picture text -->



The rest of this chapter explains how SVMs make predictions and how their training algorithms work, starting with linear SVM classifiers. If you are just getting started with machine learning, you can safely skip this and go straight to the exercises at the end of this chapter, and come back later when you want to get a deeper understanding of SVMs. 

## **Under the Hood of Linear SVM Classifiers** 

A linear SVM classifier predicts the class of a new instance **x** by first computing the decision function **θ**<sup>⊺</sup> **x** = _θ_ 0 _x_ 0 + ⋯ + _θn xn_ , where _x_ 0 is the bias feature (always equal to 1). If the result is positive, then the predicted class _ŷ_ is the positive class (1); otherwise it is the negative class (0). This is exactly like `LogisticRegression` (discussed in Chapter 4). 



Up to now, I have used the convention of putting all the model parameters in one vector **θ** , including the bias term **θ** 0 and the input feature weights **θ** 1 to **θ** _n_ . This required adding a bias input _x_ 0 = 1 to all instances. Another very common convention is to separate the bias term _b_ (equal to **θ** 0) and the feature weights vector **w** (containing **θ** 1 to **θ** _n_ ). In this case, no bias feature needs to be added to the input feature vectors, and the linear SVM’s decision function is equal to **w**<sup>⊺</sup> **x** + _b_ = _w_ 1 _x_ 1 + ⋯ + _wn xn_ + _b_ . I will use this convention throughout the rest of this book. 

So, making predictions with a linear SVM classifier is quite straightforward. How about training? This requires finding the weights vector **w** and the bias term _b_ that make the street, or margin, as wide as possible while limiting the number of margin violations. Let’s start with the width of the street: to make it larger, we need to make **w** smaller. This may be easier to visualize in 2D, as shown in Figure 5-12. Let’s define the borders of the street as the points where the decision function is equal to –1 or +1. In the left plot the weight _w1_ is 1, so the points at which _w_ 1 _x_ 1 = –1 or +1 are _x_ 1 = –1 and +1: therefore the margin’s size is 2. In the right plot the weight is 0.5, so the points at which _w_ 1 _x_ 1 = –1 or +1 are _x_ 1 = –2 and +2: the margin’s size is 4. So, we need to keep **w** as small as possible. Note that the bias term _b_ has no influence on the size of the margin: tweaking it just shifts the margin around, without affecting its size. 

**186 | Chapter 5: Support Vector Machines** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:02. 



<!-- Start of picture text -->
> Ww4= 1 w= 0.5<br>aes)| Ae<br>CCA-1 |<br>4S<br>73 —2 -1 0 1 2 3-3 —2 -1 0 1 2 3<br>XI X1<br><!-- End of picture text -->

( ( () ) 



To get the soft margin objective, we need to introduce a _slack variable ζ_<sup>(</sup><sup>_i_)</sup> ≥ 0 for each instance:<sup>3</sup> _ζ_<sup>(</sup><sup>_i_)</sup> measures how much the _i_<sup>th</sup> instance is allowed to violate the margin. We now have two conflicting objectives: make the slack variables as small as possible to reduce the margin violations, and make ½ **w**<sup>⊺</sup> **w** as small as possible to increase the margin. This is where the `C` hyperparameter comes in: it allows us to define the trade-off between these two objectives. This gives us the constrained optimization problem in Equation 5-2. 

_Equation 5-2. Soft margin linear SVM classifier objective_ 



The hard margin and soft margin problems are both convex quadratic optimization problems with linear constraints. Such problems are known as _quadratic program‐ ming_ (QP) problems. Many off-the-shelf solvers are available to solve QP problems by using a variety of techniques that are outside the scope of this book.<sup>4</sup> 

Using a QP solver is one way to train an SVM. Another is to use gradient descent to minimize the _hinge loss_ or the _squared hinge loss_ (see Figure 5-13). Given an instance **x** of the positive class (i.e., with _t_ = 1), the loss is 0 if the output _s_ of the decision function ( _s_ = **w**<sup>⊺</sup> **x** + _b_ ) is greater than or equal to 1. This happens when the instance is off the street and on the positive side. Given an instance of the negative class (i.e., with _t_ = –1), the loss is 0 if _s_ ≤ –1. This happens when the instance is off the street and on the negative side. The further away an instance is from the correct side of the margin, the higher the loss: it grows linearly for the hinge loss, and quadratically for the squared hinge loss. This makes the squared hinge loss more sensitive to outliers. However, if the dataset is clean, it tends to converge faster. By default, `LinearSVC` uses the squared hinge loss, while `SGDClassifier` uses the hinge loss. Both classes let you choose the loss by setting the `loss` hyperparameter to `"hinge"` or `"squared_hinge"` . The `SVC` class’s optimization algorithm finds a similar solution as minimizing the hinge loss. 

> 3 Zeta ( _ζ_ ) is the sixth letter of the Greek alphabet. 

> 4 To learn more about quadratic programming, you can start by reading Stephen Boyd and Lieven Vandenber‐ ghe’s book _Convex Optimization_ (Cambridge University Press) or watching Richard Brown’s series of video lectures. 

**188 | Chapter 5: Support Vector Machines** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:02. 



<!-- Start of picture text -->
Hinge loss = max(0,1-st) Squared Hinge loss<br>fin fot} NV<br>¢o— t=1 / — t=1<br>Nee |<br>0*INLa --- t=-1 UN/| --- t=-1<br>a 4 Ed<br>7 ee ee eee<br>” s=wx +b : ; ° ” s=wx+b : ;<br><!-- End of picture text -->



<!-- Start of picture text -->
_ QOQO00 0 ()<br>() ()O<br><!-- End of picture text -->

Once you find the vector α that minimizes this equation (using a QP solver), use Equation 5-4 to compute the w and b that minimize the primal problem. In this equation, _ns_ represents the number of support vectors. 

##### _Equation 5-4. From the dual solution to the primal solution_ 



The dual problem is faster to solve than the primal one when the number of training instances is smaller than the number of features. More importantly, the dual problem makes the kernel trick possible, while the primal problem does not. So what is this kernel trick, anyway? 

### **Kernelized SVMs** 

Suppose you want to apply a second-degree polynomial transformation to a twodimensional training set (such as the moons training set), then train a linear SVM classifier on the transformed training set. Equation 5-5 shows the second-degree polynomial mapping function _ϕ_ that you want to apply. 

_Equation 5-5. Second-degree polynomial mapping_ 



Notice that the transformed vector is 3D instead of 2D. Now let’s look at what happens to a couple of 2D vectors, **a** and **b** , if we apply this second-degree polynomial mapping and then compute the dot product<sup>6</sup> of the transformed vectors (see Equa‐ tion 5-6). 

> 6 As explained in Chapter 4, the dot product of two vectors **a** and **b** is normally noted **a** · **b** . However, in machine learning, vectors are frequently represented as column vectors (i.e., single-column matrices), so the dot product is achieved by computing **a**<sup>⊺</sup> **b** . To remain consistent with the rest of the book, we will use this notation here, ignoring the fact that this technically results in a single-cell matrix rather than a scalar value. 

**190 | Chapter 5: Support Vector Machines** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:02. 

_Equation 5-6. Kernel trick for a second-degree polynomial mapping_ 



How about that? The dot product of the transformed vectors is equal to the square of the dot product of the original vectors: _ϕ_ ( **a** )<sup>⊺</sup> _ϕ_ ( **b** ) = ( **a**<sup>⊺</sup> **b** )<sup>2</sup> . 

Here is the key insight: if you apply the transformation _ϕ_ to all training instances, then the dual problem (see Equation 5-3) will contain the dot product _ϕ_ ( **x**<sup>(</sup><sup>_i_)</sup> )<sup>⊺</sup> _ϕ_ ( **x**<sup>(</sup><sup>_j_)</sup> ). But if _ϕ_ is the second-degree polynomial transformation defined in Equation 5-5, 

2 then you can replace this dot product of transformed vectors simply by x i ⊺x j . So, you don’t need to transform the training instances at all; just replace the dot product by its square in Equation 5-3. The result will be strictly the same as if you had gone through the trouble of transforming the training set and then fitting a linear SVM algorithm, but this trick makes the whole process much more computationally efficient. 

The function _K_ ( **a** , **b** ) = ( **a**<sup>⊺</sup> **b** )<sup>2</sup> is a second-degree polynomial kernel. In machine learning, a _kernel_ is a function capable of computing the dot product _ϕ_ ( **a** )<sup>⊺</sup> _ϕ_ ( **b** ), based only on the original vectors **a** and **b** , without having to compute (or even to know about) the transformation _ϕ_ . Equation 5-7 lists some of the most commonly used kernels. 

_Equation 5-7. Common kernels_ 

Linear: K a , b = a<sup>⊺</sup> b d Polynomial: K a , b = γa<sup>⊺</sup> b + r Gaussian RBF: K a , b = exp − γ∥a − b ∥<sup>2</sup> Sigmoid: K a , b = tanh γa<sup>⊺</sup> b + r 

**The Dual Problem | 191** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:02. 

#### **Mercer’s Theorem** 

According to _Mercer’s theorem_ , if a function _K_ ( **a** , **b** ) respects a few mathematical conditions called _Mercer’s conditions_ (e.g., _K_ must be continuous and symmetric in its arguments so that _K_ ( **a** , **b** ) = _K_ ( **b** , **a** ), etc.), then there exists a function _ϕ_ that maps **a** and **b** into another space (possibly with much higher dimensions) such that _K_ ( **a** , **b** ) = _ϕ_ ( **a** )<sup>⊺</sup> _ϕ_ ( **b** ). You can use _K_ as a kernel because you know _ϕ_ exists, even if you don’t know what _ϕ_ is. In the case of the Gaussian RBF kernel, it can be shown that _ϕ_ maps each training instance to an infinite-dimensional space, so it’s a good thing you don’t need to actually perform the mapping! 

Note that some frequently used kernels (such as the sigmoid kernel) don’t respect all of Mercer’s conditions, yet they generally work well in practice. 

There is still one loose end we must tie up. Equation 5-4 shows how to go from the dual solution to the primal solution in the case of a linear SVM classifier. But if you apply the kernel trick, you end up with equations that include _ϕ_ ( _x_<sup>(</sup><sup>_i_)</sup> ). In fact, w must have the same number of dimensions as _ϕ_ ( _x_<sup>(</sup><sup>_i_)</sup> ), which may be huge or even infinite, so you can’t compute it. But how can you make predictions without knowing w ? Well, the good news is that you can plug the formula for w from Equation 5-4 into the decision function for a new instance **x**<sup>(</sup><sup>_n_)</sup> , and you get an equation with only dot products between input vectors. This makes it possible to use the kernel trick (Equation 5-8). 

_Equation 5-8. Making predictions with a kernelized SVM_ 



Note that since _α_<sup>(</sup><sup>_i_)</sup> ≠ 0 only for support vectors, making predictions involves comput‐ ing the dot product of the new input vector **x**<sup>(</sup><sup>_n_)</sup> with only the support vectors, not all the training instances. Of course, you need to use the same trick to compute the bias term b (Equation 5-9). 

**192 | Chapter 5: Support Vector Machines** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:02. 

_Equation 5-9. Using the kernel trick to compute the bias term_ 



If you are starting to get a headache, that’s perfectly normal: it’s an unfortunate side effect of the kernel trick. 



It is also possible to implement online kernelized SVMs, capable of incremental learning, as described in the papers “Incremental and Decremental Support Vector Machine Learning”<sup>7</sup> and “Fast Kernel Classifiers with Online and Active Learning”.<sup>8</sup> These kernelized SVMs are implemented in Matlab and C++. But for large-scale nonlinear problems, you may want to consider using random for‐ ests (see Chapter 7) or neural networks (see Part II). 

## **Exercises** 

**1.** What is the fundamental idea behind support vector machines? 

**2.** What is a support vector? 

**3.** Why is it important to scale the inputs when using SVMs? 

**4.** Can an SVM classifier output a confidence score when it classifies an instance? What about a probability? 

**5.** How can you choose between `LinearSVC` , `SVC` , and `SGDClassifier` ? 

**6.** Say you’ve trained an SVM classifier with an RBF kernel, but it seems to underfit the training set. Should you increase or decrease _γ_ ( `gamma` )? What about `C` ? 

**7.** What does it mean for a model to be ϵ _-insensitive_ ? 

**8.** What is the point of using the kernel trick? 

> 7 Gert Cauwenberghs and Tomaso Poggio, “Incremental and Decremental Support Vector Machine Learning”, _Proceedings of the 13th International Conference on Neural Information Processing Systems_ (2000): 388–394. 

> 8 Antoine Bordes et al., “Fast Kernel Classifiers with Online and Active Learning”, _Journal of Machine Learning Research_ 6 (2005): 1579–1619. 

**Exercises | 193** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:02. 

**9.** Train a `LinearSVC` on a linearly separable dataset. Then train an `SVC` and a `SGDClassifier` on the same dataset. See if you can get them to produce roughly the same model. 

**10.** Train an SVM classifier on the wine dataset, which you can load using `sklearn.datasets.load_wine()` . This dataset contains the chemical analyses of 178 wine samples produced by 3 different cultivators: the goal is to train a classification model capable of predicting the cultivator based on the wine’s chemical analysis. Since SVM classifiers are binary classifiers, you will need to use one-versus-all to classify all three classes. What accuracy can you reach? 

**11.** Train and fine-tune an SVM regressor on the California housing dataset. You can use the original dataset rather than the tweaked version we used in Chapter 2, which you can load using `sklearn.datasets.fetch_california_housing()` . The targets represent hundreds of thousands of dollars. Since there are over 20,000 instances, SVMs can be slow, so for hyperparameter tuning you should use far fewer instances (e.g., 2,000) to test many more hyperparameter combina‐ tions. What is your best model’s RMSE? 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

###### **194 | Chapter 5: Support Vector Machines** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:02. 

