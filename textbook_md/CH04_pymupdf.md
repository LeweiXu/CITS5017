# **CHAPTER 4 Training Models** 

So far we have treated machine learning models and their training algorithms mostly like black boxes. If you went through some of the exercises in the previous chapters, you may have been surprised by how much you can get done without knowing any‐ thing about what’s under the hood: you optimized a regression system, you improved a digit image classifier, and you even built a spam classifier from scratch, all without knowing how they actually work. Indeed, in many situations you don’t really need to know the implementation details. 

However, having a good understanding of how things work can help you quickly home in on the appropriate model, the right training algorithm to use, and a good set of hyperparameters for your task. Understanding what’s under the hood will also help you debug issues and perform error analysis more efficiently. Lastly, most of the topics discussed in this chapter will be essential in understanding, building, and training neural networks (discussed in Part II of this book). 

In this chapter we will start by looking at the linear regression model, one of the simplest models there is. We will discuss two very different ways to train it: 

- Using a “closed-form” equation<sup>1</sup> that directly computes the model parameters that best fit the model to the training set (i.e., the model parameters that mini‐ mize the cost function over the training set). 

- Using an iterative optimization approach called gradient descent (GD) that grad‐ ually tweaks the model parameters to minimize the cost function over the train‐ ing set, eventually converging to the same set of parameters as the first method. We will look at a few variants of gradient descent that we will use again and 

> 1 A closed-form equation is only composed of a finite number of constants, variables, and standard operations: for example, _a_ = sin( _b_ – _c_ ). No infinite sums, no limits, no integrals, etc. 

**131** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 

again when we study neural networks in Part II: batch GD, mini-batch GD, and stochastic GD. 

Next we will look at polynomial regression, a more complex model that can fit nonlinear datasets. Since this model has more parameters than linear regression, it is more prone to overfitting the training data. We will explore how to detect whether or not this is the case using learning curves, and then we will look at several regularization techniques that can reduce the risk of overfitting the training set. 

Finally, we will examine two more models that are commonly used for classification tasks: logistic regression and softmax regression. 



There will be quite a few math equations in this chapter, using basic notions of linear algebra and calculus. To understand these equa‐ tions, you will need to know what vectors and matrices are; how to transpose them, multiply them, and inverse them; and what partial derivatives are. If you are unfamiliar with these concepts, please go through the linear algebra and calculus introductory tutorials available as Jupyter notebooks in the online supplemental material. For those who are truly allergic to mathematics, you should still go through this chapter and simply skip the equations; hopefully, the text will be sufficient to help you understand most of the concepts. 

## **Linear Regression** 

In Chapter 1 we looked at a simple regression model of life satisfaction: 

_life_satisfaction_ = _θ_ 0 + _θ_ 1 × _GDP_per_capita_ 

This model is just a linear function of the input feature `GDP_per_capita` . _θ_ 0 and _θ_ 1 are the model’s parameters. 

More generally, a linear model makes a prediction by simply computing a weighted sum of the input features, plus a constant called the _bias term_ (also called the _intercept term_ ), as shown in Equation 4-1. 

_Equation 4-1. Linear regression model prediction_ 

y = θ 0 + θ 1 x 1 + θ 2 x 2 + ⋯ + θnxn 

In this equation: 

- _ŷ_ is the predicted value. 

- _n_ is the number of features. 

###### **132 | Chapter 4: Training Models** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 

- _xi_ is the _i_<sup>th</sup> feature value. 

- _θj_ is the _j_<sup>th</sup> model parameter, including the bias term _θ_ 0 and the feature weights _θ_ 1, _θ_ 2, ⋯ , _θn_ . 

This can be written much more concisely using a vectorized form, as shown in Equation 4-2. 

_Equation 4-2. Linear regression model prediction (vectorized form)_ 

y = ℎθ x = θ · x 

In this equation: 

- _h_ **θ** is the hypothesis function, using the model parameters **θ** . 

- **θ** is the model’s _parameter vector_ , containing the bias term _θ_ 0 and the feature weights _θ_ 1 to _θn_ . 

- **x** is the instance’s _feature vector_ , containing _x_ 0 to _xn_ , with _x_ 0 always equal to 1. 

- **θ** · **x** is the dot product of the vectors **θ** and **x** , which is equal to _θ_ 0 _x_ 0 + _θ_ 1 _x_ 1 + _θ_ 2 _x_ 2 + ... + _θnxn_ . 



In machine learning, vectors are often represented as _column vec‐ tors_ , which are 2D arrays with a single column. If **θ** and **x** are column vectors, then the prediction is y = θ<sup>⊺</sup> x , where θ<sup>⊺</sup> is the _transpose_ of **θ** (a row vector instead of a column vector) and θ<sup>⊺</sup> x is the matrix multiplication of θ<sup>⊺</sup> and **x** . It is of course the same prediction, except that it is now represented as a single-cell matrix rather than a scalar value. In this book I will use this notation to avoid switching between dot products and matrix multiplications. 

OK, that’s the linear regression model—but how do we train it? Well, recall that training a model means setting its parameters so that the model best fits the training set. For this purpose, we first need a measure of how well (or poorly) the model fits the training data. In Chapter 2 we saw that the most common performance measure of a regression model is the root mean square error (Equation 2-1). Therefore, to train a linear regression model, we need to find the value of **θ** that minimizes the RMSE. In practice, it is simpler to minimize the mean square error (MSE) than the RMSE, and it leads to the same result (because the value that minimizes a positive function also minimizes its square root). 

**Linear Regression | 133** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 



Learning algorithms will often optimize a different loss function during training than the performance measure used to evaluate the final model. This is generally because the function is easier to optimize and/or because it has extra terms needed during training only (e.g., for regularization). A good performance metric is as close as possible to the final business objective. A good training loss is easy to optimize and strongly correlated with the metric. For example, classifiers are often trained using a cost function such as the log loss (as you will see later in this chapter) but evaluated using precision/recall. The log loss is easy to minimize, and doing so will usually improve precision/recall. 

The MSE of a linear regression hypothesis _h_ **θ** on a training set **X** is calculated using Equation 4-3. 

_Equation 4-3. MSE cost function for a linear regression model_ 



Most of these notations were presented in Chapter 2 (see “Notations” on page 44). The only difference is that we write _h_ **θ** instead of just _h_ to make it clear that the model is parametrized by the vector **θ** . To simplify notations, we will just write MSE( **θ** ) instead of MSE( **X** , _h_ **θ** ). 

### **The Normal Equation** 

To find the value of **θ** that minimizes the MSE, there exists a _closed-form solution_ —in other words, a mathematical equation that gives the result directly. This is called the _Normal equation_ (Equation 4-4). 

_Equation 4-4. Normal equation_ 



In this equation: 

- θ is the value of **θ** that minimizes the cost function. 

- **y** is the vector of target values containing _y_<sup>(1)</sup> to _y_<sup>(</sup><sup>_m_)</sup> . 

**134 | Chapter 4: Training Models** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 



<!-- Start of picture text -->
up| ft ft fT<br>oeee<br>e e<br>a eS<br>se eee eeeee<br>eeheeRe TF| ft<br>ee)<br>0.00 0.25 050 °&# 0.75 100 1.25 1.50 1.75 2.00<br>Xl<br><!-- End of picture text -->





<!-- Start of picture text -->
u See— yf | dT<br>ow | PP<br>10ee<br>feect) eee<br>4 poo"ee<br>0<br>tf0.00 0.25 0.50tT0.75 1.00[ 1.25| 1.50ty1.75 2.00<br>X1<br><!-- End of picture text -->

Performing linear regression using Scikit-Learn is relatively straightforward: 

```
>>> fromsklearn.linear_modelimportLinearRegression
>>> lin_reg=LinearRegression()
>>> lin_reg.fit(X, y)
>>> lin_reg.intercept_, lin_reg.coef_
(array([4.21509616]), array([[2.77011339]]))
>>> lin_reg.predict(X_new)
array([[4.21509616],
       [9.75532293]])
```

Notice that Scikit-Learn separates the bias term ( `intercept_` ) from the feature weights ( `coef_` ). The `LinearRegression` class is based on the `scipy.linalg.lstsq()` function (the name stands for “least squares”), which you could call directly: 

```
>>> theta_best_svd, residuals, rank, s=np.linalg.lstsq(X_b, y, rcond=1e-6)
>>> theta_best_svd
array([[4.21509616],
       [2.77011339]])
```

This function computes θ = X<sup>+</sup> y , where X<sup>+</sup> is the _pseudoinverse_ of **X** (specifi‐ cally, the Moore–Penrose inverse). You can use `np.linalg.pinv()` to compute the pseudoinverse directly: 

```
>>> np.linalg.pinv(X_b) @y
array([[4.21509616],
       [2.77011339]])
```

The pseudoinverse itself is computed using a standard matrix factorization technique called _singular value decomposition_ (SVD) that can decompose the training set matrix **X** into the matrix multiplication of three matrices **U Σ V**<sup>⊺</sup> (see `numpy.linalg.svd()` ). The pseudoinverse is computed as X<sup>+</sup> = VΣ<sup>+</sup> U<sup>⊺</sup> . To compute the matrix Σ<sup>+</sup> , the algorithm takes **Σ** and sets to zero all values smaller than a tiny threshold value, then it replaces all the nonzero values with their inverse, and finally it transposes the resulting matrix. This approach is more efficient than computing the Normal equation, plus it handles edge cases nicely: indeed, the Normal equation may not work if the matrix **X**<sup>⊺</sup> **X** is not invertible (i.e., singular), such as if _m_ < _n_ or if some features are redundant, but the pseudoinverse is always defined. 

### **Computational Complexity** 

The Normal equation computes the inverse of **X**<sup>⊺</sup> **X** , which is an ( _n_ + 1) × ( _n_ + 1) matrix (where _n_ is the number of features). The _computational complexity_ of inverting such a matrix is typically about _O_ ( _n_<sup>2.4</sup> ) to _O_ ( _n_<sup>3</sup> ), depending on the implementation. In other words, if you double the number of features, you multiply the computation time by roughly 2<sup>2.4</sup> = 5.3 to 2<sup>3</sup> = 8. 

**Linear Regression | 137** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 



<!-- Start of picture text -->
|<br><!-- End of picture text -->

| 



<!-- Start of picture text -->
Cost<br>Learning step<br>'<br>'<br>1<br>'<br>'<br>'<br>'<br>'<br>1 Minimum<br>'<br>'<br>Random initial'  value A' (-)<br>0<br>Cost<br>1<br>‘<br>t<br>to:<br>tix)t ’<br>tot<br>'<br>t<br>1<br>'<br>t<br>Nee,i]<br>M76<br>Start<br><!-- End of picture text -->



<!-- Start of picture text -->
Cost<br>1<br>'<br>'<br>0<br>SS<br>Start<br><!-- End of picture text -->



<!-- Start of picture text -->
Cost<br>Plateau<br>H<br>'<br>Local minimum1 Global tminimum f)<br><!-- End of picture text -->





<!-- Start of picture text -->
B<br><!-- End of picture text -->

This diagram also illustrates the fact that training a model means searching for a combination of model parameters that minimizes a cost function (over the training set). It is a search in the model’s _parameter space_ . The more parameters a model has, the more dimensions this space has, and the harder the search is: searching for a needle in a 300-dimensional haystack is much trickier than in 3 dimensions. Fortunately, since the cost function is convex in the case of linear regression, the needle is simply at the bottom of the bowl. 

### **Batch Gradient Descent** 

To implement gradient descent, you need to compute the gradient of the cost func‐ tion with regard to each model parameter _θj_ . In other words, you need to calculate how much the cost function will change if you change _θj_ just a little bit. This is called a _partial derivative_ . It is like asking, “What is the slope of the mountain under my feet if I face east”? and then asking the same question facing north (and so on for all other dimensions, if you can imagine a universe with more than three dimensions). Equation 4-5 computes the partial derivative of the MSE with regard to parameter _θj_ , noted ∂ MSE( **θ** ) / ∂θ _j_ . 

_Equation 4-5. Partial derivatives of the cost function_ 



Instead of computing these partial derivatives individually, you can use Equation 4-6 to compute them all in one go. The gradient vector, noted ∇ **θ** MSE( **θ** ), contains all the partial derivatives of the cost function (one for each model parameter). 

_Equation 4-6. Gradient vector of the cost function_ 



**142 | Chapter 4: Training Models** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 



Notice that this formula involves calculations over the full training set **X** , at each gradient descent step! This is why the algorithm is called _batch gradient descent_ : it uses the whole batch of training data at every step (actually, _full gradient descent_ would probably be a better name). As a result, it is terribly slow on very large training sets (we will look at some much faster gradient descent algorithms shortly). However, gradient descent scales well with the number of features; training a linear regression model when there are hundreds of thousands of features is much faster using gradient descent than using the Normal equation or SVD decomposition. 

Once you have the gradient vector, which points uphill, just go in the opposite direction to go downhill. This means subtracting ∇ **θ** MSE( **θ** ) from **θ** . This is where the learning rate _η_ comes into play:<sup>4</sup> multiply the gradient vector by _η_ to determine the size of the downhill step (Equation 4-7). 

_Equation 4-7. Gradient descent step_ 

θ next step = θ − η∇θ MSE θ 

Let’s look at a quick implementation of this algorithm: 

```
eta=0.1# learning rate
n_epochs=1000
m=len(X_b)  # number of instances
```

```
np.random.seed(42)
theta=np.random.randn(2, 1)  # randomly initialized model parameters
```

```
forepochinrange(n_epochs):
gradients=2/m*X_b.T@ (X_b@theta-y)
theta=theta-eta*gradients
```

That wasn’t too hard! Each iteration over the training set is called an _epoch_ . Let’s look at the resulting `theta` : 

```
>>> theta
array([[4.21509616],
       [2.77011339]])
```

Hey, that’s exactly what the Normal equation found! Gradient descent worked per‐ fectly. But what if you had used a different learning rate ( `eta` )? Figure 4-8 shows the first 20 steps of gradient descent using three different learning rates. The line at the bottom of each plot represents the random starting point, then each epoch is represented by a darker and darker line. 

> 4 Eta ( _η_ ) is the seventh letter of the Greek alphabet. 

**Gradient Descent | 143** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 



<!-- Start of picture text -->
14 a a TEE a jd<br>ooZea0.5 1.0 1.5 2.0 0.0 0.5 1.0 5 2.0 0.0po0.5 1.0 15 2.0<br>X1 X1 x1<br><!-- End of picture text -->



<!-- Start of picture text -->
9,<br>a<br>9,<br><!-- End of picture text -->

at each iteration is called the _learning schedule_ . If the learning rate is reduced too quickly, you may get stuck in a local minimum, or even end up frozen halfway to the minimum. If the learning rate is reduced too slowly, you may jump around the minimum for a long time and end up with a suboptimal solution if you halt training too early. 

This code implements stochastic gradient descent using a simple learning schedule: 

```
n_epochs=50
t0, t1=5, 50# learning schedule hyperparameters
deflearning_schedule(t):
returnt0/ (t+t1)
np.random.seed(42)
theta=np.random.randn(2, 1)  # random initialization
```

```
forepochinrange(n_epochs):
foriterationinrange(m):
random_index=np.random.randint(m)
xi=X_b[random_index : random_index+1]
yi=y[random_index : random_index+1]
gradients=2*xi.T@ (xi@theta-yi)  # for SGD, do not divide by m
eta=learning_schedule(epoch*m+iteration)
theta=theta-eta*gradients
```

By convention we iterate by rounds of _m_ iterations; each round is called an _epoch_ , as earlier. While the batch gradient descent code iterated 1,000 times through the whole training set, this code goes through the training set only 50 times and reaches a pretty good solution: 

```
>>> theta
array([[4.21076011],
       [2.74856079]])
```

Figure 4-10 shows the first 20 steps of training (notice how irregular the steps are). 

Note that since instances are picked randomly, some instances may be picked several times per epoch, while others may not be picked at all. If you want to be sure that the algorithm goes through every instance at each epoch, another approach is to shuffle the training set (making sure to shuffle the input features and the labels jointly), then go through it instance by instance, then shuffle it again, and so on. However, this approach is more complex, and it generally does not improve the result. 

**146 | Chapter 4: Training Models** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 



<!-- Start of picture text -->
uf _| | J Jj Jf J J<br>of | | | fp ft } tt<br>a<br>asa<br>—_.| ee~~ |<br>2<br>0<br>0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00<br>X1<br><!-- End of picture text -->





<!-- Start of picture text -->
|<br><!-- End of picture text -->

| 



<!-- Start of picture text -->
3.4<br>32+—_——a— Stochasticpn patch % Ae<br>3.0 Ls Catal ph/<br>2.8 a<br>2.6 i Cia 4<br>ra’<br>aa<br>2.4<br>2.75 3.00 3.25 3.50 3.75 4.00 4.25 4.50<br>8<br><!-- End of picture text -->



<!-- Start of picture text -->
: Pf ff oe<br>|6 Pf ff ft de<br>. eae nee are ee<br>)<br>-3 -2 -1 ) 1 2 3<br>Xl<br><!-- End of picture text -->



<!-- Start of picture text -->
108 a ee (4<br>Jo<br>eh<br>Pspee<br>0<br>3 —2 -1 0 1 2 3<br>X1<br><!-- End of picture text -->





<!-- Start of picture text -->
10<br>34 ~-- 2 degrees ri<br>x<br>4 LE eng TT<br>wldM ee ll<br>3 -2 -1 0 1 2 3<br>X1<br><!-- End of picture text -->



<!-- Start of picture text -->
2.5<br>—= valid<br>20<br>on<br>=<br>/<br>(eee<br>0.0<br>0 10 20 30 40 50 60 70 80<br>Training set size<br><!-- End of picture text -->





<!-- Start of picture text -->
2.5<br>;ilinn ~<br>Ww 1.5 L |<br>ac 1.0 |“ es |<br>0 10 20 30 40 50 60 70 80<br>Training set size<br><!-- End of picture text -->

hallmark of an overfitting model. If you used a much larger training set, however, the two curves would continue to get closer. 



One way to improve an overfitting model is to feed it more training data until the validation error reaches the training error. 

#### **The Bias/Variance Trade-Off** 

An important theoretical result of statistics and machine learning is the fact that a model’s generalization error can be expressed as the sum of three very different errors: 

##### _Bias_ 

This part of the generalization error is due to wrong assumptions, such as assuming that the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data.<sup>6</sup> 

##### _Variance_ 

This part is due to the model’s excessive sensitivity to small variations in the training data. A model with many degrees of freedom (such as a high-degree polynomial model) is likely to have high variance and thus overfit the training data. 

##### _Irreducible error_ 

This part is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data (e.g., fix the data sources, such as broken sensors, or detect and remove outliers). 

Increasing a model’s complexity will typically increase its variance and reduce its bias. Conversely, reducing a model’s complexity increases its bias and reduces its variance. This is why it is called a trade-off. 

## **Regularized Linear Models** 

As you saw in Chapters 1 and 2, a good way to reduce overfitting is to regularize the model (i.e., to constrain it): the fewer degrees of freedom it has, the harder it will be for it to overfit the data. A simple way to regularize a polynomial model is to reduce the number of polynomial degrees. 

> 6 This notion of bias is not to be confused with the bias term of linear models. 

**Regularized Linear Models | 155** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 

For a linear model, regularization is typically achieved by constraining the weights of the model. We will now look at ridge regression, lasso regression, and elastic net regression, which implement three different ways to constrain the weights. 

### **Ridge Regression** 

_Ridge regression_ (also called _Tikhonov regularization_ ) is a regularized version of linear regression: a _regularization term_ equal to mα<sup>∑</sup> n<sup>i= 1</sup> θi 2 is added to the MSE. This forces 

the learning algorithm to not only fit the data but also keep the model weights as small as possible. Note that the regularization term should only be added to the cost function during training. Once the model is trained, you want to use the unregularized MSE (or the RMSE) to evaluate the model’s performance. 

The hyperparameter _α_ controls how much you want to regularize the model. If _α_ = 0, then ridge regression is just linear regression. If _α_ is very large, then all weights end up very close to zero and the result is a flat line going through the data’s mean. Equation 4-8 presents the ridge regression cost function.<sup>7</sup> 

_Equation 4-8. Ridge regression cost function_ 



Note that the bias term _θ_ 0 is not regularized (the sum starts at _i_ = 1, not 0). If we define **w** as the vector of feature weights ( _θ_ 1 to _θn_ ), then the regularization term is equal to _α_ ( ∥ **w** ∥ 2)<sup>2</sup> / _m_ , where ∥ **w** ∥ 2 represents the ℓ2 norm of the weight vector.<sup>8</sup> For batch gradient descent, just add 2 _α_ **w** / _m_ to the part of the MSE gradient vector that corresponds to the feature weights, without adding anything to the gradient of the bias term (see Equation 4-6). 



It is important to scale the data (e.g., using a `StandardScaler` ) before performing ridge regression, as it is sensitive to the scale of the input features. This is true of most regularized models. 

Figure 4-17 shows several ridge models that were trained on some very noisy linear data using different _α_ values. On the left, plain ridge models are used, leading to linear predictions. On the right, the data is first expanded using 

> 7 It is common to use the notation _J_ ( **θ** ) for cost functions that don’t have a short name; I’ll often use this notation throughout the rest of this book. The context will make it clear which cost function is being discussed. 

> 8 Norms are discussed in Chapter 2. 

**156 | Chapter 4: Training Models** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 



<!-- Start of picture text -->
weeeee a=0 weeeee a=0<br>3.0<br>°<br>i ee ee |.<br>15 SF —Zi|<br>ors<br>0.0<br>0.0 0.5 1.0 1.5 2.0 2.5 3.0 0.0 0.5 1.0 1.5 2.0 2.5 3.0<br>X1 X1<br><!-- End of picture text -->

And using stochastic gradient descent:<sup>10</sup> 

```
>>> sgd_reg=SGDRegressor(penalty="l2", alpha=0.1/m, tol=None,
... max_iter=1000, eta0=0.01, random_state=42)
...
>>> sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets
>>> sgd_reg.predict([[1.5]])
array([1.55302613])
```

The `penalty` hyperparameter sets the type of regularization term to use. Specifying `"l2"` indicates that you want SGD to add a regularization term to the MSE cost function equal to `alpha` times the square of the ℓ2 norm of the weight vector. This is just like ridge regression, except there’s no division by _m_ in this case; that’s why we passed `alpha=0.1 / m` , to get the same result as `Ridge(alpha=0.1)` . 



The `RidgeCV` class also performs ridge regression, but it automat‐ ically tunes hyperparameters using cross-validation. It’s roughly equivalent to using `GridSearchCV` , but it’s optimized for ridge regression and runs _much_ faster. Several other estimators (mostly linear) also have efficient CV variants, such as `LassoCV` and `ElasticNetCV` . 

### **Lasso Regression** 

_Least absolute shrinkage and selection operator regression_ (usually simply called _lasso regression_ ) is another regularized version of linear regression: just like ridge regres‐ sion, it adds a regularization term to the cost function, but it uses the ℓ1 norm of the weight vector instead of the square of the ℓ2 norm (see Equation 4-10). Notice that the ℓ1 norm is multiplied by 2 _α_ , whereas the ℓ2 norm was multiplied by _α_ / _m_ in ridge regression. These factors were chosen to ensure that the optimal _α_ value is independent from the training set size: different norms lead to different factors (see Scikit-Learn issue #15657 for more details). 

_Equation 4-10. Lasso regression cost function_ 

J θ = MSE θ + 2 α ∑ ni = 1 θi 

Figure 4-18 shows the same thing as Figure 4-17 but replaces the ridge models with lasso models and uses different _α_ values. 

> 10 Alternatively, you can use the `Ridge` class with the `"sag"` solver. Stochastic average GD is a variant of stochastic GD. For more details, see the presentation “Minimizing Finite Sums with the Stochastic Average Gradient Algorithm” by Mark Schmidt et al. from the University of British Columbia. 

###### **158 | Chapter 4: Training Models** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 



<!-- Start of picture text -->
weeeee a=0 weeeee a=0<br>3.0<br>;<br>ola.e Td we° a .<br>15 = =we - . “7%.o : :<br>paneer ||) bea||<br>0.0<br>0.0 0.5 1.0 1.5 2.0 2.5 3.0 0.0 0.5 1.0 1.5 2.0 2.5 3.0<br>X1 X1<br><!-- End of picture text -->



<!-- Start of picture text -->
1s 2, penalty Lasso<br>1.06 —(rs ;-¥—_}| | | | a __ ae<br>92 4.0 ee |<br>-0.5 |<br>-1.0 1 ~ |NN\ \o<br>-15<br>Lz penalty Ridge<br>1s<br>92 060.5 | —}——_ pot |ee<br>-0.5 |<br>-1.0 |<br>-15<br>-10 -05 00 O05 10 415 20 25 30 -10 -05 00 O05 10 415 20 25 3.0<br>0, 01<br><!-- End of picture text -->



The lasso cost function is not differentiable at _θi_ = 0 (for _i_ = 1, 2, ⋯ , _n_ ), but gradient descent still works if you use a _subgradient vector_ **g**<sup>11</sup> instead when any _θi_ = 0. Equation 4-11 shows a subgradient vector equation you can use for gradient descent with the lasso cost function. 

_Equation 4-11. Lasso regression subgradient vector_ 



Here is a small Scikit-Learn example using the `Lasso` class: 

```
>>> fromsklearn.linear_modelimportLasso
>>> lasso_reg=Lasso(alpha=0.1)
>>> lasso_reg.fit(X, y)
>>> lasso_reg.predict([[1.5]])
array([1.53788174])
```

Note that you could instead use `SGDRegressor(penalty="l1", alpha=0.1)` . 

### **Elastic Net Regression** 

_Elastic net regression_ is a middle ground between ridge regression and lasso regres‐ sion. The regularization term is a weighted sum of both ridge and lasso’s regulariza‐ tion terms, and you can control the mix ratio _r_ . When _r_ = 0, elastic net is equivalent to ridge regression, and when _r_ = 1, it is equivalent to lasso regression (Equation 4-12). 

_Equation 4-12. Elastic net cost function_ 



So when should you use elastic net regression, or ridge, lasso, or plain linear regres‐ sion (i.e., without any regularization)? It is almost always preferable to have at least a little bit of regularization, so generally you should avoid plain linear regression. Ridge is a good default, but if you suspect that only a few features are useful, you should prefer lasso or elastic net because they tend to reduce the useless features’ weights down to zero, as discussed earlier. In general, elastic net is preferred over 

> 11 You can think of a subgradient vector at a nondifferentiable point as an intermediate vector between the gradient vectors around that point. 

**Regularized Linear Models | 161** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 



<!-- Start of picture text -->
3.5<br>uu 2.0 ON pescadoBe node q<br>a) \<br>Ne<br>a 1.5 iPr rr leerrr en et>iae<br>0.0<br>0 100 200 300 400 500<br>Epoch<br><!-- End of picture text -->



With stochastic and mini-batch gradient descent, the curves are not so smooth, and it may be hard to know whether you have reached the minimum or not. One solution is to stop only after the validation error has been above the minimum for some time (when you are confident that the model will not do any better), then roll back the model parameters to the point where the validation error was at a minimum. 

Here is a basic implementation of early stopping: 

```
fromcopyimportdeepcopy
fromsklearn.metricsimportroot_mean_squared_error
fromsklearn.preprocessingimportStandardScaler
X_train, y_train, X_valid, y_valid= [...]  # split the quadratic dataset
preprocessing=make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
StandardScaler())
X_train_prep=preprocessing.fit_transform(X_train)
X_valid_prep=preprocessing.transform(X_valid)
sgd_reg=SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs=500
best_valid_rmse=float('inf')
forepochinrange(n_epochs):
sgd_reg.partial_fit(X_train_prep, y_train)
y_valid_predict=sgd_reg.predict(X_valid_prep)
val_error=root_mean_squared_error(y_valid, y_valid_predict)
ifval_error<best_valid_rmse:
best_valid_rmse=val_error
best_model=deepcopy(sgd_reg)
```

This code first adds the polynomial features and scales all the input features, both for the training set and for the validation set (the code assumes that you have split the original training set into a smaller training set and a validation set). Then it creates an `SGDRegressor` model with no regularization and a small learning rate. In the training loop, it calls `partial_fit()` instead of `fit()` , to perform incremental learning. At each epoch, it measures the RMSE on the validation set. If it is lower than the lowest RMSE seen so far, it saves a copy of the model in the `best_model` variable. This implementation does not actually stop training, but it lets you revert to the best model after training. Note that the model is copied using `copy.deepcopy()` , because it copies both the model’s hyperparameters _and_ the learned parameters. In contrast, `sklearn.base.clone()` only copies the model’s hyperparameters. 

**Regularized Linear Models | 163** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 

~ () (¢ ) 

() ~~—T7~~ 



<!-- Start of picture text -->
1.00<br>ff =H i ee<br>eeGO|<br>-6 -4 -2 0 2 4 6<br>t<br><!-- End of picture text -->

Once the logistic regression model has estimated the probability p = _h_ **θ** ( **x** ) that an instance **x** belongs to the positive class, it can make its prediction _ŷ_ easily (see Equation 4-15). 

_Equation 4-15. Logistic regression model prediction using a 50% threshold probability_ 



Notice that _σ_ ( _t_ ) < 0.5 when _t_ < 0, and _σ_ ( _t_ ) ≥ 0.5 when _t_ ≥ 0, so a logistic regression model using the default threshold of 50% probability predicts 1 if **θ**<sup>⊺</sup> **x** is positive and 0 if it is negative. 



The score _t_ is often called the _logit_ . The name comes from the fact that the logit function, defined as logit( _p_ ) = log( _p_ / (1 – _p_ )), is the inverse of the logistic function. Indeed, if you compute the logit of the estimated probability _p_ , you will find that the result is _t_ . The logit is also called the _log-odds_ , since it is the log of the ratio between the estimated probability for the positive class and the estimated probability for the negative class. 

### **Training and Cost Function** 

Now you know how a logistic regression model estimates probabilities and makes predictions. But how is it trained? The objective of training is to set the parameter vector **θ** so that the model estimates high probabilities for positive instances ( _y_ = 1) and low probabilities for negative instances ( _y_ = 0). This idea is captured by the cost function shown in Equation 4-16 for a single training instance **x** . 

_Equation 4-16. Cost function of a single training instance_ 



This cost function makes sense because –log( _t_ ) grows very large when _t_ approaches 0, so the cost will be large if the model estimates a probability close to 0 for a positive instance, and it will also be large if the model estimates a probability close to 1 for a negative instance. On the other hand, –log( _t_ ) is close to 0 when _t_ is close to 1, so the cost will be close to 0 if the estimated probability is close to 0 for a negative instance or close to 1 for a positive instance, which is precisely what we want. 

**Logistic Regression | 165** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 

The cost function over the whole training set is the average cost over all training instances. It can be written in a single expression called the _log loss_ , shown in Equation 4-17. 

##### _Equation 4-17. Logistic regression cost function (log loss)_ 





The log loss was not just pulled out of a hat. It can be shown mathematically (using Bayesian inference) that minimizing this loss will result in the model with the _maximum likelihood_ of being optimal, assuming that the instances follow a Gaussian distribution around the mean of their class. When you use the log loss, this is the implicit assumption you are making. The more wrong this assumption is, the more biased the model will be. Similarly, when we used the MSE to train linear regression models, we were implic‐ itly assuming that the data was purely linear, plus some Gaussian noise. So, if the data is not linear (e.g., if it’s quadratic) or if the noise is not Gaussian (e.g., if outliers are not exponentially rare), then the model will be biased. 

The bad news is that there is no known closed-form equation to compute the value of **θ** that minimizes this cost function (there is no equivalent of the Normal equation). But the good news is that this cost function is convex, so gradient descent (or any other optimization algorithm) is guaranteed to find the global minimum (if the learning rate is not too large and you wait long enough). The partial derivatives of the cost function with regard to the _j_<sup>th</sup> model parameter _θj_ are given by Equation 4-18. 

_Equation 4-18. Logistic cost function partial derivatives_ 



This equation looks very much like Equation 4-5: for each instance it computes the prediction error and multiplies it by the _j_<sup>th</sup> feature value, and then it computes the average over all training instances. Once you have the gradient vector containing all the partial derivatives, you can use it in the batch gradient descent algorithm. That’s it: you now know how to train a logistic regression model. For stochastic GD you would take one instance at a time, and for mini-batch GD you would use a mini-batch at a time. 

**166 | Chapter 4: Training Models** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 

### **Decision Boundaries** 

We can use the iris dataset to illustrate logistic regression. This is a famous dataset that contains the sepal and petal length and width of 150 iris flowers of three different species: _Iris setosa_ , _Iris versicolor_ , and _Iris virginica_ (see Figure 4-22). 



_Figure 4-22. Flowers of three iris plant species_<sup>_12_</sup> 

Let’s try to build a classifier to detect the _Iris virginica_ type based only on the petal width feature. The first step is to load the data and take a quick peek: 

```
>>> fromsklearn.datasetsimportload_iris
>>> iris=load_iris(as_frame=True)
>>> list(iris)
['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names',
 'filename', 'data_module']
>>> iris.data.head(3)
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
>>> iris.target.head(3)  # note that the instances are not shuffled
0    0
1    0
2    0
Name: target, dtype: int64
>>> iris.target_names
array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
```

> 12 Photos reproduced from the corresponding Wikipedia pages. _Iris virginica_ photo by Frank Mayfield (Creative Commons BY-SA 2.0), _Iris versicolor_ photo by D. Gordon E. Robertson (Creative Commons BY-SA 3.0), _Iris setosa_ photo public domain. 

**Logistic Regression | 167** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 



<!-- Start of picture text -->
eee<br>a a e ae ee We ee ee We we e r ee ee ———<br>0.8 ee<<br>= 6777 Not Iris virginica proba<br>Fo) — Mevioinea<br>noes OK<br>= : SS<br>:<br>0.20.0 - wpiZii ge s| ~<br>0.0 0.5 1.0 15 2.0 2.5 3.0<br>Petal width (cm)<br><!-- End of picture text -->



<!-- Start of picture text -->
Petal length<br><!-- End of picture text -->



Just like the other linear models, logistic regression models can be regularized using ℓ1 or ℓ2 penalties. Scikit-Learn actually adds an ℓ2 penalty by default. 

### **Softmax Regression** 

The logistic regression model can be generalized to support multiple classes directly, without having to train and combine multiple binary classifiers (as discussed in Chapter 3). This is called _softmax regression_ , or _multinomial logistic regression_ . 

The idea is simple: when given an instance **x** , the softmax regression model first computes a score _sk_ ( **x** ) for each class _k_ , then estimates the probability of each class by applying the _softmax function_ (also called the _normalized exponential_ ) to the scores. The equation to compute _sk_ ( **x** ) should look familiar, as it is just like the equation for linear regression prediction (see Equation 4-19). 

_Equation 4-19. Softmax score for class k_ 



Note that each class has its own dedicated parameter vector **θ**<sup>(</sup><sup>_k_)</sup> . All these vectors are typically stored as rows in a _parameter matrix_ **Θ** . 

Once you have computed the score of every class for the instance **x** , you can estimate the probability p k<sup>that the instance belongs to class</sup><sup>_k_by running the scores through</sup> the softmax function (Equation 4-20). The function computes the exponential of every score, then normalizes them (dividing by the sum of all the exponentials). The scores are generally called logits or log-odds (although they are actually unnormal‐ ized log-odds). 

_Equation 4-20. Softmax function_ 



In this equation: 

- _K_ is the number of classes. 

- **s** ( **x** ) is a vector containing the scores of each class for the instance **x** . 

- _σ_ ( **s** ( **x** )) _k_ is the estimated probability that the instance **x** belongs to class _k_ , given the scores of each class for that instance. 

**170 | Chapter 4: Training Models** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 

Just like the logistic regression classifier, by default the softmax regression classifier predicts the class with the highest estimated probability (which is simply the class with the highest score), as shown in Equation 4-21. 

##### _Equation 4-21. Softmax regression classifier prediction_ 



The _argmax_ operator returns the value of a variable that maximizes a function. In this equation, it returns the value of _k_ that maximizes the estimated probability _σ_ ( **s** ( **x** )) _k_ . 



The softmax regression classifier predicts only one class at a time (i.e., it is multiclass, not multioutput), so it should be used only with mutually exclusive classes, such as different species of plants. You cannot use it to recognize multiple people in one picture. 

Now that you know how the model estimates probabilities and makes predictions, let’s take a look at training. The objective is to have a model that estimates a high probability for the target class (and consequently a low probability for the other classes). Minimizing the cost function shown in Equation 4-22, called the _cross entropy_ , should lead to this objective because it penalizes the model when it estimates a low probability for a target class. Cross entropy is frequently used to measure how well a set of estimated class probabilities matches the target classes. 

##### _Equation 4-22. Cross entropy cost function_ 



i In this equation, yk is the target probability that the _i_<sup>th</sup> instance belongs to class _k_ . In general, it is either equal to 1 or 0, depending on whether the instance belongs to the class or not. 

Notice that when there are just two classes ( _K_ = 2), this cost function is equivalent to the logistic regression cost function (log loss; see Equation 4-17). 

#### **Cross Entropy** 

Cross entropy originated from Claude Shannon’s _information theory_ . Suppose you want to efficiently transmit information about the weather every day. If there are eight options (sunny, rainy, etc.), you could encode each option using 3 bits, because 2<sup>3</sup> = 8. However, if you think it will be sunny almost every day, it would be much more 

**Logistic Regression | 171** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 

efficient to code “sunny” on just one bit (0) and the other seven options on four bits (starting with a 1). Cross entropy measures the average number of bits you actually send per option. If your assumption about the weather is perfect, cross entropy will be equal to the entropy of the weather itself (i.e., its intrinsic unpredictability). But if your assumption is wrong (e.g., if it rains often), cross entropy will be greater by an amount called the _Kullback–Leibler (KL) divergence_ . 

The cross entropy between two probability distributions _p_ and _q_ is defined as _H_ ( _p_ , _q_ ) = –Σ _x p_ ( _x_ ) log _q_ ( _x_ ) (at least when the distributions are discrete). For more details, check out my video on the subject. 

The gradient vector of this cost function with regard to **θ**<sup>(</sup><sup>_k_)</sup> is given by Equation 4-23. 

_Equation 4-23. Cross entropy gradient vector for class k_ 



Now you can compute the gradient vector for every class, then use gradient descent (or any other optimization algorithm) to find the parameter matrix **Θ** that minimizes the cost function. 

Let’s use softmax regression to classify the iris plants into all three classes. ScikitLearn’s `LogisticRegression` classifier uses softmax regression automatically when you train it on more than two classes (assuming you use `solver="lbfgs"` , which is the default). It also applies ℓ2 regularization by default, which you can control using the hyperparameter `C` , as mentioned earlier: 

```
X=iris.data[["petal length (cm)", "petal width (cm)"]].values
y=iris["target"]
X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=42)
softmax_reg=LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)
```

So the next time you find an iris with petals that are 5 cm long and 2 cm wide, you can ask your model to tell you what type of iris it is, and it will answer _Iris virginica_ (class 2) with 96% probability (or _Iris versicolor_ with 4% probability): 

```
>>> softmax_reg.predict([[5, 2]])
array([2])
>>> softmax_reg.predict_proba([[5, 2]]).round(2)
array([[0.  , 0.04, 0.96]])
```

Figure 4-25 shows the resulting decision boundaries, represented by the background colors. Notice that the decision boundaries between any two classes are linear. The figure also shows the probabilities for the _Iris versicolor_ class, represented by the 

###### **172 | Chapter 4: Training Models** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 



<!-- Start of picture text -->
8 rd |<br>ae sinl<br>Boo 4 Iris virginica<br>3elsa© 3. [Jr! =e IrisIrisP  setosa versicolor ANeliie rl ;<br>0.5 Oo<br>f 0<br>0.0<br>1 2 3 4 5 6 7<br>Petal length<br><!-- End of picture text -->

**6.** Is it a good idea to stop mini-batch gradient descent immediately when the validation error goes up? 

**7.** Which gradient descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the others converge as well? 

**8.** Suppose you are using polynomial regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this? 

**9.** Suppose you are using ridge regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter _α_ or reduce it? 

**10.** Why would you want to use: 

   - **a.** Ridge regression instead of plain linear regression (i.e., without any regularization)? 

   - **b.** Lasso instead of ridge regression? 

   - **c.** Elastic net instead of lasso regression? 

**11.** Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two logistic regression classifiers or one softmax regres‐ sion classifier? 

**12.** Implement batch gradient descent with early stopping for softmax regression without using Scikit-Learn, only NumPy. Use it on a classification task such as the iris dataset. 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

###### **174 | Chapter 4: Training Models** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:14:47. 

