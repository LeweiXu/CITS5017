# **APPENDIX B Autodiff** 

This appendix explains how TensorFlow’s autodifferentiation (autodiff) feature works, and how it compares to other solutions. 

Suppose you define a function _f_ ( _x_ , _y_ ) = _x_<sup>2</sup> _y_ + _y_ + 2, and you need its partial derivatives ∂ _f_ /∂ _x_ and ∂ _f_ /∂ _y_ , typically to perform gradient descent (or some other optimization algorithm). Your main options are manual differentiation, finite difference approxi‐ mation, forward-mode autodiff, and reverse-mode autodiff. TensorFlow implements reverse-mode autodiff, but to understand it, it’s useful to look at the other options first. So let’s go through each of them, starting with manual differentiation. 

## **Manual Differentiation** 

The first approach to compute derivatives is to pick up a pencil and a piece of paper and use your calculus knowledge to derive the appropriate equation. For the function _f_ ( _x_ , _y_ ) just defined, it is not too hard; you just need to use five rules: 

- The derivative of a constant is 0. 

- The derivative of _λx_ is _λ_ (where _λ_ is a constant). 

- The derivative of _x_<sup>λ</sup> is _λx_<sup>_λ_–1</sup> , so the derivative of _x_<sup>2</sup> is 2 _x_ . 

- The derivative of a sum of functions is the sum of these functions’ derivatives. 

- The derivative of _λ_ times a function is _λ_ times its derivative. 

**785** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:00. 

From these rules, you can derive Equation B-1. 

_Equation B-1. Partial derivatives of f(x, y)_ 



This approach can become very tedious for more complex functions, and you run the risk of making mistakes. Fortunately, there are other options. Let’s look at finite difference approximation now. 

## **Finite Difference Approximation** 

Recall that the derivative _h_ ′ ( _x_ 0) of a function _h_ ( _x_ ) at a point _x_ 0 is the slope of the function at that point. More precisely, the derivative is defined as the limit of the slope of a straight line going through this point _x_ 0 and another point _x_ on the function, as _x_ gets infinitely close to _x_ 0 (see Equation B-2). 

_Equation B-2. Definition of the derivative of a function h(x) at point x0_ 



So, if we wanted to calculate the partial derivative of _f_ ( _x_ , _y_ ) with regard to _x_ at _x_ = 3 and _y_ = 4, we could compute _f_ (3 + _ε_ , 4) – _f_ (3, 4) and divide the result by _ε_ , using a very small value for _ε_ . This type of numerical approximation of the derivative is called a _finite difference approximation_ , and this specific equation is called _Newton’s difference quotient_ . That’s exactly what the following code does: 

```
deff(x, y):
returnx**2*y+y+2
defderivative(f, x, y, x_eps, y_eps):
return (f(x+x_eps, y+y_eps) -f(x, y)) / (x_eps+y_eps)
df_dx=derivative(f, 3, 4, 0.00001, 0)
df_dy=derivative(f, 3, 4, 0, 0.00001)
```

Unfortunately, the result is imprecise (and it gets worse for more complicated func‐ tions). The correct results are respectively 24 and 10, but instead we get: 

**786 | Appendix B: Autodiff** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:00. 

```
>>> df_dx
24.000039999805264
>>> df_dy
10.000000000331966
```

Notice that to compute both partial derivatives, we have to call `f()` at least three times (we called it four times in the preceding code, but it could be optimized). If there were 1,000 parameters, we would need to call `f()` at least 1,001 times. When you are dealing with large neural networks, this makes finite difference approximation way too inefficient. 

However, this method is so simple to implement that it is a great tool to check that the other methods are implemented correctly. For example, if it disagrees with your manually derived function, then your function probably contains a mistake. 

So far, we have considered two ways to compute gradients: using manual differentia‐ tion and using finite difference approximation. Unfortunately, both are fatally flawed for training a large-scale neural network. So let’s turn to autodiff, starting with forward mode. 

## **Forward-Mode Autodiff** 

Figure B-1 shows how forward-mode autodiff works on an even simpler function, _g_ ( _x_ , _y_ ) = 5 + _xy_ . The graph for that function is represented on the left. After forwardmode autodiff, we get the graph on the right, which represents the partial derivative ∂ _g_ /∂ _x_ = 0 + (0 × _x_ + _y_ × 1) = _y_ (we could similarly obtain the partial derivative with regard to _y_ ). 

The algorithm will go through the computation graph from the inputs to the outputs (hence the name “forward mode”). It starts by getting the partial derivatives of the leaf nodes. The constant node (5) returns the constant 0, since the derivative of a constant is always 0. The variable _x_ returns the constant 1 since ∂ _x_ /∂ _x_ = 1, and the variable _y_ returns the constant 0 since ∂ _y_ /∂ _x_ = 0 (if we were looking for the partial derivative with regard to _y_ , it would be the reverse). 

Now we have all we need to move up the graph to the multiplication node in function _g_ . Calculus tells us that the derivative of the product of two functions _u_ and _v_ is ∂( _u_ × _v_ )/∂ _x_ = ∂ _v_ /∂ _x_ × _u_ + _v_ × ∂ _u_ /∂ _x_ . We can therefore construct a large part of the graph on the right, representing 0 × _x_ + _y_ × 1. 

Finally, we can go up to the addition node in function _g_ . As mentioned, the derivative of a sum of functions is the sum of these functions’ derivatives, so we just need to create an addition node and connect it to the parts of the graph we have already computed. We get the correct partial derivative: ∂ _g_ /∂ _x_ = 0 + (0 × _x_ + _y_ × 1). 

**Autodiff | 787** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:00. 



<!-- Start of picture text -->
eeRie<br>ro<br>| oe<br>[=] x | :<br>s<br>;<br>a<br>: - 4____ Ox0x/Ox=1a oF 0 ~<br>g( a ; g/o x=0 O-x8"Xt yl )= y<br><!-- End of picture text -->

( ) ( ) ( )C ) CC) ( )C ) ( ) ( ) 



<!-- Start of picture text -->
( )<br><!-- End of picture text -->



<!-- Start of picture text -->
FOG y= wy ty +2 f3,4)=42<br>42+24¢ af _<br>8F 3.4)=24<br>36+24¢€ 6<br>9+6e7%2 4 4 2<br>3tEC+)3te o<br>4<br>, atbe with e-0 '<br>not as 42.000024<br><!-- End of picture text -->



<!-- Start of picture text -->
F(x, y) = x?y +y +2 df/dn7=1<br>naa)<br>=1x1=1 =1*1=1<br>of/dn,=0f/dn,xdn./dn, of/dn5=df/dn,*dn./dn,<br>=1xn5=4 =1xn ,=9<br>(1) (2)<br>3<br>n, x Of/dy=df/dn,+d0f/dn,=9+1=10<br>(1) (2)<br>Of/Ox=n,*44+n,*x4=24<br><!-- End of picture text -->

The idea is to gradually go down the graph, computing the partial derivative of _f_ ( _x_ , _y_ ) with regard to each consecutive node, until we reach the variable nodes. For this, reverse-mode autodiff relies heavily on the _chain rule_ , shown in Equation B-4. 

_Equation B-4. Chain rule_ 



Since _n_ 7 is the output node, _f_ = _n_ 7 so ∂ _f_ / ∂ _n_ 7 = 1. 

Let’s continue down the graph to _n_ 5: how much does _f_ vary when _n_ 5 varies? The answer is ∂ _f_ / ∂ _n_ 5 = ∂ _f_ / ∂ _n_ 7 × ∂ _n_ 7 / ∂ _n_ 5. We already know that ∂ _f_ / ∂ _n_ 7 = 1, so all we need is ∂ _n_ 7 / ∂ _n_ 5. Since _n_ 7 simply performs the sum _n_ 5 + _n_ 6, we find that ∂ _n_ 7 / ∂ _n_ 5 = 1, so ∂ _f_ / ∂ _n_ 5 = 1 × 1 = 1. 

Now we can proceed to node _n_ 4: how much does _f_ vary when _n_ 4 varies? The answer is ∂ _f_ / ∂ _n_ 4 = ∂ _f_ / ∂ _n_ 5 × ∂ _n_ 5 / ∂ _n_ 4. Since _n_ 5 = _n_ 4 × _n_ 2, we find that ∂ _n_ 5 / ∂ _n_ 4 = _n_ 2, so ∂ _f_ / ∂ _n_ 4 = 1 × _n_ 2 = 4. 

The process continues until we reach the bottom of the graph. At that point we will have calculated all the partial derivatives of _f_ ( _x_ , _y_ ) at the point _x_ = 3 and _y_ = 4. In this example, we find ∂ _f_ / ∂ _x_ = 24 and ∂ _f_ / ∂ _y_ = 10. Sounds about right! 

Reverse-mode autodiff is a very powerful and accurate technique, especially when there are many inputs and few outputs, since it requires only one forward pass plus one reverse pass per output to compute all the partial derivatives for all outputs with regard to all the inputs. When training neural networks, we generally want to minimize the loss, so there is a single output (the loss), and hence only two passes through the graph are needed to compute the gradients. Reverse-mode autodiff can also handle functions that are not entirely differentiable, as long as you ask it to compute the partial derivatives at points that are differentiable. 

In Figure B-3, the numerical results are computed on the fly, at each node. However, that’s not exactly what TensorFlow does: instead, it creates a new computation graph. In other words, it implements _symbolic_ reverse-mode autodiff. This way, the compu‐ tation graph to compute the gradients of the loss with regard to all the parameters in the neural network only needs to be generated once, and then it can be executed over and over again, whenever the optimizer needs to compute the gradients. Moreover, this makes it possible to compute higher-order derivatives if needed. 

**Autodiff | 791** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:00. 



If you ever want to implement a new type of low-level TensorFlow operation in C++, and you want to make it compatible with auto‐ diff, then you will need to provide a function that returns the par‐ tial derivatives of the function’s outputs with regard to its inputs. For example, suppose you implement a function that computes the square of its input: _f_ ( _x_ ) = _x_<sup>2</sup> . In that case you would need to provide the corresponding derivative function: _f_ ′ ( _x_ ) = 2 _x_ . 

**792 | Appendix B: Autodiff** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:19:00. 

