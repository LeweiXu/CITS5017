# **CHAPTER 15 Processing Sequences Using RNNs and CNNs** 

Predicting the future is something you do all the time, whether you are finishing a friend’s sentence or anticipating the smell of coffee at breakfast. In this chapter we will discuss recurrent neural networks (RNNs)—a class of nets that can predict the future (well, up to a point). RNNs can analyze time series data, such as the number of daily active users on your website, the hourly temperature in your city, your home’s daily power consumption, the trajectories of nearby cars, and more. Once an RNN learns past patterns in the data, it is able to use its knowledge to forecast the future, assuming of course that past patterns still hold in the future. 

More generally, RNNs can work on sequences of arbitrary lengths, rather than on fixed-sized inputs. For example, they can take sentences, documents, or audio samples as input, making them extremely useful for natural language processing applications such as automatic translation or speech-to-text. 

In this chapter, we will first go through the fundamental concepts underlying RNNs and how to train them using backpropagation through time. Then, we will use them to forecast a time series. Along the way, we will look at the popular ARMA family of models, often used to forecast time series, and use them as baselines to compare with our RNNs. After that, we’ll explore the two main difficulties that RNNs face: 

- Unstable gradients (discussed in Chapter 11), which can be alleviated using vari‐ ous techniques, including _recurrent dropout_ and _recurrent layer normalization_ . 

- A (very) limited short-term memory, which can be extended using LSTM and GRU cells. 

**537** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 



<!-- Start of picture text -->
y Yo) fy J) V3)<br>0 (Y»\ (4\\ Y'\ YY<br>\z/ VE/J KT \ Ke<br>x Xo) Xa) X() X(3)<br>—_—_————PTime<br><!-- End of picture text -->



<!-- Start of picture text -->
t<br>y<br>eS io In So<br>—\\ (ZY) YD YN OD|| § wo ot<br>GZ IOI MAI<br>SSSSS —<br>BSS) Xo) Xa) X02)<br>X _—_——> Time<br><!-- End of picture text -->



<!-- Start of picture text -->
QO ( Oo ( ) )<br>QO (0 (cy) )<br>(oo « l) | |<br><!-- End of picture text -->

In this equation: 

- **Ŷ** ( _t_ ) is an _m_ × _n_ neurons matrix containing the layer’s outputs at time step _t_ for each instance in the mini-batch ( _m_ is the number of instances in the mini-batch and _n_ neurons is the number of neurons). 

- **X** ( _t_ ) is an _m_ × _n_ inputs matrix containing the inputs for all instances ( _n_ inputs is the number of input features). 

- **W** _x_ is an _n_ inputs × _n_ neurons matrix containing the connection weights for the inputs of the current time step. 

- **W** _ŷ_ is an _n_ neurons × _n_ neurons matrix containing the connection weights for the outputs of the previous time step. 

- **b** is a vector of size _n_ neurons containing each neuron’s bias term. 

- The weight matrices **W** _x_ and **W** _ŷ_ are often concatenated vertically into a single weight matrix **W** of shape ( _n_ inputs + _n_ neurons) × _n_ neurons (see the second line of Equation 15-2). 

- The notation [ **X** ( _t_ ) **Ŷ** ( _t_ –1)] represents the horizontal concatenation of the matrices **X** ( _t_ ) and **Ŷ** ( _t_ –1). 

Notice that **Ŷ** ( _t_ ) is a function of **X** ( _t_ ) and **Ŷ** ( _t_ –1), which is a function of **X** ( _t_ –1) and **Ŷ** ( _t_ –2), which is a function of **X** ( _t_ –2) and **Ŷ** ( _t_ –3), and so on. This makes **Ŷ** ( _t_ ) a function of all the inputs since time _t_ = 0 (that is, **X** (0), **X** (1), …, **X** ( _t_ )). At the first time step, _t_ = 0, there are no previous outputs, so they are typically assumed to be all zeros. 

### **Memory Cells** 

Since the output of a recurrent neuron at time step _t_ is a function of all the inputs from previous time steps, you could say it has a form of _memory_ . A part of a neural network that preserves some state across time steps is called a _memory cell_ (or simply a _cell_ ). A single recurrent neuron, or a layer of recurrent neurons, is a very basic cell, capable of learning only short patterns (typically about 10 steps long, but this varies depending on the task). Later in this chapter, we will look at some more complex and powerful types of cells capable of learning longer patterns (roughly 10 times longer, but again, this depends on the task). 

A cell’s state at time step _t_ , denoted **h** ( _t_ ) (the “h” stands for “hidden”), is a function of some inputs at that time step and its state at the previous time step: **h** ( _t_ ) = _f_ ( **x** ( _t_ ), **h** ( _t_ –1)). Its output at time step _t_ , denoted **ŷ** ( _t_ ), is also a function of the previous state and the current inputs. In the case of the basic cells we have discussed so far, the output is just equal to the state, but in more complex cells this is not always the case, as shown in Figure 15-3. 

###### **540 | Chapter 15: Processing Sequences Using RNNs and CNNs** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 



<!-- Start of picture text -->
y Yo) Vy V2)<br>h -.<br>hoy ha<br>x Xo) Xa) X(2)<br><!-- End of picture text -->



<!-- Start of picture text -->
srr rrr rr rr rrr ’<br>Yo YY YY Yy Ye nr)<br>Xo) Xa Xa) Xy Xo  Xy XX<br>; ; ; (“Encoder "Decoder 3<br>Yo YY Yo YQ u Yq Yo Yea:<br>X X X X | Xo Xt ;<br><!-- End of picture text -->



<!-- Start of picture text -->
LV). Yay Yay Yay Yay)<br>WY 4 [a<br>4 '<br>‘2° Ge i! ;<br>7" _ - eo - a ~ |<br>Yo) Y) Yo Yea Ya<br>A A A rn rn<br>|(||'<br>Xo)‘ Xyt X()| Xx)| Xa‘<br><!-- End of picture text -->



<!-- Start of picture text -->
800000700000 fe= A . HAS Ae cH .<br>sooo fF LU LPL Wt APA T EP UP<br>sooooo 4 ra HE L E L<br>400000 \ Ve<br>ooo lt Nf Y Vy yY ¥ yi y y Ys<br>200000 A+————___|ee es Sn<br>2019 Apr May<br>date<br><!-- End of picture text -->

Note that Pandas includes both the start and end month in the range, so this plots the data from the 1st of March all the way up to the 31st of May. This is a _time series_ : data with values at different time steps, usually at regular intervals. More specifically, since there are multiple values per time step, this is called a _multivariate time series_ . If we only looked at the `bus` column, it would be a _univariate time series_ , with a single value per time step. Predicting future values (i.e., forecasting) is the most typical task when dealing with time series, and this is what we will focus on in this chapter. Other tasks include imputation (filling in missing past values), classification, anomaly detection, and more. 

Looking at Figure 15-6, we can see that a similar pattern is clearly repeated every week. This is called a weekly _seasonality_ . In fact, it’s so strong in this case that forecasting tomorrow’s ridership by just copying the values from a week earlier will yield reasonably good results. This is called _naive forecasting_ : simply copying a past value to make our forecast. Naive forecasting is often a great baseline, and it can even be tricky to beat in some cases. 



In general, naive forecasting means copying the latest known value (e.g., forecasting that tomorrow will be the same as today). How‐ ever, in our case, copying the value from the previous week works better, due to the strong weekly seasonality. 

To visualize these naive forecasts, let’s overlay the two time series (for bus and rail) as well as the same time series lagged by one week (i.e., shifted toward the right) using dotted lines. We’ll also plot the difference between the two (i.e., the value at time _t_ minus the value at time _t_ – 7); this is called _differencing_ (see Figure 15-7): 

```
diff_7=df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]
```

```
fig, axs=plt.subplots(2, 1, sharex=True, figsize=(8, 5))
df.plot(ax=axs[0], legend=False, marker=".")  # original time series
df.shift(7).plot(ax=axs[0], grid=True, legend=False, linestyle=":")  # lagged
diff_7.plot(ax=axs[1], grid=True, marker=".")  # 7-day difference time series
plt.show()
```

Not too bad! Notice how closely the lagged time series track the actual time series. When a time series is correlated with a lagged version of itself, we say that the time series is _autocorrelated_ . As you can see, most of the differences are fairly small, except at the end of May. Maybe there was a holiday at that time? Let’s check the `day_type` column: 

```
>>> list(df.loc["2019-05-25":"2019-05-27"]["day_type"])
['A', 'U', 'U']
```

**Forecasting a Time Series | 545** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 



<!-- Start of picture text -->
800000alalalmlalalmialclalsialmla- = AA. ‘ MAAR MRAy  RP°<br>WO WU AAPAAW<br>so0000 |__ J<br>—200000 tome\J ~ Ix =, anne. manneTa VA .<br>— bus ee eee<br>Mar Apr May<br>2019<br>date<br><!-- End of picture text -->





<!-- Start of picture text -->
10 a<br>' -k-tasksal ihe<br>ind 0h0A<br><!-- End of picture text -->



<!-- Start of picture text -->
50000 he f | ' ! ao il<br>‘wee ae eee<br>50000 | I no! “WY hh pt 5 ha LS la<br>Ra vege<br>aeons<br>2003 2005 2007 2009 2011 2013 2015 2017 2019<br>2001 eee<br>date<br><!-- End of picture text -->

### **The ARMA Model Family** 

We’ll start with the _autoregressive moving average_ (ARMA) model, developed by Herman Wold in the 1930s: it computes its forecasts using a simple weighted sum of lagged values and corrects these forecasts by adding a moving average, very much like we just discussed. Specifically, the moving average component is computed using a weighted sum of the last few forecast errors. Equation 15-3 shows how the model makes its forecasts. 

_Equation 15-3. Forecasting using an ARMA model_ 



In this equation: 

- _ŷ_ ( _t_ ) is the model’s forecast for time step _t_ . 

- _y_ ( _t_ ) is the time series’ value at time step _t_ . 

- The first sum is the weighted sum of the past _p_ values of the time series, using the learned weights _αi_ . The number _p_ is a hyperparameter, and it determines how far back into the past the model should look. This sum is the _autoregressive_ component of the model: it performs regression based on past values. 

- The second sum is the weighted sum over the past _q_ forecast errors _ε_ ( _t_ ), using the learned weights _θi_ . The number _q_ is a hyperparameter. This sum is the moving average component of the model. 

Importantly, this model assumes that the time series is stationary. If it is not, then differencing may help. Using differencing over a single time step will produce an approximation of the derivative of the time series: indeed, it will give the slope of the series at each time step. This means that it will eliminate any linear trend, transforming it into a constant value. For example, if you apply one-step differencing to the series [3, 5, 7, 9, 11], you get the differenced series [2, 2, 2, 2]. 

If the original time series has a quadratic trend instead of a linear trend, then a single round of differencing will not be enough. For example, the series [1, 4, 9, 16, 25, 36] becomes [3, 5, 7, 9, 11] after one round of differencing, but if you run differencing for a second round, then you get [2, 2, 2, 2]. So, running two rounds of differencing will eliminate quadratic trends. More generally, running _d_ consecutive rounds of differencing computes an approximation of the _d_<sup>th</sup> order derivative of the time series, so it will eliminate polynomial trends up to degree _d_ . This hyperparameter _d_ is called the _order of integration_ . 

**Forecasting a Time Series | 549** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

Differencing is the central contribution of the _autoregressive integrated moving average_ (ARIMA) model, introduced in 1970 by George Box and Gwilym Jenkins in their book _Time Series Analysis_ (Wiley): this model runs _d_ rounds of differencing to make the time series more stationary, then it applies a regular ARMA model. When making forecasts, it uses this ARMA model, then it adds back the terms that were subtracted by differencing. 

One last member of the ARMA family is the _seasonal ARIMA_ (SARIMA) model: it models the time series in the same way as ARIMA, but it additionally models a seasonal component for a given frequency (e.g., weekly), using the exact same ARIMA approach. It has a total of seven hyperparameters: the same _p_ , _d_ , and _q_ hyperparameters as ARIMA, plus additional _P_ , _D_ , and _Q_ hyperparameters to model the seasonal pattern, and lastly the period of the seasonal pattern, noted _s_ . The hyperparameters _P_ , _D_ , and _Q_ are just like _p_ , _d_ , and _q_ , but they are used to model the time series at _t_ – _s_ , _t_ – 2 _s_ , _t_ – 3 _s_ , etc. 

Let’s see how to fit a SARIMA model to the rail time series, and use it to make a forecast for tomorrow’s ridership. We’ll pretend today is the last day of May 2019, and we want to forecast the rail ridership for “tomorrow”, the 1st of June, 2019. For this, we can use the `statsmodels` library, which contains many different statistical models, including the ARMA model and its variants, implemented by the `ARIMA` class: 

```
fromstatsmodels.tsa.arima.modelimportARIMA
```

```
origin, today="2019-01-01", "2019-05-31"
rail_series=df.loc[origin:today]["rail"].asfreq("D")
model=ARIMA(rail_series,
order=(1, 0, 0),
seasonal_order=(0, 1, 1, 7))
model=model.fit()
y_pred=model.forecast()  # returns 427,758.6
```

In this code example: 

- We start by importing the `ARIMA` class, then we take the rail ridership data from the start of 2019 up to “today”, and we use `asfreq("D")` to set the time series’ frequency to daily: this doesn’t change the data at all in this case, since it’s already daily, but without this the `ARIMA` class would have to guess the frequency, and it would display a warning. 

- Next, we create an `ARIMA` instance, passing it all the data until “today”, and we set the model hyperparameters: `order=(1, 0, 0)` means that _p_ = 1, _d_ = 0, _q_ = 0, and `seasonal_order=(0, 1, 1, 7)` means that _P_ = 0, _D_ = 1, _Q_ = 1, and _s_ = 7. Notice that the `statsmodels` API differs a bit from Scikit-Learn’s API, since we pass the data to the model at construction time, instead of passing it to the `fit()` method. 

###### **550 | Chapter 15: Processing Sequences Using RNNs and CNNs** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

- Next, we fit the model, and we use it to make a forecast for “tomorrow”, the 1st of June, 2019. 

The forecast is 427,759 passengers, when in fact there were 379,044. Yikes, we’re 12.9% off—that’s pretty bad. It’s actually slightly worse than naive forecasting, which forecasts 426,932, off by 12.6%. But perhaps we were just unlucky that day? To check this, we can run the same code in a loop to make forecasts for every day in March, April, and May, and compute the MAE over that period: 

```
origin, start_date, end_date="2019-01-01", "2019-03-01", "2019-05-31"
time_period=pd.date_range(start_date, end_date)
rail_series=df.loc[origin:end_date]["rail"].asfreq("D")
y_preds= []
fortodayintime_period.shift(-1):
model=ARIMA(rail_series[origin:today],  # train on data up to "today"
order=(1, 0, 0),
seasonal_order=(0, 1, 1, 7))
model=model.fit()  # note that we retrain the model every day!
y_pred=model.forecast()[0]
y_preds.append(y_pred)
```

```
y_preds=pd.Series(y_preds, index=time_period)
mae= (y_preds-rail_series[time_period]).abs().mean()  # returns 32,040.7
```

Ah, that’s much better! The MAE is about 32,041, which is significantly lower than the MAE we got with naive forecasting (42,143). So although the model is not perfect, it still beats naive forecasting by a large margin, on average. 

At this point, you may be wondering how to pick good hyperparameters for the SARIMA model. There are several methods, but the simplest to understand and to get started with is the brute-force approach: just run a grid search. For each model you want to evaluate (i.e., each hyperparameter combination), you can run the preceding code example, changing only the hyperparameter values. Good _p_ , _q_ , _P_ , and _Q_ values are usually fairly small (typically 0 to 2, sometimes up to 5 or 6), and _d_ and _D_ are typically 0 or 1, sometimes 2. As for _s_ , it’s just the main seasonal pattern’s period: in our case it’s 7 since there’s a strong weekly seasonality. The model with the lowest MAE wins. Of course, you can replace the MAE with another metric if it better matches your business objective. And that’s it!<sup>4</sup> 

> 4 There are other more principled approaches to selecting good hyperparameters, based on analyzing the _autocorrelation function_ (ACF) and _partial autocorrelation function_ (PACF), or minimizing the AIC or BIC metrics (introduced in Chapter 9) to penalize models that use too many parameters and reduce the risk of overfitting the data, but grid search is a good place to start. For more details on the ACF-PACF approach, check out this very nice post by Jason Brownlee. 

**Forecasting a Time Series | 551** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

### **Preparing the Data for Machine Learning Models** 

Now that we have two baselines, naive forecasting and SARIMA, let’s try to use the machine learning models we’ve covered so far to forecast this time series, starting with a basic linear model. Our goal will be to forecast tomorrow’s ridership based on the ridership of the past 8 weeks of data (56 days). The inputs to our model will therefore be sequences (usually a single sequence per day once the model is in production), each containing 56 values from time steps _t_ – 55 to _t_ . For each input sequence, the model will output a single value: the forecast for time step _t_ + 1. 

But what will we use as training data? Well, that’s the trick: we will use every 56-day window from the past as training data, and the target for each window will be the value immediately following it. 

Keras actually has a nice utility function called `tf.keras.utils.timeseries_ dataset_from_array()` to help us prepare the training set. It takes a time series as input, and it builds a tf.data.Dataset (introduced in Chapter 13) containing all the windows of the desired length, as well as their corresponding targets. Here’s an example that takes a time series containing the numbers 0 to 5 and creates a dataset containing all the windows of length 3, with their corresponding targets, grouped into batches of size 2: 

```
importtensorflowastf
my_series= [0, 1, 2, 3, 4, 5]
my_dataset=tf.keras.utils.timeseries_dataset_from_array(
my_series,
targets=my_series[3:],  # the targets are 3 steps into the future
sequence_length=3,
batch_size=2
)
```

Let’s inspect the contents of this dataset: 

```
>>> list(my_dataset)
[(<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
  array([[0, 1, 2],
         [1, 2, 3]], dtype=int32)>,
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 4], dtype=int32)>),
 (<tf.Tensor: shape=(1, 3), dtype=int32, numpy=array([[2, 3, 4]], dtype=int32)>,
  <tf.Tensor: shape=(1,), dtype=int32, numpy=array([5], dtype=int32)>)]
```

Each sample in the dataset is a window of length 3, along with its corresponding target (i.e., the value immediately after the window). The windows are [0, 1, 2], [1, 2, 3], and [2, 3, 4], and their respective targets are 3, 4, and 5. Since there are three windows in total, which is not a multiple of the batch size, the last batch only contains one window instead of two. 

###### **552 | Chapter 15: Processing Sequences Using RNNs and CNNs** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

Another way to get the same result is to use the `window()` method of tf.data’s `Dataset` class. It’s more complex, but it gives you full control, which will come in handy later in this chapter, so let’s see how it works. The `window()` method returns a dataset of window datasets: 

```
>>> forwindow_datasetintf.data.Dataset.range(6).window(4, shift=1):
... forelementinwindow_dataset:
... print(f"{element}", end=" ")
... print()
...
0 1 2 3
1 2 3 4
2 3 4 5
3 4 5
4 5
5
```

In this example, the dataset contains six windows, each shifted by one step compared to the previous one, and the last three windows are smaller because they’ve reached the end of the series. In general you’ll want to get rid of these smaller windows by passing `drop_remainder=True` to the `window()` method. 

The `window()` method returns a _nested dataset_ , analogous to a list of lists. This is useful when you want to transform each window by calling its dataset methods (e.g., to shuffle them or batch them). However, we cannot use a nested dataset directly for training, as our model will expect tensors as input, not datasets. 

Therefore, we must call the `flat_map()` method: it converts a nested dataset into a _flat dataset_ (one that contains tensors, not datasets). For example, suppose {1, 2, 3} represents a dataset containing the sequence of tensors 1, 2, and 3. If you flatten the nested dataset {{1, 2}, {3, 4, 5, 6}}, you get back the flat dataset {1, 2, 3, 4, 5, 6}. 

Moreover, the `flat_map()` method takes a function as an argument, which allows you to transform each dataset in the nested dataset before flattening. For example, if you pass the function `lambda ds: ds.batch(2)` to `flat_map()` , then it will transform the nested dataset {{1, 2}, {3, 4, 5, 6}} into the flat dataset {[1, 2], [3, 4], [5, 6]}: it’s a dataset containing 3 tensors, each of size 2. 

With that in mind, we are ready to flatten our dataset: 

```
>>> dataset=tf.data.Dataset.range(6).window(4, shift=1, drop_remainder=True)
>>> dataset=dataset.flat_map(lambdawindow_dataset: window_dataset.batch(4))
>>> forwindow_tensorindataset:
""
... print(f{window_tensor})
...
[0 1 2 3]
[1 2 3 4]
[2 3 4 5]
```

**Forecasting a Time Series | 553** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

Since each window dataset contains exactly four items, calling `batch(4)` on a window produces a single tensor of size 4. Great! We now have a dataset containing consecu‐ tive windows represented as tensors. Let’s create a little helper function to make it easier to extract windows from a dataset: 

```
defto_windows(dataset, length):
dataset=dataset.window(length, shift=1, drop_remainder=True)
returndataset.flat_map(lambdawindow_ds: window_ds.batch(length))
```

The last step is to split each window into inputs and targets, using the `map()` method. We can also group the resulting windows into batches of size 2: 

**`>>>`** `dataset = to_windows(tf.data.Dataset.range(6), 4)` _`# 3 inputs + 1 target = 4`_ **`>>>`** `dataset = dataset.map(` **`lambda`** `window: (window[:-1], window[-1]))` **`>>>`** `list(dataset.batch(2)) [(<tf.Tensor: shape=(2, 3), dtype=int64, numpy= array([[0, 1, 2], [1, 2, 3]])>, <tf.Tensor: shape=(2,), dtype=int64, numpy=array([3, 4])>), (<tf.Tensor: shape=(1, 3), dtype=int64, numpy=array([[2, 3, 4]])>, <tf.Tensor: shape=(1,), dtype=int64, numpy=array([5])>)]` As you can see, we now have the same output as we got earlier with the `timeseries_dataset_from_array()` function (with a bit more effort, but it will be worthwhile soon). 

Now, before we start training, we need to split our data into a training period, a validation period, and a test period. We will focus on the rail ridership for now. We will also scale it down by a factor of one million, to ensure the values are near the 0–1 range; this plays nicely with the default weight initialization and learning rate: 

```
rail_train=df["rail"]["2016-01":"2018-12"] /1e6
rail_valid=df["rail"]["2019-01":"2019-05"] /1e6
rail_test=df["rail"]["2019-06":] /1e6
```



When dealing with time series, you generally want to split across time. However, in some cases you may be able to split along other dimensions, which will give you a longer time period to train on. For example, if you have data about the financial health of 10,000 companies from 2001 to 2019, you might be able to split this data across the different companies. It’s very likely that many of these companies will be strongly correlated, though (e.g., whole economic sectors may go up or down jointly), and if you have correlated companies across the training set and the test set, your test set will not be as useful, as its measure of the generalization error will be optimistically biased. 

###### **554 | Chapter 15: Processing Sequences Using RNNs and CNNs** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

Next, let’s use `timeseries_dataset_from_array()` to create datasets for training and validation. Since gradient descent expects the instances in the training set to be independent and identically distributed (IID), as we saw in Chapter 4, we must set the argument `shuffle=True` to shuffle the training windows (but not their contents): 

```
seq_length=56
train_ds=tf.keras.utils.timeseries_dataset_from_array(
rail_train.to_numpy(),
targets=rail_train[seq_length:],
sequence_length=seq_length,
batch_size=32,
shuffle=True,
seed=42
)
valid_ds=tf.keras.utils.timeseries_dataset_from_array(
rail_valid.to_numpy(),
targets=rail_valid[seq_length:],
sequence_length=seq_length,
batch_size=32
)
```

And now we’re ready to build and train any regression model we want! 

### **Forecasting Using a Linear Model** 

Let’s try a basic linear model first. We will use the Huber loss, which usually works better than minimizing the MAE directly, as discussed in Chapter 10. We’ll also use early stopping: 

```
tf.random.set_seed(42)
model=tf.keras.Sequential([
tf.keras.layers.Dense(1, input_shape=[seq_length])
])
early_stopping_cb=tf.keras.callbacks.EarlyStopping(
monitor="val_mae", patience=50, restore_best_weights=True)
opt=tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
history=model.fit(train_ds, validation_data=valid_ds, epochs=500,
callbacks=[early_stopping_cb])
```

This model reaches a validation MAE of about 37,866 (your mileage may vary). That’s better than naive forecasting, but worse than the SARIMA model.<sup>5</sup> 

Can we do better with an RNN? Let’s see! 

> 5 Note that the validation period starts on the 1st of January 2019, so the first prediction is for the 26th of February 2019, eight weeks later. When we evaluated the baseline models we used predictions starting on the 1st of March instead, but this should be close enough. 

**Forecasting a Time Series | 555** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

### **Forecasting Using a Simple RNN** 

Let’s try the most basic RNN, containing a single recurrent layer with just one recurrent neuron, as we saw in Figure 15-1: 

```
model=tf.keras.Sequential([
tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])
```

```
])
```

All recurrent layers in Keras expect 3D inputs of shape [ _batch size_ , _time steps_ , _dimen‐ sionality_ ], where _dimensionality_ is 1 for univariate time series and more for multivari‐ ate time series. Recall that the `input_shape` argument ignores the first dimension (i.e., the batch size), and since recurrent layers can accept input sequences of any length, we can set the second dimension to `None` , which means “any size”. Lastly, since we’re dealing with a univariate time series, we need the last dimension’s size to be 1. This is why we specified the input shape `[None, 1]` : it means “univariate sequences of any length”. Note that the datasets actually contain inputs of shape [ _batch size_ , _time steps_ ], so we’re missing the last dimension, of size 1, but Keras is kind enough to add it for us in this case. 

This model works exactly as we saw earlier: the initial state _h_ (init) is set to 0, and it is passed to a single recurrent neuron, along with the value of the first time step, _x_ (0). The neuron computes a weighted sum of these values plus the bias term, and it applies the activation function to the result, using the hyperbolic tangent function by default. The result is the first output, _y_ 0. In a simple RNN, this output is also the new state _h_ 0. This new state is passed to the same recurrent neuron along with the next input value, _x_ (1), and the process is repeated until the last time step. At the end, the layer just outputs the last value: in our case the sequences are 56 steps long, so the last value is _y_ 55. All of this is performed simultaneously for every sequence in the batch, of which there are 32 in this case. 



By default, recurrent layers in Keras only return the final output. To make them return one output per time step, you must set `return_sequences=True` , as you will see. 

So that’s our first recurrent model! It’s a sequence-to-vector model. Since there’s a single output neuron, the output vector has a size of 1. 

Now if you compile, train, and evaluate this model just like the previous model, you will find that it’s no good at all: its validation MAE is greater than 100,000! Ouch. That was to be expected, for two reasons: 

###### **556 | Chapter 15: Processing Sequences Using RNNs and CNNs** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

**1.** The model only has a single recurrent neuron, so the only data it can use to make a prediction at each time step is the input value at the current time step and the output value from the previous time step. That’s not much to go on! In other words, the RNN’s memory is extremely limited: it’s just a single number, its previous output. And let’s count how many parameters this model has: since there’s just one recurrent neuron with only two input values, the whole model only has three parameters (two weights plus a bias term). That’s far from enough for this time series. In contrast, our previous model could look at all 56 previous values at once, and it had a total of 57 parameters. 

**2.** The time series contains values from 0 to about 1.4, but since the default activa‐ tion function is tanh, the recurrent layer can only output values between –1 and +1. There’s no way it can predict values between 1.0 and 1.4. 

Let’s fix both of these issues: we will create a model with a larger recurrent layer, containing 32 recurrent neurons, and we will add a dense output layer on top of it with a single output neuron and no activation function. The recurrent layer will be able to carry much more information from one time step to the next, and the dense output layer will project the final output from 32 dimensions down to 1, without any value range constraints: 

```
univar_model=tf.keras.Sequential([
tf.keras.layers.SimpleRNN(32, input_shape=[None, 1]),
```

```
tf.keras.layers.Dense(1)  # no activation function by default
```

- `])` 

Now if you compile, fit, and evaluate this model just like the previous one, you will find that its validation MAE reaches 27,703. That’s the best model we’ve trained so far, and it even beats the SARIMA model: we’re doing pretty well! 



We’ve only normalized the time series, without removing trend and seasonality, and yet the model still performs well. This is con‐ venient, as it makes it possible to quickly search for promising models without worrying too much about preprocessing. However, to get the best performance, you may want to try making the time series more stationary; for example, using differencing. 

### **Forecasting Using a Deep RNN** 

It is quite common to stack multiple layers of cells, as shown in Figure 15-10. This gives you a _deep RNN_ . 

**Forecasting a Time Series | 557** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 



<!-- Start of picture text -->
aalreaw Casale<br>SELLE<br>PLE! tA<br>CC} tH<br><!-- End of picture text -->



### **Forecasting Multivariate Time Series** 

A great quality of neural networks is their flexibility: in particular, they can deal with multivariate time series with almost no change to their architecture. For example, let’s try to forecast the rail time series using both the bus and rail data as input. In fact, let’s also throw in the day type! Since we can always know in advance whether tomorrow is going to be a weekday, a weekend, or a holiday, we can shift the day type series one day into the future, so that the model is given tomorrow’s day type as input. For simplicity, we’ll do this processing using Pandas: 

```
df_mulvar=df[["bus", "rail"]] /1e6# use both bus & rail series as input
df_mulvar["next_day_type"] =df["day_type"].shift(-1)  # we know tomorrow's type
df_mulvar=pd.get_dummies(df_mulvar)  # one-hot encode the day type
```

Now `df_mulvar` is a DataFrame with five columns: the bus and rail data, plus three columns containing the one-hot encoding of the next day’s type (recall that there are three possible day types, `W` , `A` , and `U` ). Next we can proceed much like we did earlier. First we split the data into three periods, for training, validation, and testing: 

```
mulvar_train=df_mulvar["2016-01":"2018-12"]
mulvar_valid=df_mulvar["2019-01":"2019-05"]
mulvar_test=df_mulvar["2019-06":]
```

Then we create the datasets: 

```
train_mulvar_ds=tf.keras.utils.timeseries_dataset_from_array(
mulvar_train.to_numpy(),  # use all 5 columns as input
targets=mulvar_train["rail"][seq_length:],  # forecast only the rail series
    [...]  # the other 4 arguments are the same as earlier
)
valid_mulvar_ds=tf.keras.utils.timeseries_dataset_from_array(
mulvar_valid.to_numpy(),
targets=mulvar_valid["rail"][seq_length:],
    [...]  # the other 2 arguments are the same as earlier
)
```

And finally we create the RNN: 

```
mulvar_model=tf.keras.Sequential([
tf.keras.layers.SimpleRNN(32, input_shape=[None, 5]),
tf.keras.layers.Dense(1)
])
```

Notice that the only difference from the `univar_model` RNN we built earlier is the input shape: at each time step, the model now receives five inputs instead of one. This model actually reaches a validation MAE of 22,062. Now we’re making big progress! 

In fact, it’s not too hard to make the RNN forecast both the bus and rail rid‐ ership. You just need to change the targets when creating the datasets, setting them to `mulvar_train[["bus", "rail"]][seq_length:]` for the training set, and `mulvar_valid[["bus", "rail"]][seq_length:]` for the validation set. You must 

**Forecasting a Time Series | 559** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

also add an extra neuron in the output `Dense` layer, since it must now make two forecasts: one for tomorrow’s bus ridership, and the other for rail. That’s all there is to it! 

As we discussed in Chapter 10, using a single model for multiple related tasks often results in better performance than using a separate model for each task, since features learned for one task may be useful for the other tasks, and also because having to perform well across multiple tasks prevents the model from overfitting (it’s a form of regularization). However, it depends on the task, and in this particular case the multitask RNN that forecasts both the bus and the rail ridership doesn’t perform quite as well as dedicated models that forecast one or the other (using all five columns as input). Still, it reaches a validation MAE of 25,330 for rail and 26,369 for bus, which is pretty good. 

### **Forecasting Several Time Steps Ahead** 

So far we have only predicted the value at the next time step, but we could just as easily have predicted the value several steps ahead by changing the targets appropri‐ ately (e.g., to predict the ridership 2 weeks from now, we could just change the targets to be the value 14 days ahead instead of 1 day ahead). But what if we want to predict the next 14 values? 

The first option is to take the `univar_model` RNN we trained earlier for the rail time series, make it predict the next value, and add that value to the inputs, acting as if the predicted value had actually occurred; we would then use the model again to predict the following value, and so on, as in the following code: 

```
importnumpyasnp
```

```
X=rail_valid.to_numpy()[np.newaxis, :seq_length, np.newaxis]
forstep_aheadinrange(14):
```

```
y_pred_one=univar_model.predict(X)
```

```
X=np.concatenate([X, y_pred_one.reshape(1, 1, 1)], axis=1)
```

In this code, we take the rail ridership of the first 56 days of the validation period, and we convert the data to a NumPy array of shape [1, 56, 1] (recall that recurrent layers expect 3D inputs). Then we repeatedly use the model to forecast the next value, and we append each forecast to the input series, along the time axis ( `axis=1` ). The resulting forecasts are plotted in Figure 15-11. 



If the model makes an error at one time step, then the forecasts for the following time steps are impacted as well: the errors tend to accumulate. So, it’s preferable to use this technique only for a small number of steps. 

###### **560 | Chapter 15: Processing Sequences Using RNNs and CNNs** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 



<!-- Start of picture text -->
800000<br>700000 f \ I1 ey lr<br>600000 + —.— Trye 1 |<br>! ! \<br>500000 +|<br>4oo000 +.—<—~~~~ TodayPredictions | \K<br>I |<br>300000 \ \ \II\iN<br>200000<br>04 11 18 25 04 11<br>Feb Mar<br>2019<br>date<br><!-- End of picture text -->

```
ahead_model=tf.keras.Sequential([
tf.keras.layers.SimpleRNN(32, input_shape=[None, 5]),
tf.keras.layers.Dense(14)
```

```
])
```

After training this model, you can predict the next 14 values at once like this: 

```
X=mulvar_valid.to_numpy()[np.newaxis, :seq_length]  # shape [1, 56, 5]
Y_pred=ahead_model.predict(X)  # shape [1, 14]
```

This approach works quite well. Its forecasts for the next day are obviously better than its forecasts for 14 days into the future, but it doesn’t accumulate errors like the previous approach did. However, we can still do better, using a sequence-to-sequence (or _seq2seq_ ) model. 

### **Forecasting Using a Sequence-to-Sequence Model** 

Instead of training the model to forecast the next 14 values only at the very last time step, we can train it to forecast the next 14 values at each and every time step. In other words, we can turn this sequence-to-vector RNN into a sequence-to-sequence RNN. The advantage of this technique is that the loss will contain a term for the output of the RNN at each and every time step, not just for the output at the last time step. 

This means there will be many more error gradients flowing through the model, and they won’t have to flow through time as much since they will come from the output of each time step, not just the last one. This will both stabilize and speed up training. 

To be clear, at time step 0 the model will output a vector containing the forecasts for time steps 1 to 14, then at time step 1 the model will forecast time steps 2 to 15, and so on. In other words, the targets are sequences of consecutive windows, shifted by one time step at each time step. The target is not a vector anymore, but a sequence of the same length as the inputs, containing a 14-dimensional vector at each step. 

Preparing the datasets is not trivial, since each instance has a window as input and a sequence of windows as output. One way to do this is to use the `to_windows()` utility function we created earlier, twice in a row, to get windows of consecutive windows. For example, let’s turn the series of numbers 0 to 6 into a dataset containing sequences of 4 consecutive windows, each of length 3: 

```
>>> my_series=tf.data.Dataset.range(7)
>>> dataset=to_windows(to_windows(my_series, 3), 4)
>>> list(dataset)
[<tf.Tensor: shape=(4, 3), dtype=int64, numpy=
 array([[0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]])>,
 <tf.Tensor: shape=(4, 3), dtype=int64, numpy=
 array([[1, 2, 3],
        [2, 3, 4],
```

###### **562 | Chapter 15: Processing Sequences Using RNNs and CNNs** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

```
        [3, 4, 5],
        [4, 5, 6]])>]
```

Now we can use the `map()` method to split these windows of windows into inputs and targets: 

```
>>> dataset=dataset.map(lambdaS: (S[:, 0], S[:, 1:]))
>>> list(dataset)
[(<tf.Tensor: shape=(4,), dtype=int64, numpy=array([0, 1, 2, 3])>,
  <tf.Tensor: shape=(4, 2), dtype=int64, numpy=
  array([[1, 2],
         [2, 3],
         [3, 4],
         [4, 5]])>),
 (<tf.Tensor: shape=(4,), dtype=int64, numpy=array([1, 2, 3, 4])>,
  <tf.Tensor: shape=(4, 2), dtype=int64, numpy=
  array([[2, 3],
         [3, 4],
         [4, 5],
         [5, 6]])>)]
```

Now the dataset contains sequences of length 4 as inputs, and the targets are sequen‐ ces containing the next two steps, for each time step. For example, the first input sequence is [0, 1, 2, 3], and its corresponding targets are [[1, 2], [2, 3], [3, 4], [4, 5]], which are the next two values for each time step. If you’re like me, you will probably need a few minutes to wrap your head around this. Take your time! 



It may be surprising that the targets contain values that appear in the inputs. Isn’t that cheating? Fortunately, not at all: at each time step, an RNN only knows about past time steps; it cannot look ahead. It is said to be a _causal_ model. 

Let’s create another little utility function to prepare the datasets for our sequence-tosequence model. It will also take care of shuffling (optional) and batching: 

```
defto_seq2seq_dataset(series, seq_length=56, ahead=14, target_col=1,
batch_size=32, shuffle=False, seed=None):
ds=to_windows(tf.data.Dataset.from_tensor_slices(series), ahead+1)
ds=to_windows(ds, seq_length).map(
lambdaS: (S[:, 0], S[:, 1:, target_col]))
ifshuffle:
ds=ds.shuffle(8*batch_size, seed=seed)
returnds.batch(batch_size)
```

##### Now we can use this function to create the datasets: 

```
seq2seq_train=to_seq2seq_dataset(mulvar_train, shuffle=True, seed=42)
seq2seq_valid=to_seq2seq_dataset(mulvar_valid)
```

**Forecasting a Time Series | 563** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 



<!-- Start of picture text -->
™<br><!-- End of picture text -->

™ ™ 



<!-- Start of picture text -->
™<br><!-- End of picture text -->

## **Handling Long Sequences** 

To train an RNN on long sequences, we must run it over many time steps, making the unrolled RNN a very deep network. Just like any deep neural network it may suffer from the unstable gradients problem, discussed in Chapter 11: it may take forever to train, or training may be unstable. Moreover, when an RNN processes a long sequence, it will gradually forget the first inputs in the sequence. Let’s look at both these problems, starting with the unstable gradients problem. 

### **Fighting the Unstable Gradients Problem** 

Many of the tricks we used in deep nets to alleviate the unstable gradients problem can also be used for RNNs: good parameter initialization, faster optimizers, dropout, and so on. However, nonsaturating activation functions (e.g., ReLU) may not help as much here. In fact, they may actually lead the RNN to be even more unstable during training. Why? Well, suppose gradient descent updates the weights in a way that increases the outputs slightly at the first time step. Because the same weights are used at every time step, the outputs at the second time step may also be slightly increased, and those at the third, and so on until the outputs explode—and a nonsaturating activation function does not prevent that. 

You can reduce this risk by using a smaller learning rate, or you can use a saturating activation function like the hyperbolic tangent (this explains why it’s the default). 

In much the same way, the gradients themselves can explode. If you notice that training is unstable, you may want to monitor the size of the gradients (e.g., using TensorBoard) and perhaps use gradient clipping. 

Moreover, batch normalization cannot be used as efficiently with RNNs as with deep feedforward nets. In fact, you cannot use it between time steps, only between recurrent layers. 

To be more precise, it is technically possible to add a BN layer to a memory cell (as you will see shortly) so that it will be applied at each time step (both on the inputs for that time step and on the hidden state from the previous step). However, the same BN layer will be used at each time step, with the same parameters, regardless of the actual scale and offset of the inputs and hidden state. In practice, this does not yield good results, as was demonstrated by César Laurent et al. in a 2015 paper:<sup>7</sup> the authors found that BN was slightly beneficial only when it was applied to the layer’s inputs, not to the hidden states. In other words, it was slightly better than nothing when applied between recurrent layers (i.e., vertically in Figure 15-10), but not within 

> 7 César Laurent et al., “Batch Normalized Recurrent Neural Networks”, _Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing_ (2016): 2657–2661. 

**Handling Long Sequences | 565** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

recurrent layers (i.e., horizontally). In Keras, you can apply BN between layers simply by adding a `BatchNormalization` layer before each recurrent layer, but it will slow down training, and it may not help much. 

Another form of normalization often works better with RNNs: _layer normalization_ . This idea was introduced by Jimmy Lei Ba et al. in a 2016 paper:<sup>8</sup> it is very similar to batch normalization, but instead of normalizing across the batch dimension, layer normalization normalizes across the features dimension. One advantage is that it can compute the required statistics on the fly, at each time step, independently for each instance. This also means that it behaves the same way during training and testing (as opposed to BN), and it does not need to use exponential moving averages to estimate the feature statistics across all instances in the training set, like BN does. Like BN, layer normalization learns a scale and an offset parameter for each input. In an RNN, it is typically used right after the linear combination of the inputs and the hidden states. 

Let’s use Keras to implement layer normalization within a simple memory cell. To do this, we need to define a custom memory cell, which is just like a regular layer, except its `call()` method takes two arguments: the `inputs` at the current time step and the hidden `states` from the previous time step. 

Note that the `states` argument is a list containing one or more tensors. In the case of a simple RNN cell it contains a single tensor equal to the outputs of the previous time step, but other cells may have multiple state tensors (e.g., an `LSTMCell` has a long-term state and a short-term state, as you will see shortly). A cell must also have a `state_size` attribute and an `output_size` attribute. In a simple RNN, both are simply equal to the number of units. The following code implements a custom memory cell that will behave like a `SimpleRNNCell` , except it will also apply layer normalization at each time step: 

```
classLNSimpleRNNCell(tf.keras.layers.Layer):
def __init__(self, units, activation="tanh", **kwargs):
super().__init__(**kwargs)
self.state_size=units
self.output_size=units
self.simple_rnn_cell=tf.keras.layers.SimpleRNNCell(units,
activation=None)
self.layer_norm=tf.keras.layers.LayerNormalization()
self.activation=tf.keras.activations.get(activation)
defcall(self, inputs, states):
outputs, new_states=self.simple_rnn_cell(inputs, states)
norm_outputs=self.activation(self.layer_norm(outputs))
returnnorm_outputs, [norm_outputs]
```

8 Jimmy Lei Ba et al., “Layer Normalization”, arXiv preprint arXiv:1607.06450 (2016). 

**566 | Chapter 15: Processing Sequences Using RNNs and CNNs** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

Let’s walk through this code: 

- Our `LNSimpleRNNCell` class inherits from the `tf.keras.layers.Layer` class, just like any custom layer. 

- The constructor takes the number of units and the desired activation function and sets the `state_size` and `output_size` attributes, then creates a `SimpleRNN Cell` with no activation function (because we want to perform layer normaliza‐ tion after the linear operation but before the activation function).<sup>9</sup> Then the constructor creates the `LayerNormalization` layer, and finally it fetches the desired activation function. 

- The `call()` method starts by applying the `simpleRNNCell` , which computes a linear combination of the current inputs and the previous hidden states, and it returns the result twice (indeed, in a `SimpleRNNCell` , the outputs are just equal to the hidden states: in other words, `new_states[0]` is equal to `outputs` , so we can safely ignore `new_states` in the rest of the `call()` method). Next, the `call()` method applies layer normalization, followed by the activation function. Finally, it returns the outputs twice: once as the outputs, and once as the new hidden states. To use this custom cell, all we need to do is create a `tf.keras.layers.RNN` layer, passing it a cell instance: 

```
custom_ln_model=tf.keras.Sequential([
tf.keras.layers.RNN(LNSimpleRNNCell(32), return_sequences=True,
input_shape=[None, 5]),
tf.keras.layers.Dense(14)
```

```
])
```

Similarly, you could create a custom cell to apply dropout between each time step. But there’s a simpler way: most recurrent layers and cells provided by Keras have `dropout` and `recurrent_dropout` hyperparameters: the former defines the dropout rate to apply to the inputs, and the latter defines the dropout rate for the hidden states, between time steps. So, there’s no need to create a custom cell to apply dropout at each time step in an RNN. 

With these techniques, you can alleviate the unstable gradients problem and train an RNN much more efficiently. Now let’s look at how to deal with the short-term memory problem. 

> 9 It would have been simpler to inherit from `SimpleRNNCell` instead so that we wouldn’t have to create an internal `SimpleRNNCell` or handle the `state_size` and `output_size` attributes, but the goal here was to show how to create a custom cell from scratch. 

**Handling Long Sequences | 567** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 



When forecasting time series, it is often useful to have some error bars along with your predictions. For this, one approach is to use MC dropout, introduced in Chapter 11: use `recurrent_dropout` during training, then keep dropout active at inference time by calling the model using `model(X, training=True)` . Repeat this several times to get multiple slightly different forecasts, then com‐ pute the mean and standard deviation of these predictions for each time step. 

### **Tackling the Short-Term Memory Problem** 

Due to the transformations that the data goes through when traversing an RNN, some information is lost at each time step. After a while, the RNN’s state contains virtually no trace of the first inputs. This can be a showstopper. Imagine Dory the fish<sup>10</sup> trying to translate a long sentence; by the time she’s finished reading it, she has no clue how it started. To tackle this problem, various types of cells with long-term memory have been introduced. They have proven so successful that the basic cells are not used much anymore. Let’s first look at the most popular of these long-term memory cells: the LSTM cell. 

#### **LSTM cells** 

The _long short-term memory_ (LSTM) cell was proposed in 1997<sup>11</sup> by Sepp Hochreiter and Jürgen Schmidhuber and gradually improved over the years by several research‐ ers, such as Alex Graves, Haşim Sak,<sup>12</sup> and Wojciech Zaremba.<sup>13</sup> If you consider the LSTM cell as a black box, it can be used very much like a basic cell, except it will perform much better; training will converge faster, and it will detect longer-term patterns in the data. In Keras, you can simply use the `LSTM` layer instead of the `SimpleRNN` layer: 

```
model=tf.keras.Sequential([
```

```
tf.keras.layers.LSTM(32, return_sequences=True, input_shape=[None, 5]),
tf.keras.layers.Dense(14)
```

```
])
```

Alternatively, you could use the general-purpose `tf.keras.layers.RNN` layer, giv‐ ing it an `LSTMCell` as an argument. However, the `LSTM` layer uses an optimized implementation when running on a GPU (see Chapter 19), so in general it is prefera‐ 

> 10 A character from the animated movies _Finding Nemo_ and _Finding Dory_ who has short-term memory loss. 

> 11 Sepp Hochreiter and Jürgen Schmidhuber, “Long Short-Term Memory”, _Neural Computation_ 9, no. 8 (1997): 1735–1780. 

> 12 Haşim Sak et al., “Long Short-Term Memory Based Recurrent Neural Network Architectures for Large Vocabulary Speech Recognition”, arXiv preprint arXiv:1402.1128 (2014). 

> 13 Wojciech Zaremba et al., “Recurrent Neural Network Regularization”, arXiv preprint arXiv:1409.2329 (2014). 

###### **568 | Chapter 15: Processing Sequences Using RNNs and CNNs** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 



<!-- Start of picture text -->
Vn)<br>et<br>0 & ‘<br>(x) (X)=F> hy<br>J)<br>Fully<br>Fetes fea<br>Zz : Element-wise +<br>h¢-1) — —_ A LSTM cell}:1Additionmultiplication ‘;<br>| eee logistic |<br>X(y See<br><!-- End of picture text -->

- The main layer is the one that outputs **g** ( _t_ ). It has the usual role of analyzing the current inputs **x** ( _t_ ) and the previous (short-term) state **h** ( _t_ –1). In a basic cell, there is nothing other than this layer, and its output goes straight out to **y** ( _t_ ) and **h** ( _t_ ). But in an LSTM cell, this layer’s output does not go straight out; instead its most important parts are stored in the long-term state (and the rest is dropped). 

- The three other layers are _gate controllers_ . Since they use the logistic activation function, the outputs range from 0 to 1. As you can see, the gate controllers’ outputs are fed to element-wise multiplication operations: if they output 0s they close the gate, and if they output 1s they open it. Specifically: 

   - The _forget gate_ (controlled by **f** ( _t_ )) controls which parts of the long-term state should be erased. 

   - The _input gate_ (controlled by **i** ( _t_ )) controls which parts of **g** ( _t_ ) should be added to the long-term state. 

   - Finally, the _output gate_ (controlled by **o** ( _t_ )) controls which parts of the longterm state should be read and output at this time step, both to **h** ( _t_ ) and to **y** ( _t_ ). 

In short, an LSTM cell can learn to recognize an important input (that’s the role of the input gate), store it in the long-term state, preserve it for as long as it is needed (that’s the role of the forget gate), and extract it whenever it is needed. This explains why these cells have been amazingly successful at capturing long-term patterns in time series, long texts, audio recordings, and more. 

Equation 15-4 summarizes how to compute the cell’s long-term state, its short-term state, and its output at each time step for a single instance (the equations for a whole mini-batch are very similar). 

_Equation 15-4. LSTM computations_ 





In this equation: 

- **W** _xi_ , **W** _xf_ , **W** _xo_ , and **W** _xg_ are the weight matrices of each of the four layers for their connection to the input vector **x** ( _t_ ). 

###### **570 | Chapter 15: Processing Sequences Using RNNs and CNNs** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 



<!-- Start of picture text -->
Vt)<br>he i nor hyp<br>&)<br>6)<br>tt<br>GRU cell<br>X(t)<br><!-- End of picture text -->

The GRU cell is a simplified version of the LSTM cell, and it seems to perform just as well<sup>15</sup> (which explains its growing popularity). These are the main simplifications: 

- Both state vectors are merged into a single vector **h** ( _t_ ). 

- A single gate controller **z** ( _t_ ) controls both the forget gate and the input gate. If the gate controller outputs a 1, the forget gate is open (= 1) and the input gate is closed (1 – 1 = 0). If it outputs a 0, the opposite happens. In other words, whenever a memory must be stored, the location where it will be stored is erased first. This is actually a frequent variant to the LSTM cell in and of itself. 

- There is no output gate; the full state vector is output at every time step. How‐ ever, there is a new gate controller **r** ( _t_ ) that controls which part of the previous state will be shown to the main layer ( **g** ( _t_ )). 

Equation 15-5 summarizes how to compute the cell’s state at each time step for a single instance. 

_Equation 15-5. GRU computations_ 



Keras provides a `tf.keras.layers.GRU` layer: using it is just a matter of replacing `SimpleRNN` or `LSTM` with `GRU` . It also provides a `tf.keras.layers.GRUCell` , in case you want to create a custom cell based on a GRU cell. 

LSTM and GRU cells are one of the main reasons behind the success of RNNs. Yet while they can tackle much longer sequences than simple RNNs, they still have a fairly limited short-term memory, and they have a hard time learning long-term patterns in sequences of 100 time steps or more, such as audio samples, long time series, or long sentences. One way to solve this is to shorten the input sequences; for example, using 1D convolutional layers. 

> 15 See Klaus Greff et al., “LSTM: A Search Space Odyssey”, _IEEE Transactions on Neural Networks and Learning Systems_ 28, no. 10 (2017): 2222–2232.This paper seems to show that all LSTM variants perform roughly the same. 

###### **572 | Chapter 15: Processing Sequences Using RNNs and CNNs** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

#### **Using 1D convolutional layers to process sequences** 

In Chapter 14, we saw that a 2D convolutional layer works by sliding several fairly small kernels (or filters) across an image, producing multiple 2D feature maps (one per kernel). Similarly, a 1D convolutional layer slides several kernels across a sequence, producing a 1D feature map per kernel. Each kernel will learn to detect a single very short sequential pattern (no longer than the kernel size). If you use 10 kernels, then the layer’s output will be composed of 10 1D sequences (all of the same length), or equivalently you can view this output as a single 10D sequence. This means that you can build a neural network composed of a mix of recurrent layers and 1D convolutional layers (or even 1D pooling layers). If you use a 1D convolutional layer with a stride of 1 and `"same"` padding, then the output sequence will have the same length as the input sequence. But if you use `"valid"` padding or a stride greater than 1, then the output sequence will be shorter than the input sequence, so make sure you adjust the targets accordingly. 

For example, the following model is the same as earlier, except it starts with a 1D convolutional layer that downsamples the input sequence by a factor of 2, using a stride of 2. The kernel size is larger than the stride, so all inputs will be used to compute the layer’s output, and therefore the model can learn to preserve the useful information, dropping only the unimportant details. By shortening the sequences the convolutional layer may help the `GRU` layers detect longer patterns, so we can afford to double the input sequence length to 112 days. Note that we must also crop off the first three time steps in the targets: indeed, the kernel’s size is 4, so the first output of the convolutional layer will be based on the input time steps 0 to 3, and the first forecasts will be for time steps 4 to 17 (instead of time steps 1 to 14). Moreover, we must downsample the targets by a factor of 2, because of the stride: 

```
conv_rnn_model=tf.keras.Sequential([
tf.keras.layers.Conv1D(filters=32, kernel_size=4, strides=2,
activation="relu", input_shape=[None, 5]),
tf.keras.layers.GRU(32, return_sequences=True),
tf.keras.layers.Dense(14)
])
longer_train=to_seq2seq_dataset(mulvar_train, seq_length=112,
shuffle=True, seed=42)
longer_valid=to_seq2seq_dataset(mulvar_valid, seq_length=112)
downsampled_train=longer_train.map(lambdaX, Y: (X, Y[:, 3::2]))
downsampled_valid=longer_valid.map(lambdaX, Y: (X, Y[:, 3::2]))
[...]  # compile and fit the model using the downsampled datasets
```

If you train and evaluate this model, you will find that it outperforms the previous model (by a small margin). In fact, it is actually possible to use only 1D convolutional layers and drop the recurrent layers entirely! 

**Handling Long Sequences | 573** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 



<!-- Start of picture text -->
DDDIPDIDOIIOIIIIIIIOOO iitone<br>SAGEEEEEE SII difation 4<br>SISISSALALALLZE DILL LE<br>MMI MINTSISISNAAMSISIS I I o ane<br>|<br>TILL LILI O ‘ilation<br>Input<br><!-- End of picture text -->

```
forratein (1, 2, 4, 8) *2:
wavenet_model.add(tf.keras.layers.Conv1D(
```

```
filters=32, kernel_size=2, padding="causal", activation="relu",
dilation_rate=rate))
```

```
wavenet_model.add(tf.keras.layers.Conv1D(filters=14, kernel_size=1))
```

This `Sequential` model starts with an explicit input layer—this is simpler than trying to set `input_shape` only on the first layer. Then it continues with a 1D convolutional layer using `"causal"` padding, which is like `"same"` padding except that the zeros are appended only at the start of the input sequence, instead of on both sides. This ensures that the convolutional layer does not peek into the future when making predictions. Then we add similar pairs of layers using growing dilation rates: 1, 2, 4, 8, and again 1, 2, 4, 8. Finally, we add the output layer: a convolutional layer with 14 filters of size 1 and without any activation function. As we saw earlier, such a convolutional layer is equivalent to a `Dense` layer with 14 units. Thanks to the causal padding, every convolutional layer outputs a sequence of the same length as its input sequence, so the targets we use during training can be the full 112-day sequences: no need to crop them or downsample them. 

The models we’ve discussed in this section offer similar performance for the ridership forecasting task, but they may vary significantly depending on the task and the amount of available data. In the WaveNet paper, the authors achieved state-of-the-art performance on various audio tasks (hence the name of the architecture), including text-to-speech tasks, producing incredibly realistic voices across several languages. They also used the model to generate music, one audio sample at a time. This feat is all the more impressive when you realize that a single second of audio can contain tens of thousands of time steps—even LSTMs and GRUs cannot handle such long sequences. 



If you evaluate our best Chicago ridership models on the test period, starting in 2020, you will find that they perform much worse than expected! Why is that? Well, that’s when the Covid-19 pandemic started, which greatly affected public transportation. As mentioned earlier, these models will only work well if the patterns they learned from the past continue in the future. In any case, before deploying a model to production, verify that it works well on recent data. And once it’s in production, make sure to monitor its performance regularly. 

With that, you can now tackle all sorts of time series! In Chapter 16, we will continue to explore RNNs, and we will see how they can tackle various NLP tasks as well. 

**Handling Long Sequences | 575** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

## **Exercises** 

**1.** Can you think of a few applications for a sequence-to-sequence RNN? What about a sequence-to-vector RNN, and a vector-to-sequence RNN? 

**2.** How many dimensions must the inputs of an RNN layer have? What does each dimension represent? What about its outputs? 

**3.** If you want to build a deep sequence-to-sequence RNN, which RNN layers should have `return_sequences=True` ? What about a sequence-to-vector RNN? 

**4.** Suppose you have a daily univariate time series, and you want to forecast the next seven days. Which RNN architecture should you use? 

**5.** What are the main difficulties when training RNNs? How can you handle them? 

**6.** Can you sketch the LSTM cell’s architecture? 

**7.** Why would you want to use 1D convolutional layers in an RNN? 

**8.** Which neural network architecture could you use to classify videos? 

**9.** Train a classification model for the SketchRNN dataset, available in TensorFlow Datasets. 

**10.** Download the Bach chorales dataset and unzip it. It is composed of 382 chorales composed by Johann Sebastian Bach. Each chorale is 100 to 640 time steps long, and each time step contains 4 integers, where each integer corresponds to a note’s index on a piano (except for the value 0, which means that no note is played). Train a model—recurrent, convolutional, or both—that can predict the next time step (four notes), given a sequence of time steps from a chorale. Then use this model to generate Bach-like music, one note at a time: you can do this by giving the model the start of a chorale and asking it to predict the next time step, then appending these time steps to the input sequence and asking the model for the next note, and so on. Also make sure to check out Google’s Coconet model, which was used for a nice Google doodle about Bach. 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

###### **576 | Chapter 15: Processing Sequences Using RNNs and CNNs** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:16. 

