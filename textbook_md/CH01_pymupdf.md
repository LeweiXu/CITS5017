# **CHAPTER 1 The Machine Learning Landscape** 

Not so long ago, if you had picked up your phone and asked it the way home, it would have ignored you—and people would have questioned your sanity. But machine learning is no longer science fiction: billions of people use it every day. And the truth is it has actually been around for decades in some specialized applications, such as optical character recognition (OCR). The first ML application that really became mainstream, improving the lives of hundreds of millions of people, took over the world back in the 1990s: the _spam filter_ . It’s not exactly a self-aware robot, but it does technically qualify as machine learning: it has actually learned so well that you seldom need to flag an email as spam anymore. It was followed by hundreds of ML applications that now quietly power hundreds of products and features that you use regularly: voice prompts, automatic translation, image search, product recommenda‐ tions, and many more. 

Where does machine learning start and where does it end? What exactly does it mean for a machine to _learn_ something? If I download a copy of all Wikipedia articles, has my computer really learned something? Is it suddenly smarter? In this chapter I will start by clarifying what machine learning is and why you may want to use it. 

Then, before we set out to explore the machine learning continent, we will take a look at the map and learn about the main regions and the most notable landmarks: supervised versus unsupervised learning and their variants, online versus batch learn‐ ing, instance-based versus model-based learning. Then we will look at the workflow of a typical ML project, discuss the main challenges you may face, and cover how to evaluate and fine-tune a machine learning system. 

This chapter introduces a lot of fundamental concepts (and jargon) that every data scientist should know by heart. It will be a high-level overview (it’s the only chapter without much code), all rather simple, but my goal is to ensure everything is crystal 

**3** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:16:49. 



<!-- Start of picture text -->
|<br><!-- End of picture text -->

| 



<!-- Start of picture text -->
:<br>1. Study the problem :— |<br>4. Analyze errors<br><!-- End of picture text -->



<!-- Start of picture text -->
fe)<br>Data<br>1. Study the problem 2. Train ML model Ss<br>—?)<br>%"7-7(oe) 1. Train ML model Can be Gout)<br>1°) automated<br>Data<br><!-- End of picture text -->



<!-- Start of picture text -->
1. Study the problem 2. Train ML model Go<br>Solution<br>J<br>I<br>1<br>'<br>BO, goo 3. Inspect the<br>solution<br>) *Lots* of data<br>|<br>I<br>!poaa toaA  a eyfam 4.B4. Better understanding| d \ | /<br>\--------/ /YS<br>| - Repeat if needed of the problem! &<br><!-- End of picture text -->

## **Examples of Applications** 

Let’s look at some concrete examples of machine learning tasks, along with the techniques that can tackle them: 

_Analyzing images of products on a production line to automatically classify them_ This is image classification, typically performed using convolutional neural net‐ works (CNNs; see Chapter 14) or sometimes transformers (see Chapter 16). 

###### _Detecting tumors in brain scans_ 

This is semantic image segmentation, where each pixel in the image is classified (as we want to determine the exact location and shape of tumors), typically using CNNs or transformers. 

###### _Automatically classifying news articles_ 

This is natural language processing (NLP), and more specifically text classifica‐ tion, which can be tackled using recurrent neural networks (RNNs) and CNNs, but transformers work even better (see Chapter 16). 

_Automatically flagging offensive comments on discussion forums_ This is also text classification, using the same NLP tools. 

###### _Summarizing long documents automatically_ 

This is a branch of NLP called text summarization, again using the same tools. 

###### _Creating a chatbot or a personal assistant_ 

This involves many NLP components, including natural language understanding (NLU) and question-answering modules. 

###### _Forecasting your company’s revenue next year, based on many performance metrics_ 

This is a regression task (i.e., predicting values) that may be tackled using any regression model, such as a linear regression or polynomial regression model (see Chapter 4), a regression support vector machine (see Chapter 5), a regres‐ sion random forest (see Chapter 7), or an artificial neural network (see Chap‐ ter 10). If you want to take into account sequences of past performance metrics, you may want to use RNNs, CNNs, or transformers (see Chapters 15 and 16). 

###### _Making your app react to voice commands_ 

This is speech recognition, which requires processing audio samples: since they are long and complex sequences, they are typically processed using RNNs, CNNs, or transformers (see Chapters 15 and 16). 

###### _Detecting credit card fraud_ 

This is anomaly detection, which can be tackled using isolation forests, Gaussian mixture models (see Chapter 9), or autoencoders (see Chapter 17). 

###### **8 | Chapter 1: The Machine Learning Landscape** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:16:49. 

_Segmenting clients based on their purchases so that you can design a different marketing strategy for each segment_ 

This is clustering, which can be achieved using _k_ -means, DBSCAN, and more (see Chapter 9). 

_Representing a complex, high-dimensional dataset in a clear and insightful diagram_ 

This is data visualization, often involving dimensionality reduction techniques (see Chapter 8). 

_Recommending a product that a client may be interested in, based on past purchases_ 

This is a recommender system. One approach is to feed past purchases (and other information about the client) to an artificial neural network (see Chap‐ ter 10), and get it to output the most likely next purchase. This neural net would typically be trained on past sequences of purchases across all clients. 

_Building an intelligent bot for a game_ 

This is often tackled using reinforcement learning (RL; see Chapter 18), which is a branch of machine learning that trains agents (such as bots) to pick the actions that will maximize their rewards over time (e.g., a bot may get a reward every time the player loses some life points), within a given environment (such as the game). The famous AlphaGo program that beat the world champion at the game of Go was built using RL. 

This list could go on and on, but hopefully it gives you a sense of the incredible breadth and complexity of the tasks that machine learning can tackle, and the types of techniques that you would use for each task. 

## **Types of Machine Learning Systems** 

There are so many different types of machine learning systems that it is useful to classify them in broad categories, based on the following criteria: 

- How they are supervised during training (supervised, unsupervised, semisupervised, self-supervised, and others) 

- Whether or not they can learn incrementally on the fly (online versus batch learning) 

- Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model, much like scientists do (instance-based versus model-based learning) 

These criteria are not exclusive; you can combine them in any way you like. For example, a state-of-the-art spam filter may learn on the fly using a deep neural net‐ work model trained using human-provided examples of spam and ham; this makes it an online, model-based, supervised learning system. 

**Types of Machine Learning Systems | 9** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:16:49. 



<!-- Start of picture text -->
Label Om<br>} Om instance<br>Training set<br><!-- End of picture text -->



<!-- Start of picture text -->
Value<br>@ @@<br>@ @<br>@ @ @ 4<br>@®@ @ 9 @<br>@ @<br>ee @<br>@ @ lue?<br>© Value?<br>Feature1<br>New instance<br><!-- End of picture text -->





<!-- Start of picture text -->
Training set<br><!-- End of picture text -->



<!-- Start of picture text -->
Feature 2 I<br>&8 !'<br>oe9 x\Sfa4' g<br>9 ‘ oa ~s<br>9 71Q g \<br>' \<br>I \<br>. \ Feature1<br><!-- End of picture text -->



<!-- Start of picture text -->
ae uae ce an<br>Lemire een s+<br>te: Lapseegs Hip hasip a nPa asc ier alaMets, ee tee  +<br>+ ¢a qe AS poe es oe<br>° automobile ¢<br>rog sagesPewee:aha  betn ae DeOy eeBeas RO °4S gtrya KAY AeeiKeeuz Ph ene r ee.es a ahaa.SO a+eee ra,<br>x A<br>: es TRS Ce wn +e DIR? 5 Tee Se ps Geter v8 9 Deltas. +& Vv<br>airplane seSS ee a ape ogBe Aeemletee SRT.Se ee 4 er<br>Oe,a Oc eeUR grata oO*onoY  RN at umes,<br>orse Bir GREEKS TORS CAN ws eT Ore i eto £ 5<br>A .<br>PO eae Oe are TOU pie So meas<br>v dog e e e0605ot  Shae ae CiB e h a t OASSoaks SE Bagoa”<br>> deer<br>Mee) bse. hey xe aes * SRR eae NMS Re RVR<br>Setai A A 0a EG BR che! ee<br>° *<br><!-- End of picture text -->





<!-- Start of picture text -->
Feature 2 New instances<br>ae “"_N<br>Anomaly = N @<br>@ e*\ ee286 @<br>@@ ort yom<br>© oS @e@oe @<br>0% Se S<br>@®®@ @@ e %e @<br>@ @ Training instances Feature1<br>Feature2<br>@ co) A- eeA° ro) r) e e C) © ©<br>@ ° ° eee L] @ @<br>a ra<br>o) ° @ ° 6 ©,.° Co) ° )<br>Pe eo 2 @ O e°? % ° x @<br>°° & 8 © Ke— cass? ”° °<br>@ oe e@<br>@ @<br>de @ @ @<br>ee eee<br>8 @ © oe @ @ © Feature1<br><!-- End of picture text -->



<!-- Start of picture text -->
y ; y :<br>5 ' a ;<br><!-- End of picture text -->

The resulting model may be quite useful in itself—for example, to repair damaged images or to erase unwanted objects from pictures. But more often than not, a model trained using self-supervised learning is not the final goal. You’ll usually want to tweak and fine-tune the model for a slightly different task—one that you actually care about. 

For example, suppose that what you really want is to have a pet classification model: given a picture of any pet, it will tell you what species it belongs to. If you have a large dataset of unlabeled photos of pets, you can start by training an image-repairing model using self-supervised learning. Once it’s performing well, it should be able to distinguish different pet species: when it repairs an image of a cat whose face is masked, it must know not to add a dog’s face. Assuming your model’s architecture allows it (and most neural network architectures do), it is then possible to tweak the model so that it predicts pet species instead of repairing images. The final step consists of fine-tuning the model on a labeled dataset: the model already knows what cats, dogs, and other pet species look like, so this step is only needed so the model can learn the mapping between the species it already knows and the labels we expect from it. 



Transferring knowledge from one task to another is called _transfer learning_ , and it’s one of the most important techniques in machine learning today, especially when using _deep neural networks_ (i.e., neural networks composed of many layers of neurons). We will discuss this in detail in Part II. 

Some people consider self-supervised learning to be a part of unsupervised learning, since it deals with fully unlabeled datasets. But self-supervised learning uses (gener‐ ated) labels during training, so in that regard it’s closer to supervised learning. And the term “unsupervised learning” is generally used when dealing with tasks like clustering, dimensionality reduction, or anomaly detection, whereas self-supervised learning focuses on the same tasks as supervised learning: mainly classification and regression. In short, it’s best to treat self-supervised learning as its own category. 

##### **Reinforcement learning** 

_Reinforcement learning_ is a very different beast. The learning system, called an _agent_ in this context, can observe the environment, select and perform actions, and get _rewards_ in return (or _penalties_ in the form of negative rewards, as shown in Fig‐ ure 1-13). It must then learn by itself what is the best strategy, called a _policy_ , to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation. 

**16 | Chapter 1: The Machine Learning Landscape** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:16:49. 



<!-- Start of picture text -->
Environment 1221<br>at C>., cn 1) Observe<br>%i S — 2) Select action using policy<br>Agent<br>BE50 points con © Action!<br>Uy? <> = 4) Getreward or penalty<br>y-l b>bad!<br>3 Next time avoid it, cos 5) Update policy (learningstep)<br>n Ba wy 6) Iterate until an optimal policy is found<br><!-- End of picture text -->

##### **Batch learning** 

In _batch learning_ , the system is incapable of learning incrementally: it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called . _offline learning_ 

Unfortunately, a model’s performance tends to decay slowly over time, simply because the world continues to evolve while the model remains unchanged. This phenom‐ enon is often called _model rot_ or _data drift_ . The solution is to regularly retrain the model on up-to-date data. How often you need to do that depends on the use case: if the model classifies pictures of cats and dogs, its performance will decay very slowly, but if the model deals with fast-evolving systems, for example making predictions on the financial market, then it is likely to decay quite fast. 



Even a model trained to classify pictures of cats and dogs may need to be retrained regularly, not because cats and dogs will mutate overnight, but because cameras keep changing, along with image formats, sharpness, brightness, and size ratios. Moreover, people may love different breeds next year, or they may decide to dress their pets with tiny hats—who knows? 

If you want a batch learning system to know about new data (such as a new type of spam), you need to train a new version of the system from scratch on the full dataset (not just the new data, but also the old data), then replace the old model with the new one. Fortunately, the whole process of training, evaluating, and launching a machine learning system can be automated fairly easily (as we saw in Figure 1-3), so even a batch learning system can adapt to change. Simply update the data and train a new version of the system from scratch as often as needed. 

This solution is simple and often works fine, but training using the full set of data can take many hours, so you would typically train a new system only every 24 hours or even just weekly. If your system needs to adapt to rapidly changing data (e.g., to predict stock prices), then you need a more reactive solution. 

Also, training on the full set of data requires a lot of computing resources (CPU, memory space, disk space, disk I/O, network I/O, etc.). If you have a lot of data and you automate your system to train from scratch every day, it will end up costing you a lot of money. If the amount of data is huge, it may even be impossible to use a batch learning algorithm. 

**18 | Chapter 1: The Machine Learning Landscape** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:16:49. 



<!-- Start of picture text -->
a0 0 4. Runand learn<br>(e)<br>Data !<br>New data (on the fly)<br>°<br>-- 1. Train ML model<br>x<br>) Joes |<br>¢----------------:<br><!-- End of picture text -->



<!-- Start of picture text -->
BO, Go° =<br>= Chop into pieces 3 4, Launch!<br>fete) °<br>°<br>*Lots* ofdata tess |<br>1.Studythe problem 2. Tain onlineML Koa<br>4. Analyze errors<br><!-- End of picture text -->





<!-- Start of picture text -->
eature 2<br> AAZ<br>aa Ago<br>A, t [ | At. L<br>A. A Ou<br>Oo lew instance Qu []<br>Oo Oo OO<br><!-- End of picture text -->



<!-- Start of picture text -->
A Aso O<br>Anh m= =<br>AA ARE Oo<br>LI UH 7 UL = Feature 1<br><!-- End of picture text -->

<mark>ee</mark> 



<!-- Start of picture text -->
ew Zea<br>oc 89 |New zealand Demme\ |<br>UO<br>Sol samp 2 gece<br>wn<br>a l<br>© 6<br>5<br>oo]5 ee anceEUs.<br>urkey<br>4<br>25000 30000 35000 40000 45000 50000 55000 60000<br>GDP per capita (USD)<br><!-- End of picture text -->



<!-- Start of picture text -->
9 [Pa<br>Sst 9 9 x 10-5 6,=8*x OF<br>BL ee tee<br>ic) 7 ~_| — —_<br>2].<br>O 6 Ls at<br>gee<br>= 54 —_| O5=44 ~~<br>4 ee SS  eeOO<br>25000 30000 35000 40000 45000 50000 55000 60000<br>GDP per capita (USD)<br><!-- End of picture text -->





<!-- Start of picture text -->
9<br>ef | | | | | | if<br>ee<br>2 |!<br>i 6 |—_2 3 | Qo - |<br>©QE ee ite TeerR108 |<br>“| 5]4 | |ttt<br>25000 30000 35000 40000 45000 50000 55000 60000<br>GDP per capita (USD)<br><!-- End of picture text -->

You are finally ready to run the model to make predictions. For example, say you want to know how happy Cypriots are, and the OECD data does not have the answer. Fortunately, you can use your model to make a good prediction: you look up Cyprus’s GDP per capita, find $37,655, and then apply your model and find that life satisfaction is likely to be somewhere around 3.75 + 37,655 × 6.78 × 10<sup>–5</sup> = 6.30. 

To whet your appetite, Example 1-1 shows the Python code that loads the data, separates the inputs `X` from the labels `y` , creates a scatterplot for visualization, and then trains a linear model and makes a prediction.<sup>5</sup> 

_Example 1-1. Training and running a linear model using Scikit-Learn_ 

```
importmatplotlib.pyplotasplt
importnumpyasnp
importpandasaspd
fromsklearn.linear_modelimportLinearRegression
# Download and prepare the data
data_root="https://github.com/ageron/data/raw/main/"
lifesat=pd.read_csv(data_root+"lifesat/lifesat.csv")
X=lifesat[["GDP per capita (USD)"]].values
y=lifesat[["Life satisfaction"]].values
# Visualize the data
lifesat.plot(kind='scatter', grid=True,
x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()
# Select a linear model
model=LinearRegression()
# Train the model
model.fit(X, y)
# Make a prediction for Cyprus
X_new= [[37_655.2]]  # Cyprus' GDP per capita in 2020
print(model.predict(X_new)) # output: [[6.30165767]]
```

5 It’s OK if you don’t understand all the code yet; I will present Scikit-Learn in the following chapters. 

**Types of Machine Learning Systems | 25** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:16:49. 



## **Main Challenges of Machine Learning** 

In short, since your main task is to select a model and train it on some data, the two things that can go wrong are “bad model” and “bad data”. Let’s start with examples of bad data. 

### **Insufficient Quantity of Training Data** 

For a toddler to learn what an apple is, all it takes is for you to point to an apple and say “apple” (possibly repeating this procedure a few times). Now the child is able to recognize apples in all sorts of colors and shapes. Genius. 

Machine learning is not quite there yet; it takes a lot of data for most machine learning algorithms to work properly. Even for very simple problems you typically need thousands of examples, and for complex problems such as image or speech recognition you may need millions of examples (unless you can reuse parts of an existing model). 

#### **The Unreasonable Effectiveness of Data** 

In a famous paper published in 2001, Microsoft researchers Michele Banko and Eric Brill showed that very different machine learning algorithms, including fairly simple ones, performed almost identically well on a complex problem of natural language disambiguation<sup>6</sup> once they were given enough data (as you can see in Figure 1-21). 

As the authors put it, “these results suggest that we may want to reconsider the tradeoff between spending time and money on algorithm development versus spending it on corpus development”. 

The idea that data matters more than algorithms for complex problems was further popularized by Peter Norvig et al. in a paper titled “The Unreasonable Effectiveness of Data”, published in 2009.<sup>7</sup> It should be noted, however, that small and medium-sized datasets are still very common, and it is not always easy or cheap to get extra training data—so don’t abandon algorithms just yet. 

> 6 For example, knowing whether to write “to”, “two”, or “too”, depending on the context. 

> 7 Peter Norvig et al., “The Unreasonable Effectiveness of Data”, _IEEE Intelligent Systems_ 24, no. 2 (2009): 8–12. 

**Main Challenges of Machine Learning | 27** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:16:49. 



<!-- Start of picture text -->
1.00<br>Ab<br>0.95 Aza<br>a<br>leLr”<br>oOFt3 a l0.90 YiZ =ya 2 ~<br>5 =<br>0.85 A<br>~ ° Va<br>nJ<br>o- 4<br>H<br>0.80<br>—6— Memory-Based<br>0.75 —<— Winnow<br>—A— Perceptron<br>—a—Naive Bayes<br>0.70<br>0.1 1 10 100 1000<br>Millions of Words<br><!-- End of picture text -->



<!-- Start of picture text -->
5 9 8  ColombiaBra | ——_|<br>ri T ee<br>s rr<br>L 5 Ireland<br>a ~~ . uxembourg<br>4 een South Africa<br>0 20000 40000 60000 80000 100000<br>GDP per capita (USD)<br><!-- End of picture text -->

in Brazil you will get a lot of “funk carioca” videos, which sound nothing like James Brown). On the other hand, how else can you get a large training set? 

### **Poor-Quality Data** 

Obviously, if your training data is full of errors, outliers, and noise (e.g., due to poorquality measurements), it will make it harder for the system to detect the underlying patterns, so your system is less likely to perform well. It is often well worth the effort to spend time cleaning up your training data. The truth is, most data scientists spend a significant part of their time doing just that. The following are a couple examples of when you’d want to clean up training data: 

- If some instances are clearly outliers, it may help to simply discard them or try to fix the errors manually. 

- If some instances are missing a few features (e.g., 5% of your customers did not specify their age), you must decide whether you want to ignore this attribute altogether, ignore these instances, fill in the missing values (e.g., with the median age), or train one model with the feature and one model without it. 

### **Irrelevant Features** 

As the saying goes: garbage in, garbage out. Your system will only be capable of learning if the training data contains enough relevant features and not too many irrelevant ones. A critical part of the success of a machine learning project is coming up with a good set of features to train on. This process, called _feature engineering_ , involves the following steps: 

- _Feature selection_ (selecting the most useful features to train on among existing features) 

- _Feature extraction_ (combining existing features to produce a more useful one—as we saw earlier, dimensionality reduction algorithms can help) 

- Creating new features by gathering new data 

Now that we have looked at many examples of bad data, let’s look at a couple examples of bad algorithms. 

### **Overfitting the Training Data** 

Say you are visiting a foreign country and the taxi driver rips you off. You might be tempted to say that _all_ taxi drivers in that country are thieves. Overgeneralizing is something that we humans do all too often, and unfortunately machines can fall into the same trap if we are not careful. In machine learning this is called _overfitting_ : it 

**30 | Chapter 1: The Machine Learning Landscape** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:16:49. 



<!-- Start of picture text -->
9<br>ee<br>Y<br>U<br>a ee eee<br>3 6 IN |<br>2) ¢R LZ | \ of<br>wo<br>uea a<br>se<br>4<br>0 20000 40000 60000 80000 100000<br>GDP per capita (USD)<br><!-- End of picture text -->





<!-- Start of picture text -->
5 es ee<br>8 an<br>@78wn reSeece =|<br>©6::<br>£5<br>4 pe || —--- LinearRegularized modellinear on all model data  on partial data<br>0 20000 40000 60000 80000 100000<br>GDP per capita (USD)<br><!-- End of picture text -->

part of building a machine learning system (you will see a detailed example in the next chapter). 

### **Underfitting the Training Data** 

As you might guess, _underfitting_ is the opposite of overfitting: it occurs when your model is too simple to learn the underlying structure of the data. For example, a linear model of life satisfaction is prone to underfit; reality is just more complex than the model, so its predictions are bound to be inaccurate, even on the training examples. 

Here are the main options for fixing this problem: 

- Select a more powerful model, with more parameters. 

- Feed better features to the learning algorithm (feature engineering). 

- Reduce the constraints on the model (for example by reducing the regularization hyperparameter). 

### **Stepping Back** 

By now you know a lot about machine learning. However, we went through so many concepts that you may be feeling a little lost, so let’s step back and look at the big picture: 

- Machine learning is about making machines get better at some task by learning from data, instead of having to explicitly code rules. 

- There are many different types of ML systems: supervised or not, batch or online, instance-based or model-based. 

- In an ML project you gather data in a training set, and you feed the training set to a learning algorithm. If the algorithm is model-based, it tunes some parameters to fit the model to the training set (i.e., to make good predictions on the training set itself), and then hopefully it will be able to make good predictions on new cases as well. If the algorithm is instance-based, it just learns the examples by heart and generalizes to new instances by using a similarity measure to compare them to the learned instances. 

- The system will not perform well if your training set is too small, or if the data is not representative, is noisy, or is polluted with irrelevant features (garbage in, garbage out). Lastly, your model needs to be neither too simple (in which case it will underfit) nor too complex (in which case it will overfit). 

There’s just one last important topic to cover: once you have trained a model, you don’t want to just “hope” it generalizes to new cases. You want to evaluate it and fine-tune it if necessary. Let’s see how to do that. 

**Main Challenges of Machine Learning | 33** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:16:49. 

## **Testing and Validating** 

The only way to know how well a model will generalize to new cases is to actually try it out on new cases. One way to do that is to put your model in production and monitor how well it performs. This works well, but if your model is horribly bad, your users will complain—not the best idea. 

A better option is to split your data into two sets: the _training set_ and the _test set_ . As these names imply, you train your model using the training set, and you test it using the test set. The error rate on new cases is called the _generalization error_ (or _out-of-sample error_ ), and by evaluating your model on the test set, you get an estimate of this error. This value tells you how well your model will perform on instances it has never seen before. 

If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is overfitting the training data. 



It is common to use 80% of the data for training and _hold out_ 20% for testing. However, this depends on the size of the dataset: if it contains 10 million instances, then holding out 1% means your test set will contain 100,000 instances, probably more than enough to get a good estimate of the generalization error. 

### **Hyperparameter Tuning and Model Selection** 

Evaluating a model is simple enough: just use a test set. But suppose you are hesitat‐ ing between two types of models (say, a linear model and a polynomial model): how can you decide between them? One option is to train both and compare how well they generalize using the test set. 

Now suppose that the linear model generalizes better, but you want to apply some regularization to avoid overfitting. The question is, how do you choose the value of the regularization hyperparameter? One option is to train 100 different models using 100 different values for this hyperparameter. Suppose you find the best hyperparame‐ ter value that produces a model with the lowest generalization error—say, just 5% error. You launch this model into production, but unfortunately it does not perform as well as expected and produces 15% errors. What just happened? 

The problem is that you measured the generalization error multiple times on the test set, and you adapted the model and hyperparameters to produce the best model _for that particular set_ . This means the model is unlikely to perform as well on new data. 

A common solution to this problem is called _holdout validation_ (Figure 1-25): you simply hold out part of the training set to evaluate several candidate models and 

**34 | Chapter 1: The Machine Learning Landscape** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:16:49. 



<!-- Start of picture text -->
wwrwr nmr nr nn nm mM eM = mlevoKKK =<br>u it 1<br>i! 1<br>1<br>eee! 1<br>J<br><!-- End of picture text -->



<!-- Start of picture text -->
eI Clad EETe)<br>iw MeWe go e [eralSee. hy ages Bsawih walsiea<br><!-- End of picture text -->

#### **No Free Lunch Theorem** 

A model is a simplified representation of the data. The simplifications are meant to discard the superfluous details that are unlikely to generalize to new instances. When you select a particular type of model, you are implicitly making _assumptions_ about the data. For example, if you choose a linear model, you are implicitly assuming that the data is fundamentally linear and that the distance between the instances and the straight line is just noise, which can safely be ignored. 

In a famous 1996 paper,<sup>9</sup> David Wolpert demonstrated that if you make absolutely no assumption about the data, then there is no reason to prefer one model over any other. This is called the _No Free Lunch_ (NFL) theorem. For some datasets the best model is a linear model, while for other datasets it is a neural network. There is no model that is _a priori_ guaranteed to work better (hence the name of the theorem). The only way to know for sure which model is best is to evaluate them all. Since this is not possible, in practice you make some reasonable assumptions about the data and evaluate only a few reasonable models. For example, for simple tasks you may evaluate linear models with various levels of regularization, and for a complex problem you may evaluate various neural networks. 

## **Exercises** 

In this chapter we have covered some of the most important concepts in machine learning. In the next chapters we will dive deeper and write more code, but before we do, make sure you can answer the following questions: 

**1.** How would you define machine learning? 

**2.** Can you name four types of applications where it shines? 

**3.** What is a labeled training set? 

**4.** What are the two most common supervised tasks? 

**5.** Can you name four common unsupervised tasks? 

**6.** What type of algorithm would you use to allow a robot to walk in various unknown terrains? 

**7.** What type of algorithm would you use to segment your customers into multiple groups? 

**8.** Would you frame the problem of spam detection as a supervised learning prob‐ lem or an unsupervised learning problem? 

- 9 David Wolpert, “The Lack of A Priori Distinctions Between Learning Algorithms”, _Neural Computation_ 8, no. 7 (1996): 1341–1390. 

**Exercises | 37** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:16:49. 

**9.** What is an online learning system? 

**10.** What is out-of-core learning? 

**11.** What type of algorithm relies on a similarity measure to make predictions? 

**12.** What is the difference between a model parameter and a model hyperparameter? 

**13.** What do model-based algorithms search for? What is the most common strategy they use to succeed? How do they make predictions? 

**14.** Can you name four of the main challenges in machine learning? 

**15.** If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions? 

**16.** What is a test set, and why would you want to use it? 

**17.** What is the purpose of a validation set? 

**18.** What is the train-dev set, when do you need it, and how do you use it? 

**19.** What can go wrong if you tune hyperparameters using the test set? 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

###### **38 | Chapter 1: The Machine Learning Landscape** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:16:49. 

