# **CHAPTER 2 End-to-End Machine Learning Project** 

In this chapter you will work through an example project end to end, pretending to be a recently hired data scientist at a real estate company. This example is fictitious; the goal is to illustrate the main steps of a machine learning project, not to learn anything about the real estate business. Here are the main steps we will walk through: 

**1.** Look at the big picture. 

**2.** Get the data. 

**3.** Explore and visualize the data to gain insights. 

**4.** Prepare the data for machine learning algorithms. 

**5.** Select a model and train it. 

**6.** Fine-tune your model. 

**7.** Present your solution. 

**8.** Launch, monitor, and maintain your system. 

## **Working with Real Data** 

When you are learning about machine learning, it is best to experiment with realworld data, not artificial datasets. Fortunately, there are thousands of open datasets to choose from, ranging across all sorts of domains. Here are a few places you can look to get data: 

- Popular open data repositories: 

   - OpenML.org 

   - Kaggle.com 

**39** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 



<!-- Start of picture text -->
a2 500000<br>eae<br>40 Bop ban. 400000<br>»® Ry ~p hadhy 300000 2<br>E A ee. :<br>5s: Me Sage- :<br>36 Se4 aS ¢ zoow 2<br>34 4,okee 100000 °<br>— . ss~Ne":<br>124 -122 120 -118 -116 114<br>Longitude<br><!-- End of picture text -->

## **Look at the Big Picture** 

Welcome to the Machine Learning Housing Corporation! Your first task is to use California census data to build a model of housing prices in the state. This data includes metrics such as the population, median income, and median housing price for each block group in California. Block groups are the smallest geographical unit for which the US Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people). I will call them “districts” for short. 

Your model should learn from this data and be able to predict the median housing price in any district, given all the other metrics. 



Since you are a well-organized data scientist, the first thing you should do is pull out your machine learning project checklist. You can start with the one in Appendix A; it should work reasonably well for most machine learning projects, but make sure to adapt it to your needs. In this chapter we will go through many check‐ list items, but we will also skip a few, either because they are self-explanatory or because they will be discussed in later chapters. 

### **Frame the Problem** 

The first question to ask your boss is what exactly the business objective is. Building a model is probably not the end goal. How does the company expect to use and benefit from this model? Knowing the objective is important because it will determine how you frame the problem, which algorithms you will select, which performance measure you will use to evaluate your model, and how much effort you will spend tweaking it. 

Your boss answers that your model’s output (a prediction of a district’s median housing price) will be fed to another machine learning system (see Figure 2-2), along with many other signals.<sup>2</sup> This downstream system will determine whether it is worth investing in a given area. Getting this right is critical, as it directly affects revenue. 

The next question to ask your boss is what the current solution looks like (if any). The current situation will often give you a reference for performance, as well as insights on how to solve the problem. Your boss answers that the district housing prices are currently estimated manually by experts: a team gathers up-to-date information about a district, and when they cannot get the median housing price, they estimate it using complex rules. 

> 2 A piece of information fed to a machine learning system is often called a _signal_ , in reference to Claude Shannon’s information theory, which he developed at Bell Labs to improve telecommunications. His theory: you want a high signal-to-noise ratio. 

**Look at the Big Picture | 41** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 



<!-- Start of picture text -->
Your component — Other signals<br>—(@<br>componentsUpstream District pricing Investment analysis:<br>a pe Investments<br><!-- End of picture text -->



() ~~j—~~ ( (©) 0) 

#### **Notations** 

This equation introduces several very common machine learning notations that I will use throughout this book: 

- _m_ is the number of instances in the dataset you are measuring the RMSE on. 

   - For example, if you are evaluating the RMSE on a validation set of 2,000 districts, then _m_ = 2,000. 

- **x**<sup>(</sup><sup>_i_)</sup> is a vector of all the feature values (excluding the label) of the _i_<sup>th</sup> instance in the dataset, and _y_<sup>(</sup><sup>_i_)</sup> is its label (the desired output value for that instance). 

   - For example, if the first district in the dataset is located at longitude –118.29°, latitude 33.91°, and it has 1,416 inhabitants with a median income of $38,372, and the median house value is $156,400 (ignoring other features for now), then: 



<!-- Start of picture text -->
−118.29<br>x 1 = 33.91<br>1,416<br>38,372<br><!-- End of picture text -->

and: 



<!-- Start of picture text -->
y 1 = 156,400<br><!-- End of picture text -->

- **X** is a matrix containing all the feature values (excluding labels) of all instances in the dataset. There is one row per instance, and the _i_<sup>th</sup> row is equal to the transpose of **x**<sup>(</sup><sup>_i_)</sup> , noted ( **x**<sup>(</sup><sup>_i_)</sup> )<sup>⊺</sup> .<sup>3</sup> 

   - For example, if the first district is as just described, then the matrix **X** looks like this: 



<!-- Start of picture text -->
1 ⊺<br>x<br>2 ⊺<br>x<br>−118.29 33.91 1,416 38,372<br>X  = ⋮ =<br>⋮ ⋮ ⋮ ⋮<br>1999 ⊺<br>x<br>2000 ⊺<br>x<br><!-- End of picture text -->

- 3 Recall that the transpose operator flips a column vector into a row vector (and vice versa). 

###### **44 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

- _h_ is your system’s prediction function, also called a _hypothesis_ . When your system is given an instance’s feature vector **x**<sup>(</sup><sup>_i_)</sup> , it outputs a predicted value _ŷ_<sup>(</sup><sup>_i_)</sup> = _h_ ( **x**<sup>(</sup><sup>_i_)</sup> ) for that instance ( _ŷ_ is pronounced “y-hat”). 

   - For example, if your system predicts that the median housing price in the first district is $158,400, then _ŷ_<sup>(1)</sup> = _h_ ( **x**<sup>(1)</sup> ) = 158,400. The prediction error for this district is _ŷ_<sup>(1)</sup> – _y_<sup>(1)</sup> = 2,000. 

- RMSE( **X** , _h_ ) is the cost function measured on the set of examples using your hypothesis _h_ . 

We use lowercase italic font for scalar values (such as _m_ or _y_<sup>(</sup><sup>_i_)</sup> ) and function names (such as _h_ ), lowercase bold font for vectors (such as **x**<sup>(</sup><sup>_i_)</sup> ), and uppercase bold font for matrices (such as **X** ). 

Although the RMSE is generally the preferred performance measure for regression tasks, in some contexts you may prefer to use another function. For example, if there are many outlier districts. In that case, you may consider using the _mean absolute error_ (MAE, also called the _average absolute deviation_ ), shown in Equation 2-2: 

##### _Equation 2-2. Mean absolute error (MAE)_ 



Both the RMSE and the MAE are ways to measure the distance between two vectors: the vector of predictions and the vector of target values. Various distance measures, or _norms_ , are possible: 

- Computing the root of a sum of squares (RMSE) corresponds to the _Euclidean norm_ : this is the notion of distance we are all familiar with. It is also called the ℓ2 _norm_ , noted ∥ · ∥ 2 (or just ∥ · ∥ ). 

- Computing the sum of absolutes (MAE) corresponds to the ℓ1 _norm_ , noted ∥ · ∥ 1. This is sometimes called the _Manhattan norm_ because it measures the distance between two points in a city if you can only travel along orthogonal city blocks. 

- More generally, the ℓ _k norm_ of a vector **v** containing _n_ elements is defined as ∥ **v** ∥ _k_ = (| _v_ 1|<sup>_k_</sup> + | _v_ 2|<sup>_k_</sup> + ... + | _vn_ |<sup>_k_</sup> )<sup>1/</sup><sup>_k_</sup> . ℓ0 gives the number of nonzero elements in the vector, and ℓ∞ gives the maximum absolute value in the vector. 

The higher the norm index, the more it focuses on large values and neglects small ones. This is why the RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred. 

**Look at the Big Picture | 45** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

### **Check the Assumptions** 

Lastly, it is good practice to list and verify the assumptions that have been made so far (by you or others); this can help you catch serious issues early on. For example, the district prices that your system outputs are going to be fed into a downstream machine learning system, and you assume that these prices are going to be used as such. But what if the downstream system converts the prices into categories (e.g., “cheap”, “medium”, or “expensive”) and then uses those categories instead of the prices themselves? In this case, getting the price perfectly right is not important at all; your system just needs to get the category right. If that’s so, then the problem should have been framed as a classification task, not a regression task. You don’t want to find this out after working on a regression system for months. 

Fortunately, after talking with the team in charge of the downstream system, you are confident that they do indeed need the actual prices, not just categories. Great! You’re all set, the lights are green, and you can start coding now! 

## **Get the Data** 

It’s time to get your hands dirty. Don’t hesitate to pick up your laptop and walk through the code examples. As I mentioned in the preface, all the code examples in this book are open source and available online as Jupyter notebooks, which are interactive documents containing text, images, and executable code snippets (Python in our case). In this book I will assume you are running these notebooks on Google Colab, a free service that lets you run any Jupyter notebook directly online, without having to install anything on your machine. If you want to use another online plat‐ form (e.g., Kaggle) or if you want to install everything locally on your own machine, please see the instructions on the book’s GitHub page. 

### **Running the Code Examples Using Google Colab** 

First, open a web browser and visit _https://homl.info/colab3_ : this will lead you to Google Colab, and it will display the list of Jupyter notebooks for this book (see Figure 2-3). You will find one notebook per chapter, plus a few extra notebooks and tutorials for NumPy, Matplotlib, Pandas, linear algebra, and differential calculus. For example, if you click _02_end_to_end_machine_learning_project.ipynb_ , the notebook from Chapter 2 will open up in Google Colab (see Figure 2-4). 

A Jupyter notebook is composed of a list of cells. Each cell contains either executable code or text. Try double-clicking the first text cell (which contains the sentence “Welcome to Machine Learning Housing Corp.!”). This will open the cell for editing. Notice that Jupyter notebooks use Markdown syntax for formatting (e.g., `**bold**` , `*italics*` , `# Title` , `[url](link text)` , and so on). Try modifying this text, then press Shift-Enter to see the result. 

**46 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 



<!-- Start of picture text -->
Enter a GitHub URL or search by organization or user oO Include private repos<br>ageron Q<br>Repository: [7 Branch: [4<br>ageron/handson-ml3 v master v<br>Path<br>QO 01_the_machine_learning_landscape.ipynb a GG<br>©) 02.end to_end machine learning_project.ipynb a GZ<br>rs) 03_classification.ipynb Aa GB<br>i) 04_training_linear_models.ipynb AG<br><!-- End of picture text -->



<!-- Start of picture text -->
co © 02_end_to_end_machine_learning_project.ipynb 2_end_to_end_machine_learning_project.ipyn cD Share<br>File Edit View Insert Runtime Tools Help<br>= + Code + Text # Copy to Drive; oY RAMheh =[1 Y @y EditingEditi a<br>Q Chapter 2 - End-to-end Machine Learning Project<br>= Welcome to Machine Learning Housing Corp.! Your task is to predict median house values in<br>Californian districts, given a number of features from these districts.<br>Oo This notebook contains all the sample code and solutions to the exercices in chapter 2.<br>Text cells:<br>~ Setup double-click to edit<br>First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare<br>a function to save the figures.<br>Code cells:<br>{ ] # Python 23.7 is; required; click to edit<br>import sys<br>assert sys.version_info >= (3, 7)<br><!-- End of picture text -->

Next, create a new code cell by selecting Insert → “Code cell” from the menu. Alternatively, you can click the + Code button in the toolbar, or hover your mouse over the bottom of a cell until you see + Code and + Text appear, then click + Code. In the new code cell, type some Python code, such as `print("Hello World")` , then press Shift-Enter to run this code (or click the ▷ button on the left side of the cell). 

If you’re not logged in to your Google account, you’ll be asked to log in now (if you don’t already have a Google account, you’ll need to create one). Once you are logged in, when you try to run the code you’ll see a security warning telling you that this notebook was not authored by Google. A malicious person could create a notebook that tries to trick you into entering your Google credentials so they can access your personal data, so before you run a notebook, always make sure you trust its author (or double-check what each code cell will do before running it). Assuming you trust me (or you plan to check every code cell), you can now click “Run anyway”. 

Colab will then allocate a new _runtime_ for you: this is a free virtual machine located on Google’s servers that contains a bunch of tools and Python libraries, including everything you’ll need for most chapters (in some chapters, you’ll need to run a command to install additional libraries). This will take a few seconds. Next, Colab will automatically connect to this runtime and use it to execute your new code cell. Importantly, the code runs on the runtime, _not_ on your machine. The code’s output will be displayed under the cell. Congrats, you’ve run some Python code on Colab! 



To insert a new code cell, you can also type Ctrl-M (or Cmd-M on macOS) followed by A (to insert above the active cell) or B (to insert below). There are many other keyboard shortcuts available: you can view and edit them by typing Ctrl-M (or Cmd-M) then H. If you choose to run the notebooks on Kaggle or on your own machine using JupyterLab or an IDE such as Visual Studio Code with the Jupyter extension, you will see some minor differences— runtimes are called _kernels_ , the user interface and keyboard short‐ cuts are slightly different, etc.—but switching from one Jupyter environment to another is not too hard. 

### **Saving Your Code Changes and Your Data** 

You can make changes to a Colab notebook, and they will persist for as long as you keep your browser tab open. But once you close it, the changes will be lost. To avoid this, make sure you save a copy of the notebook to your Google Drive by selecting File → “Save a copy in Drive”. Alternatively, you can download the notebook to your computer by selecting File → Download → “Download .ipynb”. Then you can later visit _https://colab.research.google.com_ and open the notebook again (either from Google Drive or by uploading it from your computer). 

**48 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 





<!-- Start of picture text -->
= files (> Download<br>a ce Rename file<br>ce. Delete file<br><? gy sample_data<br>F B my_great_model my} Copy path<br>—© @+ Refresh<br><!-- End of picture text -->

run a cell. If this happens, the subsequent code cells are likely to fail. For example, the very first code cell in each notebook contains setup code (such as imports), so make sure you run it first, or else nothing will work. 



If you ever run into a weird error, try restarting the runtime (by selecting Runtime → “Restart runtime” from the menu) and then run all the cells again from the beginning of the notebook. This often solves the problem. If not, it’s likely that one of the changes you made broke the notebook: just revert to the original notebook and try again. If it still fails, please file an issue on GitHub. 

### **Book Code Versus Notebook Code** 

You may sometimes notice some little differences between the code in this book and the code in the notebooks. This may happen for several reasons: 

- A library may have changed slightly by the time you read these lines, or perhaps despite my best efforts I made an error in the book. Sadly, I cannot magically fix the code in your copy of this book (unless you are reading an electronic copy and you can download the latest version), but I _can_ fix the notebooks. So, if you run into an error after copying code from this book, please look for the fixed code in the notebooks: I will strive to keep them error-free and up-to-date with the latest library versions. 

- The notebooks contain some extra code to beautify the figures (adding labels, setting font sizes, etc.) and to save them in high resolution for this book. You can safely ignore this extra code if you want. 

I optimized the code for readability and simplicity: I made it as linear and flat as possible, defining very few functions or classes. The goal is to ensure that the code you are running is generally right in front of you, and not nested within several layers of abstractions that you have to search through. This also makes it easier for you to play with the code. For simplicity, there’s limited error handling, and I placed some of the least common imports right where they are needed (instead of placing them at the top of the file, as is recommended by the PEP 8 Python style guide). That said, your production code will not be very different: just a bit more modular, and with additional tests and error handling. 

OK! Once you’re comfortable with Colab, you’re ready to download the data. 

### **Download the Data** 

In typical environments your data would be available in a relational database or some other common data store, and spread across multiple tables/documents/files. 

###### **50 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

To access it, you would first need to get your credentials and access authorizations<sup>4</sup> and familiarize yourself with the data schema. In this project, however, things are much simpler: you will just download a single compressed file, _housing.tgz_ , which contains a comma-separated values (CSV) file called _housing.csv_ with all the data. 

Rather than manually downloading and decompressing the data, it’s usually prefera‐ ble to write a function that does it for you. This is useful in particular if the data changes regularly: you can write a small script that uses the function to fetch the latest data (or you can set up a scheduled job to do that automatically at regular intervals). Automating the process of fetching the data is also useful if you need to install the dataset on multiple machines. 

Here is the function to fetch and load the data: 

```
frompathlibimportPath
importpandasaspd
importtarfile
importurllib.request
```

```
defload_housing_data():
tarball_path=Path("datasets/housing.tgz")
ifnottarball_path.is_file():
Path("datasets").mkdir(parents=True, exist_ok=True)
url="https://github.com/ageron/data/raw/main/housing.tgz"
urllib.request.urlretrieve(url, tarball_path)
withtarfile.open(tarball_path) ashousing_tarball:
housing_tarball.extractall(path="datasets")
returnpd.read_csv(Path("datasets/housing/housing.csv"))
```

```
housing=load_housing_data()
```

When `load_housing_data()` is called, it looks for the _datasets/housing.tgz_ file. If it does not find it, it creates the _datasets_ directory inside the current directory (which is _/content_ by default, in Colab), downloads the _housing.tgz_ file from the _ageron/data_ GitHub repository, and extracts its content into the _datasets_ directory; this creates the _datasets_ / _housing_ directory with the _housing.csv_ file inside it. Lastly, the function loads this CSV file into a Pandas DataFrame object containing all the data, and returns it. 

### **Take a Quick Look at the Data Structure** 

You start by looking at the top five rows of data using the DataFrame’s `head()` method (see Figure 2-6). 

> 4 You might also need to check legal constraints, such as private fields that should never be copied to unsafe data stores. 

**Get the Data | 51** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

|housing.head()<br>|<br>|||||
|---|---|---|---|---|---|
|longitude|latitude|housing_median_age|median_income|ocean_proximity|median_house_value|
|0<br>122.23|37.88|41.0|8.3252|NEARBAY|452600.0|
|1<br>122.22|37.86|21.0|8.3014|NEARBAY|358500.0|
|2<br>“122.24|37.85|52.0|7.2574|NEAR BAY|352100.0|
|3<br>122.25|37.85|52.0|5.6431|NEARBAY|341300.0|
|4<br>122.25|37.85|52.0|3.8462|NEARBAY|342200.0|





|housin|g.describe()<br>longitude|latitude|housing_median_age|total_rooms|total_bedrooms|median_house_value|
|---|---|---|---|---|---|---|
|count|20640.000000|20640.000000|20640.000000|20640.000000|20433.000000|20640.000000|
|mean|= -119.569704|35.631861|28.639486|2635.763081|537.870553|206855.816909|
|std|2.003532|2.135952|12.585558|2181.615252|421.385070|115395.615874|
|min|-124.350000|32.540000|1.000000|2.000000|1.000000|14999.000000|
|25%|=-121.800000|33.930000|18.000000|1447.750000|296.000000|119600.000000|
|50%|-118.490000|34.260000|29.000000|+ 2127.000000|435.000000|179700.000000|
|75%|-118.010000|37.710000|37.000000|3148.000000|647.000000|264725.000000|
|max|-114.310000|41.950000|52.000000|39320.000000|6445.000000|500001.000000|





<!-- Start of picture text -->
longitude latitude housing_median_age<br>ees2000 200 rawee f| |_|} a |<br>> : =yg<br>1000 600<br>1500 |<br>:000 col lalla |<br>500 500 200 a “al<br>; ° nin |<br>-124 -122 -120 -118 -116 -114 34 36 38 40 42 0 10 20 30 40 50<br>total_rooms 5000 total_bedrooms population<br>3000<br>° ° °<br>0 10000 20000 30000 40000 0 2000 4000 6000 0 10000 20000 30000<br>5000 households median_income median_house_value<br>s000 Th’. 1 Utd 1500 Ts ee ee | 1000 a<br>el[Borevos} olliaa‘ee|<br>“TR|° a sof= |HH“ae|71.a|i<br>01000 2000 3000 4000 5000 6000 ay 2.5 5.0 75 10.0 12.5 15.0 ° 0 100000 200000 300000 400000 500000<br><!-- End of picture text -->

Looking at these histograms, you notice a few things: 

- First, the median income attribute does not look like it is expressed in US dollars (USD). After checking with the team that collected the data, you are told that the data has been scaled and capped at 15 (actually, 15.0001) for higher median incomes, and at 0.5 (actually, 0.4999) for lower median incomes. The numbers represent roughly tens of thousands of dollars (e.g., 3 actually means about $30,000). Working with preprocessed attributes is common in machine learning, and it is not necessarily a problem, but you should try to understand how the data was computed. 

- The housing median age and the median house value were also capped. The latter may be a serious problem since it is your target attribute (your labels). Your machine learning algorithms may learn that prices never go beyond that limit. You need to check with your client team (the team that will use your system’s output) to see if this is a problem or not. If they tell you that they need precise predictions even beyond $500,000, then you have two options: 

   - Collect proper labels for the districts whose labels were capped. 

   - Remove those districts from the training set (and also from the test set, since your system should not be evaluated poorly if it predicts values beyond $500,000). 

- These attributes have very different scales. We will discuss this later in this chapter, when we explore feature scaling. 

- Finally, many histograms are _skewed right_ : they extend much farther to the right of the median than to the left. This may make it a bit harder for some machine learning algorithms to detect patterns. Later, you’ll try transforming these attributes to have more symmetrical and bell-shaped distributions. 

You should now have a better understanding of the kind of data you’re dealing with. 



Wait! Before you look at the data any further, you need to create a test set, put it aside, and never look at it. 

### **Create a Test Set** 

It may seem strange to voluntarily set aside part of the data at this stage. After all, you have only taken a quick glance at the data, and surely you should learn a whole lot more about it before you decide what algorithms to use, right? This is true, but your brain is an amazing pattern detection system, which also means that it is highly prone to overfitting: if you look at the test set, you may stumble upon some 

**Get the Data | 55** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

seemingly interesting pattern in the test data that leads you to select a particular kind of machine learning model. When you estimate the generalization error using the test set, your estimate will be too optimistic, and you will launch a system that will not perform as well as expected. This is called _data snooping_ bias. 

Creating a test set is theoretically simple; pick some instances randomly, typically 20% of the dataset (or less if your dataset is very large), and set them aside: 

```
importnumpyasnp
```

```
defshuffle_and_split_data(data, test_ratio):
shuffled_indices=np.random.permutation(len(data))
test_set_size=int(len(data) *test_ratio)
test_indices=shuffled_indices[:test_set_size]
train_indices=shuffled_indices[test_set_size:]
returndata.iloc[train_indices], data.iloc[test_indices]
```

You can then use this function like this: 

```
>>> train_set, test_set=shuffle_and_split_data(housing, 0.2)
>>> len(train_set)
16512
>>> len(test_set)
4128
```

Well, this works, but it is not perfect: if you run the program again, it will generate a different test set! Over time, you (or your machine learning algorithms) will get to see the whole dataset, which is what you want to avoid. 

One solution is to save the test set on the first run and then load it in subsequent runs. Another option is to set the random number generator’s seed (e.g., with `np.random.seed(42)` )<sup>6</sup> before calling `np.random.permutation()` so that it always generates the same shuffled indices. 

However, both these solutions will break the next time you fetch an updated dataset. To have a stable train/test split even after updating the dataset, a common solution is to use each instance’s identifier to decide whether or not it should go in the test set (assuming instances have unique and immutable identifiers). For example, you could compute a hash of each instance’s identifier and put that instance in the test set if the hash is lower than or equal to 20% of the maximum hash value. This ensures that the test set will remain consistent across multiple runs, even if you refresh the dataset. The new test set will contain 20% of the new instances, but it will not contain any instance that was previously in the training set. 

> 6 You will often see people set the random seed to 42. This number has no special property, other than being the Answer to the Ultimate Question of Life, the Universe, and Everything. 

###### **56 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

Here is a possible implementation: 

```
fromzlibimportcrc32
```

```
defis_id_in_test_set(identifier, test_ratio):
returncrc32(np.int64(identifier)) <test_ratio*2**32
```

```
defsplit_data_with_id_hash(data, test_ratio, id_column):
ids=data[id_column]
in_test_set=ids.apply(lambdaid_: is_id_in_test_set(id_, test_ratio))
returndata.loc[~in_test_set], data.loc[in_test_set]
```

Unfortunately, the housing dataset does not have an identifier column. The simplest solution is to use the row index as the ID: 

```
housing_with_id=housing.reset_index()  # adds an `index` column
train_set, test_set=split_data_with_id_hash(housing_with_id, 0.2, "index")
```

If you use the row index as a unique identifier, you need to make sure that new data gets appended to the end of the dataset and that no row ever gets deleted. If this is not possible, then you can try to use the most stable features to build a unique identifier. For example, a district’s latitude and longitude are guaranteed to be stable for a few million years, so you could combine them into an ID like so:<sup>7</sup> 

```
housing_with_id["id"] =housing["longitude"] *1000+housing["latitude"]
train_set, test_set=split_data_with_id_hash(housing_with_id, 0.2, "id")
```

Scikit-Learn provides a few functions to split datasets into multiple subsets in various ways. The simplest function is `train_test_split()` , which does pretty much the same thing as the `shuffle_and_split_data()` function we defined earlier, with a couple of additional features. First, there is a `random_state` parameter that allows you to set the random generator seed. Second, you can pass it multiple datasets with an identical number of rows, and it will split them on the same indices (this is very useful, for example, if you have a separate DataFrame for labels): 

```
fromsklearn.model_selectionimporttrain_test_split
```

```
train_set, test_set=train_test_split(housing, test_size=0.2, random_state=42)
```

So far we have considered purely random sampling methods. This is generally fine if your dataset is large enough (especially relative to the number of attributes), but if it is not, you run the risk of introducing a significant sampling bias. When employees at a survey company decides to call 1,000 people to ask them a few questions, they don’t just pick 1,000 people randomly in a phone book. They try to ensure that these 1,000 people are representative of the whole population, with regard to the questions they want to ask. For example, the US population is 51.1% females and 

> 7 The location information is actually quite coarse, and as a result many districts will have the exact same ID, so they will end up in the same set (test or train). This introduces some unfortunate sampling bias. 

**Get the Data | 57** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

48.9% males, so a well-conducted survey in the US would try to maintain this ratio in the sample: 511 females and 489 males (at least if it seems possible that the answers may vary across genders). This is called _stratified sampling_ : the population is divided into homogeneous subgroups called _strata_ , and the right number of instances are sampled from each stratum to guarantee that the test set is representative of the overall population. If the people running the survey used purely random sampling, there would be about a 10.7% chance of sampling a skewed test set with less than 48.5% female or more than 53.5% female participants. Either way, the survey results would likely be quite biased. 

Suppose you’ve chatted with some experts who told you that the median income is a very important attribute to predict median housing prices. You may want to ensure that the test set is representative of the various categories of incomes in the whole dataset. Since the median income is a continuous numerical attribute, you first need to create an income category attribute. Let’s look at the median income histogram more closely (back in Figure 2-8): most median income values are clustered around 1.5 to 6 (i.e., $15,000–$60,000), but some median incomes go far beyond 6. It is important to have a sufficient number of instances in your dataset for each stratum, or else the estimate of a stratum’s importance may be biased. This means that you should not have too many strata, and each stratum should be large enough. The following code uses the `pd.cut()` function to create an income category attribute with five categories (labeled from 1 to 5); category 1 ranges from 0 to 1.5 (i.e., less than $15,000), category 2 from 1.5 to 3, and so on: 

```
housing["income_cat"] =pd.cut(housing["median_income"],
bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
labels=[1, 2, 3, 4, 5])
```

These income categories are represented in Figure 2-9: 

```
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()
```

Now you are ready to do stratified sampling based on the income category. ScikitLearn provides a number of splitter classes in the `sklearn.model_selection` package that implement various strategies to split your dataset into a training set and a test set. Each splitter has a `split()` method that returns an iterator over different training/ test splits of the same data. 

**58 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 



<!-- Start of picture text -->
oo=)55wy on 6000 5000 |ori<br>a<br>4 4000<br>{o}oe i i<br>& 3000<br>=<br>=o3 2000 mah fl<br>ie0 Eh<br>1 2 3 4 5<br>Income category<br><!-- End of picture text -->

Overall% Stratified% Random% Strat. Error% Rand. Error % 

|IncomeCategory|
|---|



|1|3.98|4.00|4.24|0.36|6.45|
|---|---|---|---|---|---|
|2|31.88|31.88|30.74|-0.02|-3.59|
|3|35.06|35.05|34.52|-0.01|-1.53|
|4|17.63|17.64|18.41|0.03|4.42|
|5|11.44|11.43|12.09|-0.08|5.63|





<!-- Start of picture text -->
ol, . SS<br>ao {2 as.<br>oO z 7 N<br>38 3<br>®a 5. mz*5we_,<br>—124 —122 —-120 —-118 —116 —114<br>longitude<br><!-- End of picture text -->



<!-- Start of picture text -->
ee<br>40 = ‘e<br>Y 38 — 2g So &<br>— 36 ies "<br>: ToT i : he 2<br>-124 -122 -120 -118 -116 -114<br>longitude<br><!-- End of picture text -->



<!-- Start of picture text -->
42 —— - @ population<br>.eeex ¢<br>.<br>Zl .af ot, .<br>oe e . &<br>40 ° ~ ef, . Cary 400000<br>iy<br>oy ii rs<br>38 hare 300000 ><br>© dk ie @ a!<br>Fa : }°<br>S log! ve oc<br>& te I c!<br>36 —* Janet 200000 F<br>Pig © &: MPee’”.e-e E<br>e°me ° .8e © 0%<br>° EM yee he}<br>34 Oh, ite y 100000<br>os 4y . e a<br>>’ .oae .<br>-124 -122 -120 -118 -116 -114<br>longitude<br><!-- End of picture text -->



<!-- Start of picture text -->
3 Re S| — . ~ iea att teay THee era titiesed |<br>gy! 00000 ee) ‘4 ESS re . ol |) apBabsati ! sal a Hh<br>7)& x“aa . aRefs.”a . .: BoeBEBes ela HHH HAH<br>200000<br>&| aanReis- - ifncieect 3a» ols: . bsRTMit TW HAHTHH TRHTHHITTITE} i HUEHiUe<br>E —:.: elle :— -_<br>ow 5 —<br>< 10 sob, siggiad tsan : xyLey2. effc a tete ra e tn : Cesel F|<br>. Rae ate® . . pp ekeceg atta RTT ea oa<br>3J | .. seeks | | i<br>oa<br>5 Li ot ! a Ss Byti | 11) bb ~ WII|<br>40000 : . = - ; .<br>w€ 30000 oe ° . . * . ee<br>o te .<br>O° ° . ? os . as<br>= 20000 Yeeces! Se te ieee es<br>so}BB 00 sooo° || . wpeeriety,i aoo o-fatt t1 et RaeSteeORF" aoty oefe . :Pas}shezttt8, feodt| aaaees TH ace<br>° ies | j<br>o°Eo |ieee ooeerTate = UghNO MR a 17: as° ERRan .<br>E20 4 EF nie 4 *~y Mates! Bsc.<br>Ss§ Ss8 Ss8 & s8&8 median_inci i “ 3 88 88 go 8  R<br>median house_value— snecome Smaltotal_rooms some’ Rousingg_median_agedian_age<br><!-- End of picture text -->



<!-- Start of picture text -->
Se ae a a ee  Se Se PT<br>: “a n<br>3 400000 Peee aeeee<br>®> 300000 aga 4 * a hs oe «< itn’vad. Cia<br>‘al<br>©<br>-!200000 —e sualP<br>3 |= | |<br>£ 100000 | : a ae?<br>0 2 4 6 8 10 12 14<br>median_income<br><!-- End of picture text -->





<!-- Start of picture text -->
1 08 04 0 -0.4 0.8 -1<br>a a . 2.<br>; rl a rc ea 3<br>1 1 1 -1 -1 -1<br>0 0 0 0 0 0 0<br>S aS pee Es Bo ee a ba a<br>eee @ gee aSage a Tene : aol aeoe ae<br><!-- End of picture text -->

compare it to the number of rooms. And the population per household also seems like an interesting attribute combination to look at. You create these new attributes as follows: 

```
housing["rooms_per_house"] =housing["total_rooms"] /housing["households"]
housing["bedrooms_ratio"] =housing["total_bedrooms"] /housing["total_rooms"]
housing["people_per_house"] =housing["population"] /housing["households"]
```

And then you look at the correlation matrix again: 

```
>>> corr_matrix=housing.corr()
>>> corr_matrix["median_house_value"].sort_values(ascending=False)
median_house_value    1.000000
median_income         0.688380
rooms_per_house       0.143663
total_rooms           0.137455
housing_median_age    0.102175
households            0.071426
total_bedrooms        0.054635
population           -0.020153
people_per_house     -0.038224
longitude            -0.050859
latitude             -0.139584
bedrooms_ratio       -0.256397
Name: median_house_value, dtype: float64
```

Hey, not bad! The new `bedrooms_ratio` attribute is much more correlated with the median house value than the total number of rooms or bedrooms. It’s a strong negative correlation, so it looks like houses with a lower bedroom/room ratio tend to be more expensive. The number of rooms per household is also more informative than the total number of rooms in a district—obviously the larger the houses, the more expensive they are. 

This round of exploration does not have to be absolutely thorough; the point is to start off on the right foot and quickly gain insights that will help you get a first reasonably good prototype. But this is an iterative process: once you get a prototype up and running, you can analyze its output to gain more insights and come back to this exploration step. 

## **Prepare the Data for Machine Learning Algorithms** 

It’s time to prepare the data for your machine learning algorithms. Instead of doing this manually, you should write functions for this purpose, for several good reasons: 

- This will allow you to reproduce these transformations easily on any dataset (e.g., the next time you get a fresh dataset). 

- You will gradually build a library of transformation functions that you can reuse in future projects. 

**Prepare the Data for Machine Learning Algorithms | 67** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

- You can use these functions in your live system to transform the new data before feeding it to your algorithms. 

- This will make it possible for you to easily try various transformations and see which combination of transformations works best. 

But first, revert to a clean training set (by copying `strat_train_set` once again). You should also separate the predictors and the labels, since you don’t necessarily want to apply the same transformations to the predictors and the target values (note that `drop()` creates a copy of the data and does not affect `strat_train_set` ): 

```
housing=strat_train_set.drop("median_house_value", axis=1)
housing_labels=strat_train_set["median_house_value"].copy()
```

### **Clean the Data** 

Most machine learning algorithms cannot work with missing features, so you’ll need to take care of these. For example, you noticed earlier that the `total_bedrooms` attribute has some missing values. You have three options to fix this: 

**1.** Get rid of the corresponding districts. 

**2.** Get rid of the whole attribute. 

**3.** Set the missing values to some value (zero, the mean, the median, etc.). This is called _imputation_ . 

You can accomplish these easily using the Pandas DataFrame’s `dropna()` , `drop()` , and `fillna()` methods: 

```
housing.dropna(subset=["total_bedrooms"], inplace=True)  # option 1
```

```
housing.drop("total_bedrooms", axis=1)  # option 2
```

```
median=housing["total_bedrooms"].median()  # option 3
housing["total_bedrooms"].fillna(median, inplace=True)
```

You decide to go for option 3 since it is the least destructive, but instead of the preceding code, you will use a handy Scikit-Learn class: `SimpleImputer` . The benefit is that it will store the median value of each feature: this will make it possible to impute missing values not only on the training set, but also on the validation set, the test set, and any new data fed to the model. To use it, first you need to create a `SimpleImputer` instance, specifying that you want to replace each attribute’s missing values with the median of that attribute: 

```
fromsklearn.imputeimportSimpleImputer
```

```
imputer=SimpleImputer(strategy="median")
```

**68 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

Since the median can only be computed on numerical attributes, you then need to create a copy of the data with only the numerical attributes (this will exclude the text attribute `ocean_proximity` ): 

```
housing_num=housing.select_dtypes(include=[np.number])
```

Now you can fit the `imputer` instance to the training data using the `fit()` method: 

```
imputer.fit(housing_num)
```

The `imputer` has simply computed the median of each attribute and stored the result in its `statistics_` instance variable. Only the `total_bedrooms` attribute had missing values, but you cannot be sure that there won’t be any missing values in new data after the system goes live, so it is safer to apply the `imputer` to all the numerical attributes: 

```
>>> imputer.statistics_
array([-118.51 , 34.26 , 29. , 2125. , 434. , 1167. , 408. , 3.5385])
>>> housing_num.median().values
array([-118.51 , 34.26 , 29. , 2125. , 434. , 1167. , 408. , 3.5385])
```

Now you can use this “trained” `imputer` to transform the training set by replacing missing values with the learned medians: 

```
X=imputer.transform(housing_num)
```

Missing values can also be replaced with the mean value ( `strategy="mean"` ), or with the most frequent value ( `strategy="most_frequent"` ), or with a constant value ( `strategy="constant", fill_value=` …). The last two strategies support nonnumerical data. 



There are also more powerful imputers available in the `sklearn.impute` package (both for numerical features only): 

- `KNNImputer` replaces each missing value with the mean of the _k_ -nearest neighbors’ values for that feature. The distance is based on all the available features. 

- `IterativeImputer` trains a regression model per feature to predict the missing values based on all the other available features. It then trains the model again on the updated data, and repeats the process several times, improving the models and the replacement values at each iteration. 

**Prepare the Data for Machine Learning Algorithms | 69** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

#### **Scikit-Learn Design** 

Scikit-Learn’s API is remarkably well designed. These are the main design principles:<sup>9</sup> 

##### _Consistency_ 

All objects share a consistent and simple interface: 

##### _Estimators_ 

Any object that can estimate some parameters based on a dataset is called an _estimator_ (e.g., a `SimpleImputer` is an estimator). The estimation itself is performed by the `fit()` method, and it takes a dataset as a parameter, or two for supervised learning algorithms—the second dataset contains the labels. Any other parameter needed to guide the estimation process is considered a hyperparameter (such as a `SimpleImputer` ’s `strategy` ), and it must be set as an instance variable (generally via a constructor parameter). 

##### _Transformers_ 

Some estimators (such as a `SimpleImputer` ) can also transform a dataset; these are called _transformers_ . Once again, the API is simple: the transforma‐ tion is performed by the `transform()` method with the dataset to transform as a parameter. It returns the transformed dataset. This transformation gen‐ erally relies on the learned parameters, as is the case for a `SimpleImputer` . All transformers also have a convenience method called `fit_transform()` , which is equivalent to calling `fit()` and then `transform()` (but sometimes `fit_transform()` is optimized and runs much faster). 

##### _Predictors_ 

Finally, some estimators, given a dataset, are capable of making predictions; they are called _predictors_ . For example, the `LinearRegression` model in the previous chapter was a predictor: given a country’s GDP per capita, it predicted life satisfaction. A predictor has a `predict()` method that takes a dataset of new instances and returns a dataset of corresponding predictions. It also has a `score()` method that measures the quality of the predictions, given a test set (and the corresponding labels, in the case of supervised learning algorithms).<sup>10</sup> 

##### _Inspection_ 

All the estimator’s hyperparameters are accessible directly via public instance variables (e.g., `imputer.strategy` ), and all the estimator’s learned parameters are accessible via public instance variables with an underscore suffix (e.g., `imputer.statistics_` ). 

9 For more details on the design principles, see Lars Buitinck et al., “API Design for Machine Learning Software: Experiences from the Scikit-Learn Project”, arXiv preprint arXiv:1309.0238 (2013). 

10 Some predictors also provide methods to measure the confidence of their predictions. 

###### **70 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

##### _Nonproliferation of classes_ 

Datasets are represented as NumPy arrays or SciPy sparse matrices, instead of homemade classes. Hyperparameters are just regular Python strings or numbers. 

##### _Composition_ 

Existing building blocks are reused as much as possible. For example, it is easy to create a `Pipeline` estimator from an arbitrary sequence of transformers followed by a final estimator, as you will see. 

##### _Sensible defaults_ 

Scikit-Learn provides reasonable default values for most parameters, making it easy to quickly create a baseline working system. 

Scikit-Learn transformers output NumPy arrays (or sometimes SciPy sparse matri‐ ces) even when they are fed Pandas DataFrames as input.<sup>11</sup> So, the output of `imputer.transform(housing_num)` is a NumPy array: `X` has neither column names nor index. Luckily, it’s not too hard to wrap `X` in a DataFrame and recover the column names and index from `housing_num` : 

```
housing_tr=pd.DataFrame(X, columns=housing_num.columns,
index=housing_num.index)
```

### **Handling Text and Categorical Attributes** 

So far we have only dealt with numerical attributes, but your data may also contain text attributes. In this dataset, there is just one: the `ocean_proximity` attribute. Let’s look at its value for the first few instances: 

```
>>> housing_cat=housing[["ocean_proximity"]]
>>> housing_cat.head(8)
      ocean_proximity
13096        NEAR BAY
14973       <1H OCEAN
3785           INLAND
14689          INLAND
20507      NEAR OCEAN
1286           INLAND
18078       <1H OCEAN
4396         NEAR BAY
```

It’s not arbitrary text: there are a limited number of possible values, each of which represents a category. So this attribute is a categorical attribute. Most machine learning algorithms prefer to work with numbers, so let’s convert these categories from text to numbers. For this, we can use Scikit-Learn’s `OrdinalEncoder` class: 

> 11 If you run `sklearn.set_config(transform_output="pandas")` , all transformers will output Pandas Data‐ Frames when they receive a DataFrame as input: Pandas in, Pandas out. 

**Prepare the Data for Machine Learning Algorithms | 71** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

```
fromsklearn.preprocessingimportOrdinalEncoder
```

```
ordinal_encoder=OrdinalEncoder()
housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)
```

Here’s what the first few encoded values in `housing_cat_encoded` look like: 

```
>>> housing_cat_encoded[:8]
array([[3.],
       [0.],
       [1.],
       [1.],
       [4.],
       [1.],
       [0.],
       [3.]])
```

You can get the list of categories using the `categories_` instance variable. It is a list containing a 1D array of categories for each categorical attribute (in this case, a list containing a single array since there is just one categorical attribute): 

```
>>> ordinal_encoder.categories_
[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
       dtype=object)]
```

One issue with this representation is that ML algorithms will assume that two nearby values are more similar than two distant values. This may be fine in some cases (e.g., for ordered categories such as “bad”, “average”, “good”, and “excellent”), but it is obviously not the case for the `ocean_proximity` column (for example, categories 0 and 4 are clearly more similar than categories 0 and 1). To fix this issue, a common solution is to create one binary attribute per category: one attribute equal to 1 when the category is `"<1H OCEAN"` (and 0 otherwise), another attribute equal to 1 when the category is `"INLAND"` (and 0 otherwise), and so on. This is called _one-hot encoding_ , because only one attribute will be equal to 1 (hot), while the others will be 0 (cold). The new attributes are sometimes called _dummy_ attributes. Scikit-Learn provides a `OneHotEncoder` class to convert categorical values into one-hot vectors: 

```
fromsklearn.preprocessingimportOneHotEncoder
```

```
cat_encoder=OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
```

###### **72 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

By default, the output of a `OneHotEncoder` is a SciPy _sparse matrix_ , instead of a NumPy array: 

```
>>> housing_cat_1hot
<16512x5 sparse matrix of type '<class 'numpy.float64'>'
 with 16512 stored elements in Compressed Sparse Row format>
```

A sparse matrix is a very efficient representation for matrices that contain mostly zeros. Indeed, internally it only stores the nonzero values and their positions. When a categorical attribute has hundreds or thousands of categories, one-hot encoding it results in a very large matrix full of 0s except for a single 1 per row. In this case, a sparse matrix is exactly what you need: it will save plenty of memory and speed up computations. You can use a sparse matrix mostly like a normal 2D array,<sup>12</sup> but if you want to convert it to a (dense) NumPy array, just call the `toarray()` method: 

```
>>> housing_cat_1hot.toarray()
array([[0., 0., 0., 1., 0.],
       [1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       ...,
       [0., 0., 0., 0., 1.],
       [1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1.]])
```

Alternatively, you can set `sparse_output=False` when creating the `OneHotEncoder` , in which case the `transform()` method will return a regular (dense) NumPy array directly: 

```
cat_encoder=OneHotEncoder(sparse_output=False)
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)  # now a dense array
```

As with the `OrdinalEncoder` , you can get the list of categories using the encoder’s `categories_` instance variable: 

```
>>> cat_encoder.categories_
[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
       dtype=object)]
```

Pandas has a function called `get_dummies()` , which also converts each categorical feature into a one-hot representation, with one binary feature per category: 

```
>>> df_test=pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
>>> pd.get_dummies(df_test)
   ocean_proximity_INLAND  ocean_proximity_NEAR BAY
0                       1                         0
1                       0                         1
```

12 See SciPy’s documentation for more details. 

**Prepare the Data for Machine Learning Algorithms | 73** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

It looks nice and simple, so why not use it instead of `OneHotEncoder` ? Well, the advantage of `OneHotEncoder` is that it remembers which categories it was trained on. This is very important because once your model is in production, it should be fed exactly the same features as during training: no more, no less. Look what our trained `cat_encoder` outputs when we make it transform the same `df_test` (using `transform()` , not `fit_transform()` ): 

```
>>> cat_encoder.transform(df_test)
array([[0., 1., 0., 0., 0.],
       [0., 0., 0., 1., 0.]])
```

See the difference? `get_dummies()` saw only two categories, so it output two columns, whereas `OneHotEncoder` output one column per learned category, in the right order. Moreover, if you feed `get_dummies()` a DataFrame containing an unknown category (e.g., `"<2H OCEAN"` ), it will happily generate a column for it: 

```
>>> df_test_unknown=pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
>>> pd.get_dummies(df_test_unknown)
   ocean_proximity_<2H OCEAN  ocean_proximity_ISLAND
0                          1                       0
1                          0                       1
```

But `OneHotEncoder` is smarter: it will detect the unknown category and raise an exception. If you prefer, you can set the `handle_unknown` hyperparameter to `"ignore"` , in which case it will just represent the unknown category with zeros: 

```
>>> cat_encoder.handle_unknown="ignore"
>>> cat_encoder.transform(df_test_unknown)
array([[0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0.]])
```



If a categorical attribute has a large number of possible categories (e.g., country code, profession, species), then one-hot encoding will result in a large number of input features. This may slow down training and degrade performance. If this happens, you may want to replace the categorical input with useful numerical fea‐ tures related to the categories: for example, you could replace the `ocean_proximity` feature with the distance to the ocean (similarly, a country code could be replaced with the country’s population and GDP per capita). Alternatively, you can use one of the encoders provided by the `category_encoders` package on GitHub. Or, when dealing with neural networks, you can replace each category with a learnable, low-dimensional vector called an _embedding_ . This is an example of _representation learning_ (see Chapters 13 and 17 for more details). 

**74 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

When you fit any Scikit-Learn estimator using a DataFrame, the estimator stores the column names in the `feature_names_in_` attribute. Scikit-Learn then ensures that any DataFrame fed to this estimator after that (e.g., to `transform()` or `predict()` ) has the same column names. Transformers also provide a `get_feature_names_out()` method that you can use to build a DataFrame around the transformer’s output: 

```
>>> cat_encoder.feature_names_in_
array(['ocean_proximity'], dtype=object)
>>> cat_encoder.get_feature_names_out()
array(['ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
       'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
       'ocean_proximity_NEAR OCEAN'], dtype=object)
>>> df_output=pd.DataFrame(cat_encoder.transform(df_test_unknown),
... columns=cat_encoder.get_feature_names_out(),
... index=df_test_unknown.index)
...
```

### **Feature Scaling and Transformation** 

One of the most important transformations you need to apply to your data is _feature scaling_ . With few exceptions, machine learning algorithms don’t perform well when the input numerical attributes have very different scales. This is the case for the housing data: the total number of rooms ranges from about 6 to 39,320, while the median incomes only range from 0 to 15. Without any scaling, most models will be biased toward ignoring the median income and focusing more on the number of rooms. 

There are two common ways to get all attributes to have the same scale: _min-max scaling_ and _standardization_ . 



As with all estimators, it is important to fit the scalers to the train‐ ing data only: never use `fit()` or `fit_transform()` for anything else than the training set. Once you have a trained scaler, you can then use it to `transform()` any other set, including the validation set, the test set, and new data. Note that while the training set values will always be scaled to the specified range, if new data contains outliers, these may end up scaled outside the range. If you want to avoid this, just set the `clip` hyperparameter to `True` . 

**Prepare the Data for Machine Learning Algorithms | 75** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

Min-max scaling (many people call this _normalization_ ) is the simplest: for each attribute, the values are shifted and rescaled so that they end up ranging from 0 to 1. This is performed by subtracting the min value and dividing by the dif‐ ference between the min and the max. Scikit-Learn provides a transformer called `MinMaxScaler` for this. It has a `feature_range` hyperparameter that lets you change the range if, for some reason, you don’t want 0–1 (e.g., neural networks work best with zero-mean inputs, so a range of –1 to 1 is preferable). It’s quite easy to use: 

```
fromsklearn.preprocessingimportMinMaxScaler
```

```
min_max_scaler=MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled=min_max_scaler.fit_transform(housing_num)
```

Standardization is different: first it subtracts the mean value (so standardized values have a zero mean), then it divides the result by the standard deviation (so standard‐ ized values have a standard deviation equal to 1). Unlike min-max scaling, standardi‐ zation does not restrict values to a specific range. However, standardization is much less affected by outliers. For example, suppose a district has a median income equal to 100 (by mistake), instead of the usual 0–15. Min-max scaling to the 0–1 range would map this outlier down to 1 and it would crush all the other values down to 0–0.15, whereas standardization would not be much affected. Scikit-Learn provides a transformer called `StandardScaler` for standardization: 

```
fromsklearn.preprocessingimportStandardScaler
```

```
std_scaler=StandardScaler()
housing_num_std_scaled=std_scaler.fit_transform(housing_num)
```



If you want to scale a sparse matrix without converting it to a dense matrix first, you can use a `StandardScaler` with its `with_mean` hyperparameter set to `False` : it will only divide the data by the standard deviation, without subtracting the mean (as this would break sparsity). 

When a feature’s distribution has a _heavy tail_ (i.e., when values far from the mean are not exponentially rare), both min-max scaling and standardization will squash most values into a small range. Machine learning models generally don’t like this at all, as you will see in Chapter 4. So _before_ you scale the feature, you should first transform it to shrink the heavy tail, and if possible to make the distribution roughly symmetrical. For example, a common way to do this for positive features with a heavy tail to the right is to replace the feature with its square root (or raise the feature to a power between 0 and 1). If the feature has a really long and heavy tail, such as a _power law distribution_ , then replacing the feature with its logarithm may help. For example, the `population` feature roughly follows a power law: districts with 10,000 inhabitants are only 10 times less frequent than districts with 1,000 inhabitants, not 

**76 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 



<!-- Start of picture text -->
22¥5Yn= 3000<br>To<br>4“ 2000<br>°<br>2ree<br>€ 1000<br>=}<br>2<br>0<br>0 2500 5000 7500 10000 12500 15000 2 4 6 8 10<br>Population Log of population<br><!-- End of picture text -->



<!-- Start of picture text -->
1.0<br>10007 —— gamma = 0.10 AX,<br>---- gamma = 0.03 if \\<br>2 HT VN 0.8<br>% 800 ; \ .<br>— I 1 0.6 ©<br>2 600 i \ =<br>S) i \ £<br>o i \ 0.4 0<br>Q 400 ! \ oD<br>£ \ <x<br>5<br>2 0.2<br>200<br>0.0<br>0<br>0 10 20 30 40 50<br>Housing median age<br><!-- End of picture text -->

`StandardScaler` expects 2D inputs. Also, in this example we just train the model on a single raw input feature (median income), for simplicity: 

```
fromsklearn.linear_modelimportLinearRegression
```

```
target_scaler=StandardScaler()
scaled_labels=target_scaler.fit_transform(housing_labels.to_frame())
```

```
model=LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data=housing[["median_income"]].iloc[:5]  # pretend this is new data
```

```
scaled_predictions=model.predict(some_new_data)
predictions=target_scaler.inverse_transform(scaled_predictions)
```

This works fine, but a simpler option is to use a `TransformedTargetRegressor` . We just need to construct it, giving it the regression model and the label transformer, then fit it on the training set, using the original unscaled labels. It will automatically use the transformer to scale the labels and train the regression model on the resulting scaled labels, just like we did previously. Then, when we want to make a prediction, it will call the regression model’s `predict()` method and use the scaler’s `inverse_trans form()` method to produce the prediction: 

```
fromsklearn.composeimportTransformedTargetRegressor
```

```
model=TransformedTargetRegressor(LinearRegression(),
transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions=model.predict(some_new_data)
```

### **Custom Transformers** 

Although Scikit-Learn provides many useful transformers, you will need to write your own for tasks such as custom transformations, cleanup operations, or combin‐ ing specific attributes. 

For transformations that don’t require any training, you can just write a function that takes a NumPy array as input and outputs the transformed array. For example, as discussed in the previous section, it’s often a good idea to transform features with heavy-tailed distributions by replacing them with their logarithm (assuming the feature is positive and the tail is on the right). Let’s create a log-transformer and apply it to the `population` feature: 

```
fromsklearn.preprocessingimportFunctionTransformer
```

```
log_transformer=FunctionTransformer(np.log, inverse_func=np.exp)
log_pop=log_transformer.transform(housing[["population"]])
```

**Prepare the Data for Machine Learning Algorithms | 79** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

The `inverse_func` argument is optional. It lets you specify an inverse transform function, e.g., if you plan to use your transformer in a `TransformedTargetRegressor` . 

Your transformation function can take hyperparameters as additional arguments. For example, here’s how to create a transformer that computes the same Gaussian RBF similarity measure as earlier: 

```
rbf_transformer=FunctionTransformer(rbf_kernel,
```

```
kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35=rbf_transformer.transform(housing[["housing_median_age"]])
```

Note that there’s no inverse function for the RBF kernel, since there are always two values at a given distance from a fixed point (except at distance 0). Also note that `rbf_kernel()` does not treat the features separately. If you pass it an array with two features, it will measure the 2D distance (Euclidean) to measure similarity. For example, here’s how to add a feature that will measure the geographic similarity between each district and San Francisco: 

```
sf_coords=37.7749, -122.41
sf_transformer=FunctionTransformer(rbf_kernel,
```

```
kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil=sf_transformer.transform(housing[["latitude", "longitude"]])
```

Custom transformers are also useful to combine features. For example, here’s a `FunctionTransformer` that computes the ratio between the input features 0 and 1: 

```
>>> ratio_transformer=FunctionTransformer(lambdaX: X[:, [0]] /X[:, [1]])
>>> ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))
array([[0.5 ],
```

```
       [0.75]])
```

`FunctionTransformer` is very handy, but what if you would like your transformer to be trainable, learning some parameters in the `fit()` method and using them later in the `transform()` method? For this, you need to write a custom class. Scikit-Learn relies on duck typing, so this class does not have to inherit from any particular base class. All it needs is three methods: `fit()` (which must return `self` ), `transform()` , and `fit_transform()` . 

You can get `fit_transform()` for free by simply adding `TransformerMixin` as a base class: the default implementation will just call `fit()` and then `transform()` . If you add `BaseEstimator` as a base class (and avoid using `*args` and `**kwargs` in your constructor), you will also get two extra methods: `get_params()` and `set_params()` . These will be useful for automatic hyperparameter tuning. 

For example, here’s a custom transformer that acts much like the `StandardScaler` : 

**80 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

```
fromsklearn.baseimportBaseEstimator, TransformerMixin
fromsklearn.utils.validationimportcheck_array, check_is_fitted
```

```
classStandardScalerClone(BaseEstimator, TransformerMixin):
def __init__(self, with_mean=True):  # no *args or **kwargs!
self.with_mean=with_mean
```

```
deffit(self, X, y=None):  # y is required even though we don't use it
X=check_array(X)  # checks that X is an array with finite float values
self.mean_=X.mean(axis=0)
self.scale_=X.std(axis=0)
self.n_features_in_=X.shape[1]  # every estimator stores this in fit()
returnself# always return self!
```

```
deftransform(self, X):
check_is_fitted(self)  # looks for learned attributes (with trailing _)
X=check_array(X)
assertself.n_features_in_==X.shape[1]
ifself.with_mean:
X=X-self.mean_
returnX/self.scale_
```

Here are a few things to note: 

- The `sklearn.utils.validation` package contains several functions we can use to validate the inputs. For simplicity, we will skip such tests in the rest of this book, but production code should have them. 

- Scikit-Learn pipelines require the `fit()` method to have two arguments `X` and `y` , which is why we need the `y=None` argument even though we don’t use `y` . 

- All Scikit-Learn estimators set `n_features_in_` in the `fit()` method, and they ensure that the data passed to `transform()` or `predict()` has this number of features. 

- The `fit()` method must return `self` . 

- This implementation is not 100% complete: all estimators should set `feature_names_in_` in the `fit()` method when they are passed a DataFrame. Moreover, all transformers should provide a `get_feature_names_out()` method, as well as an `inverse_transform()` method when their transformation can be reversed. See the last exercise at the end of this chapter for more details. 

A custom transformer can (and often does) use other estimators in its implementa‐ tion. For example, the following code demonstrates custom transformer that uses a `KMeans` clusterer in the `fit()` method to identify the main clusters in the training data, and then uses `rbf_kernel()` in the `transform()` method to measure how similar each sample is to each cluster center: 

**Prepare the Data for Machine Learning Algorithms | 81** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

```
fromsklearn.clusterimportKMeans
classClusterSimilarity(BaseEstimator, TransformerMixin):
def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
self.n_clusters=n_clusters
self.gamma=gamma
self.random_state=random_state
deffit(self, X, y=None, sample_weight=None):
self.kmeans_=KMeans(self.n_clusters, random_state=self.random_state)
self.kmeans_.fit(X, sample_weight=sample_weight)
returnself# always return self!
deftransform(self, X):
returnrbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
defget_feature_names_out(self, names=None):
return [f"Cluster {i} similarity"foriinrange(self.n_clusters)]
```



You can check whether your custom estimator respects ScikitLearn’s API by passing an instance to `check_estimator()` from the `sklearn.utils.estimator_checks` package. For the full API, check out _https://scikit-learn.org/stable/developers_ . 

As you will see in Chapter 9, _k_ -means is a clustering algorithm that locates clusters in the data. How many it searches for is controlled by the `n_clusters` hyperparameter. After training, the cluster centers are available via the `cluster_centers_` attribute. The `fit()` method of `KMeans` supports an optional argument `sample_weight` , which lets the user specify the relative weights of the samples. _k_ -means is a stochastic algorithm, meaning that it relies on randomness to locate the clusters, so if you want reproducible results, you must set the `random_state` parameter. As you can see, despite the complexity of the task, the code is fairly straightforward. Now let’s use this custom transformer: 

```
cluster_simil=ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities=cluster_simil.fit_transform(housing[["latitude", "longitude"]],
sample_weight=housing_labels)
```

This code creates a `ClusterSimilarity` transformer, setting the number of clusters to 10. Then it calls `fit_transform()` with the latitude and longitude of every district in the training set, weighting each district by its median house value. The transformer uses _k_ -means to locate the clusters, then measures the Gaussian RBF similarity between each district and all 10 cluster centers. The result is a matrix with one row per district, and one column per cluster. Let’s look at the first three rows, rounding to two decimal places: 

###### **82 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 



<!-- Start of picture text -->
2 a Cluster centers<br>ee, @ Population<br>40 \e° e of,° + ° 0.8<br>eeoan :<br>. ve<br>©. ° *<br>a) se e ae. ¥ 0.6 co&<br>se} : £<br>2 dil a<br>wv > a vya<br>g : 3<br>“ee So<br>0.4<br>36 ry a<br>“@e =<br>oy, |<br>34 a “7 0.2<br>-124 -122 -120 -118 —116 -114<br>Longitude<br><!-- End of picture text -->

The `Pipeline` constructor takes a list of name/estimator pairs (2-tuples) defining a sequence of steps. The names can be anything you like, as long as they are unique and don’t contain double underscores ( `__` ). They will be useful later, when we discuss hyperparameter tuning. The estimators must all be transformers (i.e., they must have a `fit_transform()` method), except for the last one, which can be anything: a transformer, a predictor, or any other type of estimator. 



In a Jupyter notebook, if you `import sklearn` and run `sklearn. set_config(display="diagram")` , all Scikit-Learn estimators will be rendered as interactive diagrams. This is particularly useful for visualizing pipelines. To visualize `num_pipeline` , run a cell with `num_pipeline` as the last line. Clicking an estimator will show more details. 

If you don’t want to name the transformers, you can use the `make_pipeline()` func‐ tion instead; it takes transformers as positional arguments and creates a `Pipeline` using the names of the transformers’ classes, in lowercase and without underscores (e.g., `"simpleimputer"` ): 

```
fromsklearn.pipelineimportmake_pipeline
```

```
num_pipeline=make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
```

If multiple transformers have the same name, an index is appended to their names (e.g., `"foo-1"` , `"foo-2"` , etc.). 

When you call the pipeline’s `fit()` method, it calls `fit_transform()` sequentially on all the transformers, passing the output of each call as the parameter to the next call until it reaches the final estimator, for which it just calls the `fit()` method. 

The pipeline exposes the same methods as the final estimator. In this example the last estimator is a `StandardScaler` , which is a transformer, so the pipeline also acts like a transformer. If you call the pipeline’s `transform()` method, it will sequentially apply all the transformations to the data. If the last estimator were a predictor instead of a transformer, then the pipeline would have a `predict()` method rather than a `transform()` method. Calling it would sequentially apply all the transformations to the data and pass the result to the predictor’s `predict()` method. 

Let’s call the pipeline’s `fit_transform()` method and look at the output’s first two rows, rounded to two decimal places: 

```
>>> housing_num_prepared=num_pipeline.fit_transform(housing_num)
>>> housing_num_prepared[:2].round(2)
array([[-1.42,  1.01,  1.86,  0.31,  1.37,  0.14,  1.39, -0.94],
       [ 0.6 , -0.7 ,  0.91, -0.31, -0.44, -0.69, -0.37,  1.17]])
```

**84 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

As you saw earlier, if you want to recover a nice DataFrame, you can use the pipeline’s `get_feature_names_out()` method: 

```
df_housing_num_prepared=pd.DataFrame(
```

```
housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
index=housing_num.index)
```

Pipelines support indexing; for example, `pipeline[1]` returns the second estimator in the pipeline, and `pipeline[:-1]` returns a `Pipeline` object containing all but the last estimator. You can also access the estimators via the `steps` attribute, which is a list of name/estimator pairs, or via the `named_steps` dictionary attribute, which maps the names to the estimators. For example, `num_pipeline["simpleimputer"]` returns the estimator named `"simpleimputer"` . 

So far, we have handled the categorical columns and the numerical columns sepa‐ rately. It would be more convenient to have a single transformer capable of handling all columns, applying the appropriate transformations to each column. For this, you can use a `ColumnTransformer` . For example, the following `ColumnTransformer` will apply `num_pipeline` (the one we just defined) to the numerical attributes and `cat_pipeline` to the categorical attribute: 

```
fromsklearn.composeimportColumnTransformer
```

```
num_attribs= ["longitude", "latitude", "housing_median_age", "total_rooms",
"total_bedrooms", "population", "households", "median_income"]
cat_attribs= ["ocean_proximity"]
```

```
cat_pipeline=make_pipeline(
SimpleImputer(strategy="most_frequent"),
OneHotEncoder(handle_unknown="ignore"))
```

```
preprocessing=ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
```

First we import the `ColumnTransformer` class, then we define the list of numeri‐ cal and categorical column names and construct a simple pipeline for categorical attributes. Lastly, we construct a `ColumnTransformer` . Its constructor requires a list of triplets (3-tuples), each containing a name (which must be unique and not contain double underscores), a transformer, and a list of names (or indices) of columns that the transformer should be applied to. 

**Prepare the Data for Machine Learning Algorithms | 85** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 



<!-- Start of picture text -->
™<br><!-- End of picture text -->

™ 

& 



<!-- Start of picture text -->
&<br><!-- End of picture text -->

- Missing values in numerical features will be imputed by replacing them with the median, as most ML algorithms don’t expect missing values. In categorical features, missing values will be replaced by the most frequent category. 

- The categorical feature will be one-hot encoded, as most ML algorithms only accept numerical inputs. 

- A few ratio features will be computed and added: `bedrooms_ratio` , `rooms_per_house` , and `people_per_house` . Hopefully these will better correlate with the median house value, and thereby help the ML models. 

- A few cluster similarity features will also be added. These will likely be more useful to the model than latitude and longitude. 

- Features with a long tail will be replaced by their logarithm, as most models prefer features with roughly uniform or Gaussian distributions. 

- All numerical features will be standardized, as most ML algorithms prefer when all features have roughly the same scale. 

The code that builds the pipeline to do all of this should look familiar to you by now: 

```
defcolumn_ratio(X):
returnX[:, [0]] /X[:, [1]]
defratio_name(function_transformer, feature_names_in):
return ["ratio"]  # feature names out
defratio_pipeline():
returnmake_pipeline(
SimpleImputer(strategy="median"),
FunctionTransformer(column_ratio, feature_names_out=ratio_name),
StandardScaler())
```

```
log_pipeline=make_pipeline(
SimpleImputer(strategy="median"),
FunctionTransformer(np.log, feature_names_out="one-to-one"),
StandardScaler())
cluster_simil=ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline=make_pipeline(SimpleImputer(strategy="median"),
StandardScaler())
preprocessing=ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
"households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
remainder=default_num_pipeline)  # one column remaining: housing_median_age
```

**Prepare the Data for Machine Learning Algorithms | 87** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

If you run this `ColumnTransformer` , it performs all the transformations and outputs a NumPy array with 24 features: 

```
>>> housing_prepared=preprocessing.fit_transform(housing)
>>> housing_prepared.shape
(16512, 24)
>>> preprocessing.get_feature_names_out()
array(['bedrooms__ratio', 'rooms_per_house__ratio',
       'people_per_house__ratio', 'log__total_bedrooms',
       'log__total_rooms', 'log__population', 'log__households',
       'log__median_income', 'geo__Cluster 0 similarity', [...],
       'geo__Cluster 9 similarity', 'cat__ocean_proximity_<1H OCEAN',
       'cat__ocean_proximity_INLAND', 'cat__ocean_proximity_ISLAND',
       'cat__ocean_proximity_NEAR BAY', 'cat__ocean_proximity_NEAR OCEAN',
       'remainder__housing_median_age'], dtype=object)
```

## **Select and Train a Model** 

At last! You framed the problem, you got the data and explored it, you sampled a training set and a test set, and you wrote a preprocessing pipeline to automatically clean up and prepare your data for machine learning algorithms. You are now ready to select and train a machine learning model. 

### **Train and Evaluate on the Training Set** 

The good news is that thanks to all these previous steps, things are now going to be easy! You decide to train a very basic linear regression model to get started: 

```
fromsklearn.linear_modelimportLinearRegression
```

```
lin_reg=make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
```

Done! You now have a working linear regression model. You try it out on the training set, looking at the first five predictions and comparing them to the labels: 

```
>>> housing_predictions=lin_reg.predict(housing)
>>> housing_predictions[:5].round(-2)  # -2 = rounded to the nearest hundred
array([243700., 372400., 128800.,  94400., 328300.])
>>> housing_labels.iloc[:5].values
array([458300., 483800., 101700.,  96100., 361800.])
```

Well, it works, but not always: the first prediction is way off (by over $200,000!), while the other predictions are better: two are off by about 25%, and two are off by less than 10%. Remember that you chose to use the RMSE as your performance measure, so you want to measure this regression model’s RMSE on the whole training set using Scikit-Learn’s `root_mean_squared_error()` function: 

**88 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

```
>>> fromsklearn.metricsimportroot_mean_squared_error
>>> lin_rmse=root_mean_squared_error(housing_labels, housing_predictions)
>>> lin_rmse
68687.89176589991
```

This is better than nothing, but clearly not a great score: the `median_housing_values` of most districts range between $120,000 and $265,000, so a typical prediction error of $68,628 is really not very satisfying. This is an example of a model underfitting the training data. When this happens it can mean that the features do not provide enough information to make good predictions, or that the model is not powerful enough. As we saw in the previous chapter, the main ways to fix underfitting are to select a more powerful model, to feed the training algorithm with better features, or to reduce the constraints on the model. This model is not regularized, which rules out the last option. You could try to add more features, but first you want to try a more complex model to see how it does. 

You decide to try a `DecisionTreeRegressor` , as this is a fairly powerful model capable of finding complex nonlinear relationships in the data (decision trees are presented in more detail in Chapter 6): 

```
fromsklearn.treeimportDecisionTreeRegressor
```

```
tree_reg=make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
```

Now that the model is trained, you evaluate it on the training set: 

```
>>> housing_predictions=tree_reg.predict(housing)
>>> tree_rmse=root_mean_squared_error(housing_labels, housing_predictions)
>>> tree_rmse
0.0
```

Wait, what!? No error at all? Could this model really be absolutely perfect? Of course, it is much more likely that the model has badly overfit the data. How can you be sure? As you saw earlier, you don’t want to touch the test set until you are ready to launch a model you are confident about, so you need to use part of the training set for training and part of it for model validation. 

### **Better Evaluation Using Cross-Validation** 

One way to evaluate the decision tree model would be to use the `train_ test_split()` function to split the training set into a smaller training set and a validation set, then train your models against the smaller training set and evaluate them against the validation set. It’s a bit of effort, but nothing too difficult, and it would work fairly well. 

A great alternative is to use Scikit-Learn’s _k_-fold cross-validation_ feature. The follow‐ ing code randomly splits the training set into 10 nonoverlapping subsets called _folds_ , 

**Select and Train a Model | 89** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

then it trains and evaluates the decision tree model 10 times, picking a different fold for evaluation every time and using the other 9 folds for training. The result is an array containing the 10 evaluation scores: 

```
fromsklearn.model_selectionimportcross_val_score
```

```
tree_rmses=-cross_val_score(tree_reg, housing, housing_labels,
scoring="neg_root_mean_squared_error", cv=10)
```



Scikit-Learn’s cross-validation features expect a utility function (greater is better) rather than a cost function (lower is better), so the scoring function is actually the opposite of the RMSE. It’s a negative value, so you need to switch the sign of the output to get the RMSE scores. 

##### Let’s look at the results: 

```
>>> pd.Series(tree_rmses).describe()
count       10.000000
mean     66868.027288
std       2060.966425
min      63649.536493
25%      65338.078316
50%      66801.953094
75%      68229.934454
max      70094.778246
dtype: float64
```

Now the decision tree doesn’t look as good as it did earlier. In fact, it seems to perform almost as poorly as the linear regression model! Notice that cross-validation allows you to get not only an estimate of the performance of your model, but also a measure of how precise this estimate is (i.e., its standard deviation). The decision tree has an RMSE of about 66,868, with a standard deviation of about 2,061. You would not have this information if you just used one validation set. But cross-validation comes at the cost of training the model several times, so it is not always feasible. 

If you compute the same metric for the linear regression model, you will find that the mean RMSE is 69,858 and the standard deviation is 4,182. So the decision tree model seems to perform very slightly better than the linear model, but the difference is minimal due to severe overfitting. We know there’s an overfitting problem because the training error is low (actually zero) while the validation error is high. 

Let’s try one last model now: the `RandomForestRegressor` . As you will see in Chap‐ ter 7, random forests work by training many decision trees on random subsets of the features, then averaging out their predictions. Such models composed of many other models are called _ensembles_ : they are capable of boosting the performance of the underlying model (in this case, decision trees). The code is much the same as earlier: 

###### **90 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

```
fromsklearn.ensembleimportRandomForestRegressor
```

```
forest_reg=make_pipeline(preprocessing,
RandomForestRegressor(random_state=42))
forest_rmses=-cross_val_score(forest_reg, housing, housing_labels,
scoring="neg_root_mean_squared_error", cv=10)
```

##### Let’s look at the scores: 

```
>>> pd.Series(forest_rmses).describe()
count       10.000000
mean     47019.561281
std       1033.957120
min      45458.112527
25%      46464.031184
50%      46967.596354
75%      47325.694987
max      49243.765795
dtype: float64
```

Wow, this is much better: random forests really look very promising for this task! However, if you train a `RandomForestRegressor` and measure the RMSE on the training set, you will find roughly 17,474: that’s much lower, meaning that there’s still quite a lot of overfitting going on. Possible solutions are to simplify the model, constrain it (i.e., regularize it), or get a lot more training data. Before you dive much deeper into random forests, however, you should try out many other models from various categories of machine learning algorithms (e.g., several support vector machines with different kernels, and possibly a neural network), without spending too much time tweaking the hyperparameters. The goal is to shortlist a few (two to five) promising models. 

## **Fine-Tune Your Model** 

Let’s assume that you now have a shortlist of promising models. You now need to fine-tune them. Let’s look at a few ways you can do that. 

### **Grid Search** 

One option would be to fiddle with the hyperparameters manually, until you find a great combination of hyperparameter values. This would be very tedious work, and you may not have time to explore many combinations. 

Instead, you can use Scikit-Learn’s `GridSearchCV` class to search for you. All you need to do is tell it which hyperparameters you want it to experiment with and what values to try out, and it will use cross-validation to evaluate all the possible combinations of hyperparameter values. For example, the following code searches for the best combination of hyperparameter values for the `RandomForestRegressor` : 

**Fine-Tune Your Model | 91** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

```
fromsklearn.model_selectionimportGridSearchCV
```

```
full_pipeline=Pipeline([
```

```
    ("preprocessing", preprocessing),
```

```
    ("random_forest", RandomForestRegressor(random_state=42)),
```

```
])
param_grid= [
```

- `{'preprocessing__geo__n_clusters': [5, 8, 10],` 

- `'random_forest__max_features': [4, 6, 8]},` 

- `{'preprocessing__geo__n_clusters': [10, 15],` 

```
'random_forest__max_features': [6, 8, 10]},
]
grid_search=GridSearchCV(full_pipeline, param_grid, cv=3,
scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
```

Notice that you can refer to any hyperparameter of any estimator in a pipeline, even if this estimator is nested deep inside several pipelines and column transform‐ ers. For example, when Scikit-Learn sees `"preprocessing__geo__n_clusters"` , it splits this string at the double underscores, then it looks for an estimator named `"preprocessing"` in the pipeline and finds the preprocessing `ColumnTransformer` . Next, it looks for a transformer named `"geo"` inside this `ColumnTransformer` and finds the `ClusterSimilarity` transformer we used on the latitude and longitude attributes. Then it finds this transformer’s `n_clusters` hyperparameter. Similarly, `random_forest__max_features` refers to the `max_features` hyperparameter of the estimator named `"random_forest"` , which is of course the `RandomForestRegressor` model (the `max_features` hyperparameter will be explained in Chapter 7). 



Wrapping preprocessing steps in a Scikit-Learn pipeline allows you to tune the preprocessing hyperparameters along with the model hyperparameters. This is a good thing since they often interact. For example, perhaps increasing `n_clusters` requires increasing `max_features` as well. If fitting the pipeline transformers is compu‐ tationally expensive, you can set the pipeline’s `memory` parameter to the path of a caching directory: when you first fit the pipeline, Scikit-Learn will save the fitted transformers to this directory. If you then fit the pipeline again with the same hyperparameters, Scikit-Learn will just load the cached transformers. 

There are two dictionaries in this `param_grid` , so `GridSearchCV` will first evaluate all 3 × 3 = 9 combinations of `n_clusters` and `max_features` hyperparameter values specified in the first `dict` , then it will try all 2 × 3 = 6 combinations of hyperparame‐ ter values in the second `dict` . So in total the grid search will explore 9 + 6 = 15 combinations of hyperparameter values, and it will train the pipeline 3 times per combination, since we are using 3-fold cross validation. This means there will be a 

###### **92 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

grand total of 15 × 3 = 45 rounds of training! It may take a while, but when it is done you can get the best combination of parameters like this: 

```
>>> grid_search.best_params_
{'preprocessing__geo__n_clusters': 15, 'random_forest__max_features': 6}
```

In this example, the best model is obtained by setting `n_clusters` to 15 and setting `max_features` to 6. 



Since 15 is the maximum value that was evaluated for `n_clusters` , you should probably try searching again with higher values; the score may continue to improve. 

You can access the best estimator using `grid_search.best_estimator_` . If `GridSearchCV` is initialized with `refit=True` (which is the default), then once it finds the best estimator using cross-validation, it retrains it on the whole training set. This is usually a good idea, since feeding it more data will likely improve its performance. 

The evaluation scores are available using `grid_search.cv_results_` . This is a dictio‐ nary, but if you wrap it in a DataFrame you get a nice list of all the test scores for each combination of hyperparameters and for each cross-validation split, as well as the mean test score across all splits: 

```
>>> cv_res=pd.DataFrame(grid_search.cv_results_)
>>> cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
>>> [...]  # change column names to fit on this page, and show rmse = -score
>>> cv_res.head()  # note: the 1st column is the row ID
   n_clusters max_features  split0  split1  split2  mean_test_rmse
12         15            6   43460   43919   44748           44042
13         15            8   44132   44075   45010           44406
14         15           10   44374   44286   45316           44659
7          10            6   44683   44655   45657           44999
9          10            6   44683   44655   45657           44999
```

The mean test RMSE score for the best model is 44,042, which is better than the score you got earlier using the default hyperparameter values (which was 47,019). Congratulations, you have successfully fine-tuned your best model! 

### **Randomized Search** 

The grid search approach is fine when you are exploring relatively few combinations, like in the previous example, but `RandomizedSearchCV` is often preferable, especially when the hyperparameter search space is large. This class can be used in much the same way as the `GridSearchCV` class, but instead of trying out all possible combinations it evaluates a fixed number of combinations, selecting a random value 

**Fine-Tune Your Model | 93** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

for each hyperparameter at every iteration. This may sound surprising, but this approach has several benefits: 

- If some of your hyperparameters are continuous (or discrete but with many possible values), and you let randomized search run for, say, 1,000 iterations, then it will explore 1,000 different values for each of these hyperparameters, whereas grid search would only explore the few values you listed for each one. 

- Suppose a hyperparameter does not actually make much difference, but you don’t know it yet. If it has 10 possible values and you add it to your grid search, then training will take 10 times longer. But if you add it to a random search, it will not make any difference. 

- If there are 6 hyperparameters to explore, each with 10 possible values, then grid search offers no other choice than training the model a million times, whereas random search can always run for any number of iterations you choose. 

For each hyperparameter, you must provide either a list of possible values, or a probability distribution: 

```
fromsklearn.model_selectionimportRandomizedSearchCV
fromscipy.statsimportrandint
```

```
param_distribs= {'preprocessing__geo__n_clusters': randint(low=3, high=50),
'random_forest__max_features': randint(low=2, high=20)}
```

```
rnd_search=RandomizedSearchCV(
full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
scoring='neg_root_mean_squared_error', random_state=42)
```

```
rnd_search.fit(housing, housing_labels)
```

Scikit-Learn also has `HalvingRandomSearchCV` and `HalvingGridSearchCV` hyperpara‐ meter search classes. Their goal is to use the computational resources more efficiently, either to train faster or to explore a larger hyperparameter space. Here’s how they work: in the first round, many hyperparameter combinations (called “candidates”) are generated using either the grid approach or the random approach. These candidates are then used to train models that are evaluated using cross-validation, as usual. However, training uses limited resources, which speeds up this first round consider‐ ably. By default, “limited resources” means that the models are trained on a small part of the training set. However, other limitations are possible, such as reducing the number of training iterations if the model has a hyperparameter to set it. Once every candidate has been evaluated, only the best ones go on to the second round, where they are allowed more resources to compete. After several rounds, the final candidates are evaluated using full resources. This may save you some time tuning hyperparameters. 

**94 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

### **Ensemble Methods** 

Another way to fine-tune your system is to try to combine the models that perform best. The group (or “ensemble”) will often perform better than the best individual model—just like random forests perform better than the individual decision trees they rely on—especially if the individual models make very different types of errors. For example, you could train and fine-tune a _k_ -nearest neighbors model, then create an ensemble model that just predicts the mean of the random forest prediction and that model’s prediction. We will cover this topic in more detail in Chapter 7. 

### **Analyzing the Best Models and Their Errors** 

You will often gain good insights on the problem by inspecting the best models. For example, the `RandomForestRegressor` can indicate the relative importance of each attribute for making accurate predictions: 

```
>>> final_model=rnd_search.best_estimator_# includes preprocessing
>>> feature_importances=final_model["random_forest"].feature_importances_
>>> feature_importances.round(2)
array([0.07, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.19, [...], 0.01])
```

Let’s sort these importance scores in descending order and display them next to their corresponding attribute names: 

```
>>> sorted(zip(feature_importances,
... final_model["preprocessing"].get_feature_names_out()),
... reverse=True)
...
[(0.18694559869103852, 'log__median_income'),
 (0.0748194905715524, 'cat__ocean_proximity_INLAND'),
 (0.06926417748515576, 'bedrooms__ratio'),
 (0.05446998753775219, 'rooms_per_house__ratio'),
 (0.05262301809680712, 'people_per_house__ratio'),
 (0.03819415873915732, 'geo__Cluster 0 similarity'),
 [...]
 (0.00015061247730531558, 'cat__ocean_proximity_NEAR BAY'),
 (7.301686597099842e-05, 'cat__ocean_proximity_ISLAND')]
```

With this information, you may want to try dropping some of the less useful features (e.g., apparently only one `ocean_proximity` category is really useful, so you could try dropping the others). 



The `sklearn.feature_selection.SelectFromModel` transformer can automatically drop the least useful features for you: when you fit it, it trains a model (typically a random forest), looks at its `fea ture_importances_` attribute, and selects the most useful features. Then when you call `transform()` , it drops the other features. 

**Fine-Tune Your Model | 95** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

You should also look at the specific errors that your system makes, then try to understand why it makes them and what could fix the problem: adding extra features or getting rid of uninformative ones, cleaning up outliers, etc. 

Now is also a good time to ensure that your model not only works well on average, but also on all categories of districts, whether they’re rural or urban, rich or poor, northern or southern, minority or not, etc. Creating subsets of your validation set for each category takes a bit of work, but it’s important: if your model performs poorly on a whole category of districts, then it should probably not be deployed until the issue is solved, or at least it should not be used to make predictions for that category, as it may do more harm than good. 

### **Evaluate Your System on the Test Set** 

After tweaking your models for a while, you eventually have a system that performs sufficiently well. You are ready to evaluate the final model on the test set. There is nothing special about this process; just get the predictors and the labels from your test set and run your `final_model` to transform the data and make predictions, then evaluate these predictions: 

```
X_test=strat_test_set.drop("median_house_value", axis=1)
y_test=strat_test_set["median_house_value"].copy()
```

```
final_predictions=final_model.predict(X_test)
```

```
final_rmse=root_mean_squared_error(y_test, final_predictions)
print(final_rmse)  # prints 41424.40026462184
```

In some cases, such a point estimate of the generalization error will not be quite enough to convince you to launch: what if it is just 0.1% better than the model cur‐ rently in production? You might want to have an idea of how precise this estimate is. For this, you can compute a 95% _confidence interval_ for the generalization error using `scipy.stats.bootstrap()` . You get a fairly large interval from 39,574 to 43,780, and your previous point estimate of 41,424 is roughly in the middle of it: 

```
fromscipyimportstats
```

```
defrmse(squared_errors):
returnnp.sqrt(np.mean(squared_errors))
```

```
confidence=0.95
squared_errors= (final_predictions-y_test) **2
boot_result=stats.bootstrap([squared_errors], rmse,
confidence_level=confidence, random_state=42)
rmse_lower, rmse_upper=boot_result.confidence_interval# (39,574 to 43,780)
```

###### **96 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

If you did a lot of hyperparameter tuning, the performance will usually be slightly worse than what you measured using cross-validation. That’s because your system ends up fine-tuned to perform well on the validation data and will likely not perform as well on unknown datasets. That’s not the case in this example since the test RMSE is lower than the validation RMSE, but when it happens you must resist the temptation to tweak the hyperparameters to make the numbers look good on the test set; the improvements would be unlikely to generalize to new data. 

Now comes the project prelaunch phase: you need to present your solution (high‐ lighting what you have learned, what worked and what did not, what assumptions were made, and what your system’s limitations are), document everything, and create nice presentations with clear visualizations and easy-to-remember statements (e.g., “the median income is the number one predictor of housing prices”). In this Califor‐ nia housing example, the final performance of the system is not much better than the experts’ price estimates, which were often off by 30%, but it may still be a good idea to launch it, especially if this frees up some time for the experts so they can work on more interesting and productive tasks. 

## **Launch, Monitor, and Maintain Your System** 

Perfect, you got approval to launch! You now need to get your solution ready for production (e.g., polish the code, write documentation and tests, and so on). Then you can deploy your model to your production environment. The most basic way to do this is just to save the best model you trained, transfer the file to your production environment, and load it. To save the model, you can use the `joblib` library like this: 

```
importjoblib
```

```
joblib.dump(final_model, "my_california_housing_model.pkl")
```



It’s often a good idea to save every model you experiment with so that you can come back easily to any model you want. You may also save the cross-validation scores and perhaps the actual predictions on the validation set. This will allow you to easily compare scores across model types, and compare the types of errors they make. 

Once your model is transferred to production, you can load it and use it. For this you must first import any custom classes and functions the model relies on (which means transferring the code to production), then load the model using `joblib` and use it to make predictions: 

**Launch, Monitor, and Maintain Your System | 97** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 



<!-- Start of picture text -->
1 —™‘ Inputs Web service<br>‘iC | webapp [|<br>to SY Predictions<br><!-- End of picture text -->

But deployment is not the end of the story. You also need to write monitoring code to check your system’s live performance at regular intervals and trigger alerts when it drops. It may drop very quickly, for example if a component breaks in your infrastructure, but be aware that it could also decay very slowly, which can easily go unnoticed for a long time. This is quite common because of model rot: if the model was trained with last year’s data, it may not be adapted to today’s data. 

So, you need to monitor your model’s live performance. But how do you do that? Well, it depends. In some cases, the model’s performance can be inferred from down‐ stream metrics. For example, if your model is part of a recommender system and it suggests products that the users may be interested in, then it’s easy to monitor the number of recommended products sold each day. If this number drops (compared to non-recommended products), then the prime suspect is the model. This may be because the data pipeline is broken, or perhaps the model needs to be retrained on fresh data (as we will discuss shortly). 

However, you may also need human analysis to assess the model’s performance. For example, suppose you trained an image classification model (we’ll look at these in Chapter 3) to detect various product defects on a production line. How can you get an alert if the model’s performance drops, before thousands of defective products get shipped to your clients? One solution is to send to human raters a sample of all the pictures that the model classified (especially pictures that the model wasn’t so sure about). Depending on the task, the raters may need to be experts, or they could be nonspecialists, such as workers on a crowdsourcing platform (e.g., Amazon Mechanical Turk). In some applications they could even be the users themselves, responding, for example, via surveys or repurposed captchas.<sup>14</sup> 

Either way, you need to put in place a monitoring system (with or without human raters to evaluate the live model), as well as all the relevant processes to define what to do in case of failures and how to prepare for them. Unfortunately, this can be a lot of work. In fact, it is often much more work than building and training a model. 

If the data keeps evolving, you will need to update your datasets and retrain your model regularly. You should probably automate the whole process as much as possi‐ ble. Here are a few things you can automate: 

- Collect fresh data regularly and label it (e.g., using human raters). 

- Write a script to train the model and fine-tune the hyperparameters automati‐ cally. This script could run automatically, for example every day or every week, depending on your needs. 

> 14 A captcha is a test to ensure a user is not a robot. These tests have often been used as a cheap way to label training data. 

**Launch, Monitor, and Maintain Your System | 99** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

- Write another script that will evaluate both the new model and the previous model on the updated test set, and deploy the model to production if the perfor‐ mance has not decreased (if it did, make sure you investigate why). The script should probably test the performance of your model on various subsets of the test set, such as poor or rich districts, rural or urban districts, etc. 

You should also make sure you evaluate the model’s input data quality. Sometimes performance will degrade slightly because of a poor-quality signal (e.g., a malfunc‐ tioning sensor sending random values, or another team’s output becoming stale), but it may take a while before your system’s performance degrades enough to trigger an alert. If you monitor your model’s inputs, you may catch this earlier. For example, you could trigger an alert if more and more inputs are missing a feature, or the mean or standard deviation drifts too far from the training set, or a categorical feature starts containing new categories. 

Finally, make sure you keep backups of every model you create and have the process and tools in place to roll back to a previous model quickly, in case the new model starts failing badly for some reason. Having backups also makes it possible to easily compare new models with previous ones. Similarly, you should keep backups of every version of your datasets so that you can roll back to a previous dataset if the new one ever gets corrupted (e.g., if the fresh data that gets added to it turns out to be full of outliers). Having backups of your datasets also allows you to evaluate any model against any previous dataset. 

As you can see, machine learning involves quite a lot of infrastructure. Chapter 19 discusses some aspects of this, but it’s a very broad topic called _ML Operations_ (MLOps), which deserves its own book. So don’t be surprised if your first ML project takes a lot of effort and time to build and deploy to production. Fortunately, once all the infrastructure is in place, going from idea to production will be much faster. 

## **Try It Out!** 

Hopefully this chapter gave you a good idea of what a machine learning project looks like as well as showing you some of the tools you can use to train a great system. As you can see, much of the work is in the data preparation step: building monitoring tools, setting up human evaluation pipelines, and automating regular model training. The machine learning algorithms are important, of course, but it is probably prefera‐ ble to be comfortable with the overall process and know three or four algorithms well rather than to spend all your time exploring advanced algorithms. 

So, if you have not already done so, now is a good time to pick up a laptop, select a dataset that you are interested in, and try to go through the whole process from A to Z. A good place to start is on a competition website such as Kaggle: you will have a dataset to play with, a clear goal, and people to share the experience with. Have fun! 

**100 | Chapter 2: End-to-End Machine Learning Project** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

## **Exercises** 

The following exercises are based on this chapter’s housing dataset: 

**1.** Try a support vector machine regressor ( `sklearn.svm.SVR` ) with various hyperparameters, such as `kernel="linear"` (with various values for the `C` hyperparameter) or `kernel="rbf"` (with various values for the `C` and `gamma` hyperparameters). Note that support vector machines don’t scale well to large datasets, so you should probably train your model on just the first 5,000 instances of the training set and use only 3-fold cross-validation, or else it will take hours. Don’t worry about what the hyperparameters mean for now; we’ll discuss them in Chapter 5. How does the best `SVR` predictor perform? 

**2.** Try replacing the `GridSearchCV` with a `RandomizedSearchCV` . 

**3.** Try adding a `SelectFromModel` transformer in the preparation pipeline to select only the most important attributes. 

**4.** Try creating a custom transformer that trains a _k_ -nearest neighbors regressor ( `sklearn.neighbors.KNeighborsRegressor` ) in its `fit()` method, and outputs the model’s predictions in its `transform()` method. Then add this feature to the preprocessing pipeline, using latitude and longitude as the inputs to this transformer. This will add a feature in the model that corresponds to the housing median price of the nearest districts. 

**5.** Automatically explore some preparation options using `RandomizedSearchCV` . 

**6.** Try to implement the `StandardScalerClone` class again from scratch, then add support for the `inverse_transform()` method: executing `scaler. inverse_transform(scaler.fit_transform(X))` should return an array very close to `X` . Then add support for feature names: set `feature_names_in_` in the `fit()` method if the input is a DataFrame. This attribute should be a NumPy array of column names. Lastly, implement the `get_feature_names_out()` method: it should have one optional `input_features=None` argument. If passed, the method should check that its length matches `n_features_in_` , and it should match `feature_names_in_` if it is defined; then `input_features` should be returned. If `input_features` is `None` , then the method should either return `feature_names_in_` if it is defined or `np.array(["x0", "x1", ...])` with length `n_features_in_` otherwise. 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

**Exercises | 101** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:54. 

