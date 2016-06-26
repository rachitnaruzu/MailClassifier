Mail Classifier 
====================

This project aims to implement or study machine learning models and to use them to classify emails into 2 classes:  
Spam and Ham (Non-Spam)
  
**Dataset Used**: Enron Spam/Ham Mail data  
**Models Used**: Logistic Regression, Artificial Neural Network, Support Vector Machine

Logistic Regression, ANN have been implemented (both with regularization). **LIBSVM** and **LIBLINEAR** packages were used to study SVM nature with custom Gaussian Kernel (for non-linear SVM).

**alpha** = Learning rate  
**lambda** = Regularization Parameter  
**C** = Regularization Parameter for SVM  
**sigma** = Gaussian function parameter

All the Spam mails are inserted into mails/spam directory.  
All the Ham mails are inserted into mails/ham directory

**extract.py** contains modules to:

 - tokenize the text passed
 - uses various regex to filter and replace certain pattern
 - uses porter stemmer
 - extract top [400 to 500] list of most occurring words in all the
   mails in the data set, which will be used as a feature list for the
   training and testing data.

**create_data.py** uses extract.py to convert raw emails into proper format for learning models.  
**demo.py** runs and tests all these 3 models against the data generated by create_data.py

Result
------

    Logistic_Regression:
    
    Training begins...
    iterations: 1747
    time_taken: 129.986999989 sec
    accuracy: 96.8530368195
    
    Validation on test data:
    validation accuracy: 95.351113286
    
    Top 30 Spam Predictors:
    [Features]	[Feature Weights]
    money           1.32
    no              1.25
    net             1.20
    qualiti         1.18
    secur           1.10
    de              1.05
    life            1.02
    site            0.98
    million         0.96
    low             0.92
    remov           0.88
    offer           0.86
    account         0.85
    you             0.85
    med             0.85
    stop            0.82
    info            0.81
    health          0.80
    pill            0.79
    fund            0.79
    invest          0.77
    countri         0.76
    order           0.75
    link            0.75
    our             0.74
    your            0.71
    dollar          0.71
    bodi            0.71
    worldwid        0.70
    php             0.69
    
    
    Feed_Forward_ANN:
    
    Training begins...
    iterations: 3000
    time_taken: 59.4700000286 sec
    accuracy: 97.8600650372
    
    Validation on test data:
    validation accuracy: 95.5957915341
    
    Top 30 Spam Predictors:
    [Features]	[Feature Weights]
    net             9.98
    money           9.24
    pill            8.92
    de              8.78
    health          8.70
    qualiti         8.69
    life            7.81
    low             7.76
    fund            7.68
    million         7.16
    biz             7.13
    med             6.93
    no              6.83
    bodi            6.81
    site            6.67
    countri         6.61
    within          6.51
    lose            6.49
    ciali           6.40
    pro             6.32
    worldwid        6.26
    offer           6.24
    link            6.18
    stop            6.03
    info            6.00
    drug            5.97
    super           5.92
    china           5.90
    br              5.89
    remov           5.86
    
    
    Support Vector Machine (Linear):
    
    Training begins...
    time_taken: 0.354000091553 sec
    accuracy: 97.9125144236
    
    Validation on test data:
    validation accuracy: 95.4979202349
    
    Top 30 Spam Predictors:
    [Features]	[Feature Weights]
    photoshop       2.36
    biz             1.66
    pro             1.55
    pill            1.47
    cs              1.40
    xp              1.39
    xanax           1.39
    valium          1.39
    font            1.32
    health          1.29
    paliourg        1.20
    spam            1.13
    ciali           1.12
    melissa         1.07
    fund            1.07
    br              1.06
    net             1.05
    meyer           1.05
    china           1.01
    prescript       1.01
    med             1.00
    tr              1.00
    worldwid        0.98
    qualiti         0.91
    viagra          0.89
    increas         0.89
    drug            0.85
    super           0.84
    lloyd           0.82
    money           0.81
    
    
    Support Vector Machine (Gaussian Model):
    
    Training begins...
    time_taken: 12.381000042 sec
    accuracy: 98.4999475506
    
    Validation on test data:
    validation accuracy: 96.256422804
    
    
    [This SVM module of sklearn package does not provides the feature weights in case of non-linear SVM]

Observation 
-----------

 - From the predictor list of all these models we can see that the words
   like:  	**money, offer, no, million, qualiti, life, secur** are common
   top predictors, 	which, according to human intellectual deduction,
   also somewhat refers to Spam nature of mail to some extent.

 - SVM with Gaussian kernel performed best among all the models, in
   reference to both training and validation accuracies.
   
## Logistic Regression Observations

#### Accuracy vs Learning Rate

	![lr_ac_vs_al](/images/lr_ac_vs_al.JPG)
	
#### Accuracy vs Regularization

	![lr_ac_vs_lm](/images/lr_ac_vs_lm.JPG)
	
#### Iterations vs Learning Rate

	![lr_it_vs_al](/images/lr_it_vs_al.JPG)
	
#### Cost vs Iterations

	![lr_cost_vs_iter](/images/lr_cost_vs_iter.JPG)  

###  ANN Observations

#### Accuracy vs Learning Rate
	
	![ann_ac_vs_al](/images/ann_ac_vs_al.JPG)
	
#### Accuracy vs Regularization
	
	![ann_ac_vs_lm](/images/ann_ac_vs_lm.JPG)
	
#### Iterations vs Learning Rate

	![ann_it_vs_al](/images/ann_it_vs_al.JPG)
	
#### Cost vs Iterations

	![ann_cost_vs_it](/images/ann_cost_vs_it.JPG)

### Feature Selection

	![feature_selection](/images/feature_selection.JPG)

Dependencies
------------

 - Numpy - used by artificial_neural_network.py and logistic_regression.py for Matrix Calculation.  
 - Scipy - used by matplotlib  
 - nltk - contains porter stemmer  
 - sklearn - contains SVM  
 - matplotlib - used to plot iterations vs. cost curves  
 - re - for regex filter
