# kaggle-bnp
Some comments, links and scripts on bnp-cardif competition

This is a classification problem, with log loss measure

https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/details/evaluation

## Features

As seen on several blog post, there an huge ratio of missing values (NA).
Lots of dataviz on the scripts sections :
https://www.kaggle.com/jpmiller/bnp-paribas-cardif-claims-management/visualizing-the-nas

https://www.kaggle.com/bobcz3/bnp-paribas-cardif-claims-management/exploring-bnp-data-distributions/notebook

Obviously there are categorical and continous variables.
Some script take a subset of the variable.
One of the explanation could be that feature comes from 3 differents system. It can explain some similarity. Some people have tried to separate the features in 3 parts and predict independantly, but seems not working.


## Main algos

1. ks test approach + ensembling
see 
good perf.

2. Decision tree ensembling script using scikit-learn
 Random Forest + Extra Tree + Gradient Boosting (scikit learn implementation)
https://www.kaggle.com/trottefox/bnp-paribas-cardif-claims-management/blending-trees
this script gives also an imporante figure for this 3 algos.
 
3. Extra tree
 https://www.kaggle.com/chabir/bnp-paribas-cardif-claims-management/extratreesclassifier-score-0-45-v5/files
 
 With this 3 approachs + another linear combination , you can get a top 200 submission

## Neural network
There is a Tensorflow starter code, but not very interesting (only one layer).
https://www.kaggle.com/jstaker7/bnp-paribas-cardif-claims-management/tensorflow-starter
The best one seems a starter code using Lasagne / Theano.
It demands a lot of tuning.
https://www.kaggle.com/ouranos/bnp-paribas-cardif-claims-management/lasagne-nn
 
 Bruno
