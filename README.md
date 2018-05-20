# MachineLearningProject
This project requires this sklearn library. 
The project involves taking a training dataset consisting of 30,000 features and 
extracting at most 15 features which are the most significant (this was done by taking the pearson coefficient)
and then running a custom alorigthm.

The steps are as follows:

i) Read data and labels
ii) Overall Feature selection (Reduce from approx 30k to 2k) 
iii) 5-fold cross validation:
         a)Pearson coefficients calculated
         b)Four classifiers used (worth 1 "count" each)
         c)Accuracy depends on the sum of the value counts by each classifier:
             ex. svm predicts 0, logistic_regression predicts 0
                 gaussian_nearest_means predicts 0 and nearest_centroid predicts 1
                
                 value = 0 + 0 + 0 + 1
                 if value <= 1 then classify in 0
                 if value >= 3 then classify in 1
                 else (if value = 2 or other) then classify as svm predicted
iv) Read test data and perform feature selection (features from train data) [extract 15 columns]
v) Output the num of features and the features themselves on console & save test labels 
      as a file named "sh486_testLabels"

This was a project I made for course CS675 [Machine Learning] at NJIT Fall Semester 2017.
