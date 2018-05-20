import sys
import time
import array
import copy
import math
from sklearn import svm
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid


# CS675 Project by Shereen Hameed sh486

# The code is as follows:
#   i) Read data and labels
#  ii) Overall Feature selection (Reduce from approx 30k to 2k) 
# iii) 5-fold cross validation:
#         a)Pearson coefficients calculated
#         b)Four classifiers used (worth 1 "count" each)
#         c)Accuracy depends on the sum of the value counts by each classifier:
#             ex. svm predicts 0, logistic_regression predicts 0
#                 gaussian_nearest_means predicts 0 and nearest_centroid predicts 1
                
#                 value = 0 + 0 + 0 + 1
#                 if value <= 1 then classify in 0
#                 if value >= 3 then classify in 1
#                 else (if value = 2 or other) then classify as svm predicted
# iv) Read test data and perform feature selection (features from train data) [extract 15 columns]
#  v) Output the num of features and the features themselves on console & save test labels 
#      as a file named "sh486_testLabels"
# THE CODE TYPICALLY TAKES 10 MINUTES TO RUN


start_time = time.time()

def extractCol(matrix, i):
    return [[row[i]] for row in matrix]

def mergeCol(a,b):
    return [x+y for x,y in zip(a,b)]

def PearsonCor(x, y, fi):   #x-data, y-labels
    sumX = 0
    sumX2 = 0
    ro = len(x)
    co = len(x[0])
    switch = 0
    pc = array.array("f")
    for i in range(0, co, 1):
        switch += 1
        sumY = 0
        sumY2 = 0
        sumXY = 0
        for j in range(0, ro, 1):
            if(switch == 1):
                sumX += y[j]
                sumX2 += y[j] ** 2
            sumY += x[j][i]
            sumY2 += x[j][i] ** 2
            sumXY += y[j] * x[j][i]  
        r = (ro*sumXY - sumX * sumY) / ((ro*sumX2 - (sumX**2))*(ro*sumY2 - (sumY**2)))**(0.5)
        pc.append(abs(r))

    savedToPrint = array.array("f")
    myFeatures = array.array("i")
    for i in range(0, fi, 1):
        selected = max(pc)
        # print(selected)
        savedToPrint.append(selected)
        featureIndex = pc.index(selected)
        # print(featureIndex)
        pc[featureIndex] = -1
        myFeatures.append(featureIndex)
    # print(savedToPrint)
    return myFeatures

def CreateDataSet(fea,dat):
    newData = extractCol(dat,fea[0])
    newLab = array.array("i")
    fea.remove(fea[0])
    length = len(fea)
    for i in range(0, length, 1):
        temp = extractCol(dat,fea[0])
        newData = mergeCol(newData,temp)
        fea.remove(fea[0])
    return newData

##################
# Read data
##################
datafile = sys.argv[1]
data = []
print("Starting to read data...")
with open(datafile,"r") as infile:
    for line in infile:
        temp = line.split()
        l = array.array("i")
        for i in temp:
            l.append(int(i))
        data.append(l)

####################
#Read labels
####################

labelfile = sys.argv[2]
trainlabels = array.array("i")
with open(labelfile,"r") as infile:
    for line in infile:
        temp = line.split()
        trainlabels.append(int(temp[0]))

print("Done reading data ",end="")

feat = 9

# print(data)
# print(trainlabels)

rows = len(data)
cols = len(data[0])
rowsl = len(trainlabels)

# print("rows= ",rows," cols= ",cols)
# print("rowsl= ",rowsl)
# print("Size of data: ",sys.getsizeof(data))
print("--- %s seconds" % (time.time() - start_time))

#Dimensionality Reduction
print("Starting Overall FS...")
neededFea = PearsonCor(data, trainlabels, 2000)

print("Done ",end="")

savedFea = copy.deepcopy(neededFea)

data1 = CreateDataSet(neededFea,data)

print("--- %s seconds" % (time.time() - start_time))

clf_svm = svm.SVC(gamma=0.001)
clf_log = linear_model.LogisticRegression()
clf_gnb = GaussianNB()
clf_nc = NearestCentroid()
    
allAccuracies = array.array("f")
allFeatures = []

accuracy_svm = 0
accuracy_score = 0
accuracy_log = 0
accuracy_gnb = 0
accuracy_nc = 0

my_accuracy = 0

iterations = 5
print("Cross validation iteration: ",end="")
for i in range(iterations):

    print(i)
	
    X_train, X_test, y_train, y_test = train_test_split(
        data1, trainlabels, test_size=0.2)

    newRows = len(X_train)
    newCols = len(X_train[0])
    newRowst = len(X_test)
    newColst = len(X_test[0])

    newRowsL = len(y_train)

    # print("newRows= ",newRows," newCols= ",newCols)
    # print("newRowst= ",newRowst," newColst= ",newColst)
    # print("newRowsL= ",newRowsL)

    
    PearFeatures = PearsonCor(X_train,y_train, feat)

    allFeatures.append(PearFeatures)
    argument = copy.deepcopy(PearFeatures)
    # print(i,"  Pearson Correlation Done - Train data")

    data_fea = CreateDataSet(argument,X_train)
    # print("New Data Made, rows= ",len(data_fea)," cols= ",len(data_fea[0]))

    clf_svm.fit(data_fea,y_train)
    clf_log.fit(data_fea,y_train)
    clf_gnb.fit(data_fea,y_train)
    clf_nc.fit(data_fea,y_train)


    TestFeatures = PearsonCor(X_test,y_test,feat)
    # print("   Pearson Correlation Done - Test data")

    test_fea = CreateDataSet(TestFeatures,X_test)
    # print("New Test Data Made, rows= ",len(test_fea)," cols= ",len(test_fea[0]))

    len_test_fea = len(test_fea)
    counter_svm = 0
    counter_log = 0
    counter_gnb = 0
    counter_nc = 0
    my_counter = 0
    for j in range(0,len_test_fea,1):
        predLab_svm = int(clf_svm.predict([test_fea[j]]))
        predLab_log = int(clf_log.predict([test_fea[j]]))
        predLab_gnb = int(clf_gnb.predict([test_fea[j]]))
        predLab_nc = int(clf_nc.predict([test_fea[j]]))
        h = predLab_svm + predLab_log + predLab_gnb +predLab_nc
        if(h >= 3):
            my_predLab = 1
        elif(h <= 1):
            my_predLab = 0
        else:
            my_predLab = predLab_svm
        if(my_predLab == y_test[j]):
            my_counter += 1
        if(predLab_svm == y_test[j]):
            counter_svm += 1
        if(predLab_log == y_test[j]):
            counter_log += 1
        if(predLab_gnb == y_test[j]):
            counter_gnb += 1
        if(predLab_nc == y_test[j]):
            counter_nc += 1

    # temp_score = float(clf_svm.score(test_fea,y_test))

    accuracy_svm += counter_svm/len_test_fea
    accuracy_log += counter_log/len_test_fea
    # accuracy_score += temp_score
    accuracy_gnb += counter_gnb/len_test_fea
    accuracy_nc += counter_nc/len_test_fea

    my_accuracy += my_counter/len_test_fea
    allAccuracies.append(my_counter/len_test_fea)

    # print(i," ",counter_svm/feat," ",counter_log/feat," ",counter_gnb/feat," ",counter_nc/feat)
    # print(" ",my_counter/feat)
print(" Done",end="") 
# print("Accuracy_svm: ",accuracy_svm/iterations)
# print("Accuracy_log: ",accuracy_log/iterations)
# print("Accuracy_gnb: ",accuracy_gnb/iterations)
# print("Accuracy_nc: ",accuracy_nc/iterations)
# print("Accuracy_score: ",accuracy_score/iterations)
# print("My Accuracy: ",my_accuracy/iterations)
print("--- %s seconds" % (time.time() - start_time))

# print("\nThese are my accuracies: ",allAccuracies)
bestAc = max(allAccuracies) 
bestInd = allAccuracies.index(bestAc)
bestFeatures = allFeatures[bestInd]

print("\nFeatures: ",feat)

originalFea = array.array("i")
for i in range(0,feat,1):
    realIndex = savedFea[bestFeatures[i]]
    originalFea.append(realIndex)

print("The features are: ",originalFea)

#####################################################
#Calculate Accuracy
#####################################################
argument1 = copy.deepcopy(originalFea)
AccData = CreateDataSet(argument1,data)

# print("rows of Acc= ",len(AccData)," cols= ",len(AccData[0]))

clf_svm.fit(AccData,trainlabels)
clf_log.fit(AccData,trainlabels)
clf_gnb.fit(AccData,trainlabels)
clf_nc.fit(AccData,trainlabels)

svm_counter = 0
LeCounter = 0
k = len(AccData)
for i in range(0,k,1):
    predLab_svm = int(clf_svm.predict([AccData[i]]))
    predLab_log = int(clf_log.predict([AccData[i]]))
    predLab_gnb = int(clf_gnb.predict([AccData[i]]))
    predLab_nc = int(clf_nc.predict([AccData[i]]))
    h = predLab_svm + predLab_log + predLab_gnb +predLab_nc
    if(h >= 3):
        my_predLab = 1
    elif(h <= 1):
        my_predLab = 0
    else:
        my_predLab = predLab_svm
    if(my_predLab == trainlabels[i]):
        LeCounter += 1
    if(predLab_svm == trainlabels[i]):
        svm_counter += 1

FinalAcc = LeCounter/k
SVMAc = svm_counter/k
print("The Accuracy is: ",FinalAcc*100)
# print("The SVM Accuracy is: ",FinalAcc*100)


print("\nThe test data will be read shortly and labels saved in a file named sh486_testLabels")

##################
# Read Test data
##################

testfile = sys.argv[3]
testdata = []
print("Starting to read test data...")
with open(testfile,"r") as infile:
    for line in infile:
        temp = line.split()
        l = array.array("i")
        for i in temp:
            l.append(int(i))
        testdata.append(l)

print("Done reading test data")

# print("rows= ",len(testdata)," cols= ",len(testdata[0]))

#Reducing Dimensions to slected features
print("   Starting Feature Extraction...")

argument2 = copy.deepcopy(originalFea)
testdata1 = CreateDataSet(argument2,testdata)

# print("rows= ",len(testdata1)," cols= ",len(testdata1[0]))

print("   Ending Feature Extraction")
#create a file
f1 = open("sh486_testLabels","w+")

for i in range(0, len(testdata1),1):
        lab1 = int(clf_svm.predict([testdata1[i]]))
        lab2 = int(clf_log.predict([testdata1[i]]))
        lab3 = int(clf_gnb.predict([testdata1[i]]))
        lab4 = int(clf_nc.predict([testdata1[i]]))
        h = lab1 + lab2 + lab3 + lab4
        if(h >= 3):
            f1.write(str(1)+" "+str(i)+"\n") 
        elif(h <= 1):
            f1.write(str(0)+" "+str(i)+"\n")
        else:
            f1.write(str(lab1)+" "+str(i)+"\n")
            
print("Done with test data")
print("--- %s seconds ---" % (time.time() - start_time))





    








