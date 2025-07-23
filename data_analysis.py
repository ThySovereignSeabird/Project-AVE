import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # supplies violinplot and heatmap
import matplotlib.pyplot as plt

dataset = pd.read_csv("./output/all_item_stats.csv")

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
from sklearn.ensemble import RandomForestClassifier

def heatmap():
    plt.figure(figsize=(7,4)) 
    sns.heatmap(dataset.select_dtypes(include='number').corr(),annot=True,cmap='cubehelix_r') # draws heatmap with input as the correlation matrix calculated by(iris.corr())
    plt.show()

# heatmap()

def trainSetup():
    # train, test = train_test_split(dataset, test_size = 0.2)# in this our main data is split into train and test
    train = dataset[dataset["group"] == "train"]
    test = dataset[dataset["group"] == "test"]

    featureCols = ['area','area_outline','area_inner','color_count','internal_edge_density','internal_contrast_ratio_avg','internal_contrast_ratio_dev','luminance_avg','luminance_dev','luminance_median','lum_inner_avg','lum_inner_dev','lum_outline_avg','lum_outline_dev','outline_boldness','luminance_highlight','contrast_highlight_outline','contrast_highlight_median','contrast_median_outline','contrast_top_bottom']

    train_X = train[featureCols]
    train_y=train.outline# output of our training data
    test_X= test[featureCols]
    test_y =test.outline   #output value of test data

    model=RandomForestClassifier()
    model.fit(train_X,train_y)
    prediction=model.predict(test_X)
    print('The accuracy of the Random Forest is',metrics.accuracy_score(prediction,test_y))

    probs_train = model.predict_proba(train_X)
    probs_test = model.predict_proba(test_X)

    # Get class labels
    class_labels = model.classes_
    prob_columns = [f'prob_{cls}' for cls in class_labels]

    # Create DataFrames for both sets
    df_train = pd.DataFrame(probs_train, columns=prob_columns)
    df_train['name'] = train['name'].values
    df_train['true_label'] = train_y.values
    df_train['predicted_label'] = model.predict(train_X)
    df_train['split'] = 'train'

    df_test = pd.DataFrame(probs_test, columns=prob_columns)
    df_test['name'] = test['name'].values
    df_test['true_label'] = test_y.values
    df_test['predicted_label'] = prediction
    df_test['split'] = 'test'

    # Combine both sets
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    # Export to CSV
    df_all.to_csv("output/all_prediction_probabilities.csv", index=False)

    # model = svm.SVC() #select the algorithm
    # model.fit(train_X,train_y) # we train the algorithm with the training data and the training output
    # prediction=model.predict(test_X) #now we pass the testing data to the trained algorithm
    # print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 

    # model=DecisionTreeClassifier()
    # model.fit(train_X,train_y)
    # prediction=model.predict(test_X)
    # print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))

    # model=KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class
    # model.fit(train_X,train_y)
    # prediction=model.predict(test_X)
    # print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))

# trainSetup()