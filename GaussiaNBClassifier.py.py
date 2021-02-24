#The Gaussian NaiveBayes classifier
import sklearn
from sklearn.datasets import load_breast_cancer #Importing dataset named Breast Cancer
from sklearn.model_selection import train_test_split #Organizes data into sets by importing train_test_split function from sklearn
from sklearn.naive_bayes import GaussianNB #To buid the model by importing the GaussianNB module
from sklearn.metrics import accuracy_score #To import the accuracy_score() fuction for accuracy

data = load_breast_cancer() #Loading the dataset
#Organizing data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
#print(label_names)
#print(labels[0])
print(feature_names[0])
print(features[0])
#Splitting data into training and test data sets
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.40, random_state = 42)
#Initializing the GaussianNB model
gnb = GaussianNB()
#Training the module by fitting it into the data 
model = gnb.fit(train, train_labels)
#Evaluating the model and its accuracy by making predictions on our test data and finding its accuracy the predict() function allows for making predictions
#Accuracy is found by comparing the two arrays
preds = gnb.predict(test)
print("The Predictions are:", preds) #The printed series of 1s and 0s are the predicted values for the tumor classes
#Accuracy
print("\nThe Accuracy:", accuracy_score(test_labels, preds))
