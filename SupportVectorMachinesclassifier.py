import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
#Loading the input data
iris = datasets.load_iris()
#Taking first two values
x = iris.data[:, :2]
y = iris.target
#plotting the vector machine bmoundaries with original data on a meshplot
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
x_plot = np.c_[xx.ravel(), yy.ravel()]
#Giving the value of regularization parameter
c = 1.0
#Creating SVM classifier object
svc_classifier = svc_classifier.SVC(kernel = 'linear', c=c, decision_function_shape = 'ovr').fit(x,y)
z = svc_classifier.predict(x_plot)
z = z.reshape(xx.shape)
plt.figure(figsize = (15, 5))
plt.subplot(121)
plt.contour(xx, yy, z, cmap = plt.cm.tab10, alpha = 0.3)
plt.scatter(x[:, 0], x[:, 1], c = y, cmap = plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')