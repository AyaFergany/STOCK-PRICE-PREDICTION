# STOCK-PRICE-PREDICTION
Machine Learning INTERNSHIP
Project 1


# Random forests or random decision forests
is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned.Random decision forests correct for decision trees' habit of overfitting to their training set.Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees.However, data characteristics can affect their performance.


# Implementation
Now let’s start our implementation using Python and a Jupyter Notebook.

Once the Jupyter Notebook is up and running, the first thing we should do is import the necessary libraries.

We need to import:

NumPy

Pandas

RandomForestRegressor

train_test_split

r2_score

mean squared error

Seaborn

To actually implement the random forest regressor, we’re going to use scikit-learn, and we’ll import our RandomForestRegressor from sklearn.ensemble.

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score,mean_squared_error

import seaborn as sns

Import Libraries for Random Forest Regression

# Load the Data
Once the libraries are imported, our next step is to load the data, stored here. You can download the data and keep it in your local folder. After that we can use the read_csv method of Pandas to load the data into a Pandas data frame df, as shown below.

Also shown in the snapshot of the data below, the data frame has two columns, x and y. Here, x is the feature and y is the label. We’re going to predict y using x as an independent variable.

df = pd.read_csv(‘Random-Forest-Regression-Data.csv’)

![image](https://user-images.githubusercontent.com/91394241/217531933-18d681cb-cfe8-40fb-b991-eb20b4470c5e.png)


Data snapshot for Random Forest Regression

# Data pre-processing

Before feeding the data to the random forest regression model, we need to do some pre-processing.

Here, we’ll create the x and y variables by taking them from the dataset and using the train_test_split function of scikit-learn to split the data into training and test sets.

We also need to reshape the values using the reshape method so that we can pass the data to train_test_split in the format required.

Note that the test size of 0.3 indicates we’ve used 30% of the data for testing. random_state ensures reproducibility. For the output of train_test_split, we get x_train, x_test, y_train, and y_test values.

1-    x = df.x.values.reshape(-1, 1)

2-    y = df.y.values.reshape(-1, 1)

3-    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


Data Pre processing for Random Forest Regression
# Train the model
We’re going to use x_train and y_train, obtained above, to train our random forest regression model. We’re using the fit method and passing the parameters as shown below.

Note that the output of this cell is describing a large number of parameters like criteria, max depth, etc. for the model. All these parameters are configurable, and you’re free to tune them to match your requirements.
![image](https://user-images.githubusercontent.com/91394241/217531796-d275a53f-8cc2-4694-a481-19d42ca86eff.png)


Random Forest Regression Model Training using Fit method

# Prediction
Once the model is trained, it’s ready to make predictions. We can use the predict method on the model and pass x_test as a parameter to get the output as y_pred.

Notice that the prediction output is an array of real numbers corresponding to the input array.
![image](https://user-images.githubusercontent.com/91394241/217531714-c2a4767d-241b-46e5-b882-1754a8fb4506.png)



Random Forest Regression Model Prediction

# Model Evaluation
Finally, we need to check to see how well our model is performing on the test data. For this, we evaluate our model by finding the root mean squared error produced by the model.

Mean squared error is a built in function, and we are using NumPy’s square root function (np.sqrt) on top of it to find the root mean squared error value.



![image](https://user-images.githubusercontent.com/91394241/217531564-83d5a893-1ba3-4eae-827a-671e9de9db69.png)





# Sources
https://heartbeat.comet.ml/random-forest-regression-in-python-using-scikit-learn-9e9b147e2153
