# STOCK-PRICE-PREDICTION
Machine Learning INTERNSHIP
Project 1


# Random forests or random decision forests
is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned.Random decision forests correct for decision trees' habit of overfitting to their training set.Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees.However, data characteristics can affect their performance.


Let’s see Random Forest Regression in action!
Now that we have a basic understanding of how the Random Forest Regression model works, we can assess its performance on a real-world dataset. Similar to my previous posts, I will be using data on House Sales in King County, USA.

After importing the libraries, importing the dataset, addressing null values, and dropping any necessary columns, we are ready to create our Random Forest Regression model!

Step 1: Identify your dependent (y) and independent variables (X)

Our dependent variable will be prices while our independent variables are the remaining columns left in the dataset.


Step 2: Split the dataset into the Training set and Test set


The importance of the training and test split is that the training set contains known output from which the model learns off of. The test set then tests the model’s predictions based on what it learned from the training set.

Step 3: Training the Random Forest Regression model on the whole dataset


From the sklearn package containing ensemble learning, we import the class RandomForestRegressor, create an instance of it, and assign it to a variable. The parameter n_estimators creates n number of trees in your random forest, where n is the number you pass in. We passed in 10. The .fit() function allows us to train the model, adjusting weights according to the data values in order to achieve better accuracy. After training, our model is ready to make predictions, which is called by the .predict() method.

Step 4: Predicting the Test set results


Now that we’ve successfully created a Random Forest Regression model, we must assess is performance.


R² score tells us how well our model is fitted to the data by comparing it to the average line of the dependent variable. If the score is closer to 1, then it indicates that our model performs well versus if the score is farther from 1, then it indicates that our model does not perform so well.
