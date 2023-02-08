## Project 1 
# Introduction to Locally Weighted Regression

In this page, I will explain my understanding of locally weighted regression or LOWESS. However first, we must define the preliminaries. 

***Regression***: A technique used to train an algorithm to understand the relationship between independent variables (input features) and a continuous dependent variable (output). The errors in measurement or "noise" are assumed to be normally distributed with a mean of 0 and some unknown standard deviation. We want to use the trained algorithm to predict the expected value of the output as a function of the input features. 

***Linear regression***: One of the basic types of regression technique which plots a straight line of best fit within data points to minimise error between the line and the data points. It is assumed that the relationship between independent and dependent variables is linear when this technique is used. 

When there are multiple independent variables, this is called multiple linear regression and the idea is as follows: 

$$\text{Predicted Value} = weight_1 \cdot \text{Feature}_1 + weight_2 \cdot \text{Feature}_2 + ... + weight_p \cdot \text{Feature}_p $$

The algorithm learns the optimal weights or coefficients of the input features as an iterative process, usually through a method such as gradient descent. 

Now that we know the foundation, how does an algorithm make predictions when the relationship between the independent and dependent variables are ***non-linear***? We use locally weighted regression (LOWESS).

***Locally weighted regression***: A technique that modifies the linear regression to make it fit non-linear functions. The non-parametric algorithm fits a linear model to localized subsets of the data to build a function that describes the variation in the data, point by point. Basically, there are many small local functions rather than one global function. The weights (parameters) of each training data point are computed locally (for each query data point) based on the distance of the training points from the query point, using a kernel function. This makes minimizing the errors easier and means the parameters are unique to each data point (instead of fitting a fixed set of parameters to the data). 

We use LOWESS when the distribution of data is non-linear and the number of features is smaller, as the process is highly exhaustive and computationally expensive. It also requires a large/dense dataset to produce good results since LOWESS relies on the local data structure when fitting the model.


