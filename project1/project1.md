## Project 1 
# Introduction to Locally Weighted Regression

In this page, I will explain my understanding of locally weighted regression or LOWESS. However first, we must define the preliminaries. 

***Regression***: A technique used to train an algorithm to understand the relationship between independent variables (input features) and a continuous dependent variable (output). The errors in measurement or "noise" are assumed to be normally distributed with a mean of 0 and some unknown standard deviation. We want to use the trained algorithm to predict the expected value of the output as a function of the input features. 

***Linear regression***: One of the basic types of regression technique which plots a straight line of best fit within data points to minimise error between the line and the data points. It is assumed that the relationship between independent and dependent variables is linear when this technique is used. 

When there are multiple independent variables, this is called multiple linear regression and the idea is as follows: 

$$\text{Predicted Value} = weight_1 \cdot \text{Feature}_1 + weight_2 \cdot \text{Feature}_2 + ... + weight_p \cdot \text{Feature}_p $$

The algorithm learns the optimal weights or coefficients of the input features as an iterative process, usually through a method such as gradient descent. 

Now that we know the foundation, how does an algorithm make predictions when the relationship between the independent and dependent variables are ***non-linear***? We use locally weighted regression (LOWESS).

***Locally weighted regression***: A technique that modifies the linear regression to make it fit non-linear functions.  
