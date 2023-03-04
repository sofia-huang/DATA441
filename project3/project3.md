## Project 3
# Gradient Boosting Algorithm
#### Goal: Implement the Gradient Boosting algorithm with user defined choices for Regressor_1 and Regressor_2. Test the Boosted Locally Weighted Regressor with different choices of data (such as "cars.csv", "concrete.csv" and "housing.csv") and different choice of kernels, such as Gaussian, Tricubic, Epanechnikov and Quartic. Use k-fold cross validations and compare with other regressors. 

Boosting is a strategy that combines multiple, simple or "weak", models into a single model. This overall, single model becomes a stronger predictor with the addition of the simple models. 

The term gradient, in this case, is referring to gradient descent used in the algorithm to minimize the loss function of the overall model. The simple models are added, one at a time, to the overall model, based on the gradient descent optimization process. The existing simple models do not change when new ones are added to the overall model.

Gradient Boosting is an algorithm where each predictor corrects the previous model's errors. Each predictor is trained using the previous model's residual errors as labels/target. 

Here is the boosted Lowess Regression function I used. It takes in user defined regressors and predicts "xnew" using Gradient Boosting.

```Python
def boosted_lwr(x, y, xnew, model1, model2, kernel, f=1/3,iter=2,intercept=True):
  model1.fit(x,y)
  residuals1 = y - model1.predict(x)
  model2.fit(x,residuals1)
  output = model1.predict(xnew) + model2.predict(xnew)
  return output 
```

First, I tested it on the cars.csv dataset. I split the data into training and testing sets and scaled them. Then, I used the optimal hyperparameter values I found in the previous project for the f value and number of iterations. I tested the function using various kernels and here are the results. 

| Kernel      | MSE |
| ----------- | ----------- |
| Tricubic    | 17.843509217584472 |
| Gaussian    | 17.967290079505435 |
| Epanechnikov| 18.104543915058876 |
| Quartic     | 17.83746086303032 |

[Back to Project Index](https://sofia-huang.github.io/DATA441/)
