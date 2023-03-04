## Project 3
# Gradient Boosting Algorithm
#### Goal: Implement the Gradient Boosting algorithm with user defined choices for Regressor_1 and Regressor_2. Test the Boosted Locally Weighted Regressor with different choices of data (such as "cars.csv", "concrete.csv" and "housing.csv") and different choice of kernels, such as Gaussian, Tricubic, Epanechnikov and Quartic. Use k-fold cross validations and compare with other regressors. 

Boosting is a strategy that combines multiple, simple or "weak", models into a single model. This overall, single model becomes a stronger predictor with the addition of the simple models. 

The term gradient, in this case, is referring to gradient descent used in the algorithm to minimize the loss function of the overall model. The simple models are added, one at a time, to the overall model, based on the gradient descent optimization process. The existing simple models do not change when new ones are added to the overall model.

Gradient Boosting is an algorithm where each predictor corrects the previous model's errors. Each predictor is trained using the previous model's residual errors as labels/target. 

Here is the boosted Lowess regression function I used. It takes in user defined regressors and predicts "xnew" using Gradient Boosting.

```Python
def boosted_lwr(x, y, xnew, model1, model2, kernel, f=1/3,iter=2,intercept=True):
  model1.fit(x,y)
  residuals1 = y - model1.predict(x)
  model2.fit(x,residuals1)
  output = model1.predict(xnew) + model2.predict(xnew)
  return output 
```

First, I tested it on the cars.csv dataset. I split the data into training and testing sets and scaled them. Then, I used the optimal hyperparameter values I found in the previous project for the f value and number of iterations. I tested the function using various kernels. Below is an example of the code I used to get the results. I just changed the kernel parameter when creating the 2 regressor models and kept the other hyperparameters the same.

```Python
model1 = Lowess_AG_MD(kernel='Tricubic',f=1/3,iter=2,intercept=True)
model2 = Lowess_AG_MD(kernel='Tricubic',f=1/3,iter=2,intercept=True)
yhat = boosted_lwr(xtrain,ytrain,xtest,model1,model2,kernel='Tricubic',f=1/3,iter=1,intercept=True)
mse(ytest,yhat)
```
Here are the results.

| Kernel      | MSE |
| ----------- | ----------- |
| Tricubic    | 17.843509217584472 |
| Gaussian    | 17.967290079505435 |
| Epanechnikov| 18.104543915058876 |
| Quartic     | 17.83746086303032 |

The mean squared errors are all pretty close but the best performing kernel was Quartic.

Then, I used K-Fold Cross Validation to compare the boosted Lowess regressor with a Random Forest Regressor. I used the best performing kernel and the optimal f value and iteration number for the boosted Lowess regressor. Below is the code I used.

```Python
mse_lwr = []
mse_rf = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)

for idxtrain, idxtest in kf.split(x_cr):
  xtrain = x_cr[idxtrain]
  ytrain = y_cr[idxtrain]
  ytest = y_cr[idxtest]
  xtest = x_cr[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  model1 = Lowess_AG_MD(kernel='Quartic',f=1/3,iter=2,intercept=True)
  model2 = Lowess_AG_MD(kernel='Quartic',f=1/3,iter=2,intercept=True)
  yhat_lw = boosted_lwr(xtrain,ytrain,xtest,model1,model2,kernel='Quartic',f=1/3,iter=2,intercept=True)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```

| Output      | 
| ----------- |
| The Cross-validated Mean Squared Error for Locally Weighted Regression is : 17.5301456603268    | 
| The Cross-validated Mean Squared Error for Random Forest is : 17.204446835237757    | 


[Back to Project Index](https://sofia-huang.github.io/DATA441/)
