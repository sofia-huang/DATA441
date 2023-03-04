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
#### cars.csv

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

The boosted Lowess regression model performed almost as well as the Random Forest model for the cars.csv dataset.

#### housing.csv

Next, I repeated the same process for the housing.csv dataset. I split the data into training and testing sets and scaled them. Then, found the best performing kernel based on mean squared error. Since I had not used the housing dataset in the previous project where I found the optimal f and iter hyperparameters, I manually tested to see what values produced the best results and found them pretty quickly. I ended up using an f-value of 1/60 and 2 iterations.

Here are the results.

| Kernel      | MSE |
| ----------- | ----------- |
| Tricubic    | 17.91412298055437 |
| Gaussian    | 42.106965749700066 |
| Epanechnikov| 19.60280712041183 |
| Quartic     | 18.198843595500563 |

Interestingly, the Gaussian kernel produced a much worse mean squared error than the rest of the kernels. The best was the Tricubic kernel.

Again, I preformed a K-Fold cross validation on the housing.csv dataset and used the Tricubic kernel, as well as, the optimal f and iter hyperparameters. 

```Python
mse_lwr = []
mse_rf = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)

for idxtrain, idxtest in kf.split(x_hs):
  xtrain = x_hs[idxtrain]
  ytrain = y_hs[idxtrain]
  ytest = y_hs[idxtest]
  xtest = x_hs[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  model1 = Lowess_AG_MD(kernel='Tricubic',f=1/60,iter=2,intercept=True)
  model2 = Lowess_AG_MD(kernel='Tricubic',f=1/60,iter=2,intercept=True)
  yhat_lw = boosted_lwr(xtrain,ytrain,xtest,model1,model2,kernel='Tricubic',f=1/60,iter=2,intercept=True)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```

| Output      | 
| ----------- |
| The Cross-validated Mean Squared Error for Locally Weighted Regression is : 17.12735864190333    | 
| The Cross-validated Mean Squared Error for Random Forest is : 14.950868358416638   | 

For the housing dataset, the boosted Lowess regression model did not perform quite as well as the Random Forest model. However, the difference between the 2 models' results is not large.

#### concrete.csv

Lastly, I did the same process for the concrete.csv dataset. For this data, I scaled both the features and the target variables since the variation between the values was high and making the resulting mean squared errors very large. 

| Kernel      | MSE |
| ----------- | ----------- |
| Tricubic    | 0.3017316701837452 |
| Gaussian    | 0.4438413389361671 |
| Epanechnikov| 0.30044029773915715 |
| Quartic     | 0.29969159231812365 |

```Python
mse_lwr = []
mse_rf = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)

for idxtrain, idxtest in kf.split(x_cc):
  xtrain = x_cc[idxtrain]
  ytrain = y_cc[idxtrain]
  ytest = y_cc[idxtest]
  xtest = x_cc[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  model1 = Lowess_AG_MD(kernel='Quartic',f=25/len(xtrain),iter=1,intercept=True)
  model2 = Lowess_AG_MD(kernel='Quartic',f=25/len(xtrain),iter=1,intercept=True)
  yhat_lw = boosted_lwr(xtrain,ytrain,xtest,model1,model2,kernel='Quartic',f=25/len(xtrain),iter=1,intercept=True)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```

| Output      | 
| ----------- |
| The Cross-validated Mean Squared Error for Locally Weighted Regression is : 0.24557337186739284 | 
| The Cross-validated Mean Squared Error for Random Forest is : 0.16350294426378792 | 

[Back to Project Index](https://sofia-huang.github.io/DATA441/)
