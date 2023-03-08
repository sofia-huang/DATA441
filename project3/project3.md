## Project 3
# Gradient Boosting Algorithm
#### Goal: Implement the Gradient Boosting algorithm with user defined choices for Regressor_1 and Regressor_2. Test the Boosted Locally Weighted Regressor with different choices of data (such as "cars.csv", "concrete.csv" and "housing.csv") and different choice of kernels, such as Gaussian, Tricubic, Epanechnikov and Quartic. Use k-fold cross validations and compare with other regressors. 

Boosting is a strategy that combines multiple, simple or "weak", models into a single model. This overall, single model becomes a stronger predictor with the addition of the simple models. 

The term gradient, in this case, is referring to gradient descent used in the algorithm to minimize the loss function of the overall model. The simple models are added, one at a time, to the overall model, based on the gradient descent optimization process. The existing simple models do not change when new ones are added to the overall model.

Gradient Boosting is an algorithm where each predictor corrects the previous model's errors. Each predictor is trained using the previous model's residual errors as labels/target. 

Here is the boosted Lowess regression function I used. It takes in user defined regressors and predicts "xnew" using Gradient Boosting.

```Python
def boosted_lwr(x, y, xnew, model1, model2, f=1/3,iter=2,intercept=True):
  model1.fit(x,y)
  residuals1 = y - model1.predict(x)
  model2.fit(x,residuals1)
  output = model1.predict(xnew) + model2.predict(xnew)
  return output 
```
#### cars.csv

First, I tested it on the cars.csv dataset. I split the data into training and testing sets and scaled them. Then, I used the optimal hyperparameter values I found in the previous project for the f value and number of iterations. I tested the function using various kernels. Below is an example of the code I used to get the results. I just changed the kernel parameter when creating the 2 regressor models and kept the other hyperparameters the same.

```Python
model1 = Lowess_AG_MD(kernel=Tricubic,f=1/3,iter=2,intercept=True)
model2 = Lowess_AG_MD(kernel=Tricubic,f=1/3,iter=2,intercept=True)
yhat = boosted_lwr(xtrain,ytrain,xtest,model1,model2,f=1/3,iter=2,intercept=True)
mse(ytest,yhat)
```
Here are the results.

| Kernel      | MSE |
| ----------- | ----------- |
| Tricubic    | 20.56791672624454 |
| Gaussian    | 19.273326190314418 |
| Epanechnikov| 19.258883776905023 |
| Quartic     | 20.050242259086918 |

The mean squared errors are all pretty close but the best performing kernel was Epanechnikov.

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

  model1 = Lowess_AG_MD(kernel=Epanechnikov,f=1/3,iter=2,intercept=True)
  model2 = Lowess_AG_MD(kernel=Epanechnikov,f=1/3,iter=2,intercept=True)
  yhat_lw = boosted_lwr(xtrain,ytrain,xtest,model1,model2,f=1/3,iter=2,intercept=True)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```

| Output      | 
| ----------- |
| The Cross-validated Mean Squared Error for Locally Weighted Regression is : 16.74565037038816 | 
| The Cross-validated Mean Squared Error for Random Forest is : 17.07145834751521 | 

The boosted Lowess regression model performed better than the Random Forest model for the cars.csv dataset.

#### housing.csv

Next, I repeated the same process for the housing.csv dataset. I split the data into training and testing sets and scaled them. Then, found the best performing kernel based on mean squared error. Since I had not used the housing dataset in the previous project where I found the optimal f and iter hyperparameters, I used the GridSearchCV method to find the optimal values. I ended up using an f-value of 1/3 and 1 iteration.

```Python
lwr_pipe = Pipeline([('zscores', StandardScaler()),
                     ('lwr', Lowess_AG_MD())])
params = [{'lwr__f': [1/i for i in [3, 5, 10, 20, 25, 30, 40, 50, 60]],
         'lwr__iter': [1,2,3]}]
         
gs_lowess = GridSearchCV(lwr_pipe,
                      param_grid=params,
                      scoring='neg_mean_squared_error',
                      cv=5)
gs_lowess.fit(x_hs, y_hs)
gs_lowess.best_params_
```

Here are the results.

| Kernel      | MSE |
| ----------- | ----------- |
| Tricubic    | 22.545513269180745 |
| Gaussian    | 18.98995559886556 |
| Epanechnikov| 21.972367856248418 |
| Quartic     | 22.225757831072325 |

The best performing kernel was the Gaussian kernel.

Again, I performed a K-Fold cross validation on the housing.csv dataset and used the Gaussian kernel, as well as, the optimal f and iter hyperparameters. 

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

  model1 = Lowess_AG_MD(kernel=Gaussian,f=1/3,iter=1,intercept=True)
  model2 = Lowess_AG_MD(kernel=Gaussian,f=1/3,iter=1,intercept=True)
  yhat_lw = boosted_lwr(xtrain,ytrain,xtest,model1,model2,f=1/3,iter=1,intercept=True)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```

| Output      | 
| ----------- |
| The Cross-validated Mean Squared Error for Locally Weighted Regression is : 12.5407953777719 | 
| The Cross-validated Mean Squared Error for Random Forest is : 14.939881512619337 | 

For the housing dataset, the boosted Lowess regression model performed better than the Random Forest model, yet again. 

#### concrete.csv

Lastly, I did the same process for the concrete.csv dataset. Below are the results for the boosted Lowess regression function using different kernels and the optimal hyperparameters found in the previous project. The best performing kernel was Gaussian.

| Kernel      | MSE |
| ----------- | ----------- |
| Tricubic    | 82.99646217178436 |
| Gaussian    | 76.70472625647679 |
| Epanechnikov| 81.94149152387605 |
| Quartic     | 82.7984533480926 |

Then, I used the Gaussian kernel and the optimal hyperparameters to do a k-fold cross validation and compared the results with a Random Forest regression model.

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

  model1 = Lowess_AG_MD(kernel=Gaussian,f=25/len(xtrain),iter=1,intercept=True)
  model2 = Lowess_AG_MD(kernel=Gaussian,f=25/len(xtrain),iter=1,intercept=True)
  yhat_lw = boosted_lwr(xtrain,ytrain,xtest,model1,model2,f=25/len(xtrain),iter=1,intercept=True)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```

| Output      | 
| ----------- |
| The Cross-validated Mean Squared Error for Locally Weighted Regression is : 67.13158282403913 | 
| The Cross-validated Mean Squared Error for Random Forest is : 45.792357518954006 | 

This dataset had the biggest difference in mean squared errors between the boosted Lowess regression and the Random Forest model. Compared to the other two datastes, the concrete.csv dataset took the longest to run and performed worse than Random Forest. 

The full Python notebook is linked here: [Project 3 Python Notebook](https://colab.research.google.com/drive/1E8XM0sj-mrrp9jlvJiPaL3kkTgrxz1kr#scrollTo=fc_2seba9hG8)

[Back to Project Index](https://sofia-huang.github.io/DATA441/)
