## Project 2
# Modifying A. Gramfort's LOWESS Function
#### Goal: Modify the function to accommodate train/test sets and multidimensional features, tested with k-fold cross validations. Create a SciKitLearn-compliant version, tested with GridSearchCV.

Here we have a function that calculates the Euclidean distance between all observations in u and v.

```Python
def dist(u,v):
  if len(v.shape)==1:
    v = v.reshape(1,-1)
  d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))])
  return d
```

Here is our modified version of A. Gramfort's function. 

```Python
def lw_ag_md(x, y, xnew,f=2/3,iter=3, intercept=True):

  n = len(x)
  r = int(ceil(f * n))
  yest = np.zeros(n)
  # checking data dimensionality
  if len(y.shape)==1: # here we make column vectors
    y = y.reshape(-1,1)

  if len(x.shape)==1:
    x = x.reshape(-1,1)
  # appending augmented matrix x1 
  if intercept:
    x1 = np.column_stack([np.ones((len(x),1)),x])
  else:
    x1 = x
  # compute the max bounds for the local neighborhoods
  # closer points get higher weight
  h = [np.sort(np.sqrt(np.sum((x-x[i])**2,axis=1)))[r] for i in range(n)]

  w = np.clip(dist(x,x) / h, 0.0, 1.0)
  w = (1 - w ** 3) ** 3

  #Looping through all X-points
  delta = np.ones(n)
  # robustifying part
  for iteration in range(iter):
    for i in range(n):
      W = np.diag(w[:,i])
      # solve linear system
      b = np.transpose(x1).dot(W).dot(y)
      A = np.transpose(x1).dot(W).dot(x1)
      ##
      A = A + 0.0001*np.eye(x1.shape[1]) # if we want L2 regularization
      beta = linalg.solve(A, b)
      #beta, res, rnk, s = linalg.lstsq(A, b)
      yest[i] = np.dot(x1[i],beta)
    # clip/remove data that has large residuals
    residuals = y - yest
    s = np.median(np.abs(residuals))
    delta = np.clip(residuals / (6.0 * s), -1, 1)
    delta = (1 - delta ** 2) ** 2
  # for 1-dimension, do interpolation for xnew
  if x.shape[1]==1:
    f = interp1d(x.flatten(),yest,fill_value='extrapolate')
    output = f(xnew)
  else:
  # for multiple-dimensional data
    output = np.zeros(len(xnew))
    for i in range(len(xnew)):
      ind = np.argsort(np.sqrt(np.sum((x-xnew[i])**2,axis=1)))[:r]
      # extract first 3 principle components
      pca = PCA(n_components=3)
      x_pca = pca.fit_transform(x[ind])
      # get convex hull in order to interpolate 
      tri = Delaunay(x_pca,qhull_options='QJ')
      f = LinearNDInterpolator(tri,y[ind])
      output[i] = f(pca.transform(xnew[i].reshape(1,-1))) # the output may have NaN's where the data points from xnew are outside the convex hull of X
  if sum(np.isnan(output))>0:
    # accomodates possible new data that falls out of scope of old data, cannot extrapolate, we just use the nearest observation from old data
    g = NearestNDInterpolator(x,y.ravel()) 
    output[np.isnan(output)] = g(xnew[np.isnan(output)])
  return output
  ```
  
  This function takes in the x and y data, along with new x data, hyperparameter f, and the number of iterations. f is a fraction of the data that comprises the nearest neighbors of each observation. The number of iterations is for the robustification portion of the function which reweights the points and clips the points that have high residuals so that outliers do not affect the results. 
  
  The function is able to predict y's for the xnew parameter through interpolation. For multi-dimensional data we extract the first 3 principle components and use a convex hull to interpolate.
  
  Let's test the function on some real data using k-fold cross validations. *I tried to change the kernel function that was used to calculate the weights, however this did not change the results. 
  
```Python
kf = KFold(n_splits=10,shuffle=True,random_state=123)
mse_test_lw_ag_md = []
fs = []
f_range = f_range = [1/50, 1/20, 1/10, 1/5, 1/2]

for f in f_range:

  for idxtrain, idxtest in kf.split(x_cars):
    xtrain = x_cars[idxtrain]
    xtest = x_cars[idxtest]
    ytrain = y_cars[idxtrain]
    ytest = y_cars[idxtest]    

    yhat = lw_ag_md(xtrain,ytrain,xtest,f=f,iter=3,intercept=True)
    mse_test_lw_ag_md.append(mse(ytest,yhat))
    fs.append(f)
idx = np.argmin(mse_test_lw_ag_md)
print('The validated MSE for Lowess is : '+str(np.mean(mse_test_lw_ag_md)))
print('The optimal f is ' + str(fs[idx]) + '; and its corresponding MSE is ' + str(np.min(mse_test_lw_ag_md)))

```
Output:

The validated MSE for Lowess is : 28.167368454834886

The optimal f is 0.5; and its corresponding MSE is 14.202761553214305

For the cars.csv dataset, we can see that using a f of 1/2 will result in the best MSE.

Then, I decided to optimize the number of robustifying iterations using k-fold cross validation.

```Python
kf = KFold(n_splits=10,shuffle=True,random_state=123)
mse_test_lw_ag_md = []
iters = []
i_range = [2, 3, 4, 5, 6, 7]

for i in i_range:

  for idxtrain, idxtest in kf.split(x_cars):
    xtrain = x_cars[idxtrain]
    xtest = x_cars[idxtest]
    ytrain = y_cars[idxtrain]
    ytest = y_cars[idxtest]    

    yhat = lw_ag_md(xtrain,ytrain,xtest,f=1/2,iter=i,intercept=True)
    mse_test_lw_ag_md.append(mse(ytest,yhat))
    iters.append(i)
idx = np.argmin(mse_test_lw_ag_md)
print('The validated MSE for Lowess is : '+str(np.mean(mse_test_lw_ag_md)))
print('The optimal number of iterations is ' + str(iters[idx]) + '; and its corresponding MSE is ' + str(np.min(mse_test_lw_ag_md)))
```
Output:

The validated MSE for Lowess is : 25.529407163652625

The optimal number of iterations is 2; and its corresponding MSE is 14.202761553214305

For the cars.csv dataset, we can see that using 2 iterations will result in the best MSE.


Here is the same function but scikit-learn compliant.
```Python
class Lowess_AG_MD:
    def __init__(self, f = 1/10, iter = 3,intercept=True):
        self.f = f
        self.iter = iter
        self.intercept = intercept
    
    def fit(self, x, y):
        f = self.f
        iter = self.iter
        self.xtrain_ = x
        self.yhat_ = y

    def predict(self, x_new):
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_
        f = self.f
        iter = self.iter
        intercept = self.intercept
        return lw_ag_md(x, y, x_new, f, iter, intercept)

    def get_params(self, deep=True):
    # suppose this estimator has parameters "f", "iter" and "intercept"
        return {"f": self.f, "iter": self.iter,"intercept":self.intercept}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
```
  
[Back to Project Index](https://sofia-huang.github.io/DATA441/)
