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
  # y estimates
  yest = np.zeros(n)

  if len(y.shape)==1: # here we make column vectors
    y = y.reshape(-1,1)

  if len(x.shape)==1:
    x = x.reshape(-1,1)
  
  if intercept:
    x1 = np.column_stack([np.ones((len(x),1)),x])
  else:
    x1 = x

  h = [np.sort(np.sqrt(np.sum((x-x[i])**2,axis=1)))[r] for i in range(n)]
  # can try to use argsprt instead of clipping 
  w = np.clip(dist(x,x) / h, 0.0, 1.0)
  w = (1 - w ** 3) ** 3

  #Looping through all X-points
  delta = np.ones(n)
  for iteration in range(iter):
    for i in range(n):
      W = np.diag(w[:,i])
      b = np.transpose(x1).dot(W).dot(y)
      A = np.transpose(x1).dot(W).dot(x1)

      A = A + 0.0001*np.eye(x1.shape[1])
      beta = linalg.solve(A, b)
      yest[i] = np.dot(x1[i],beta)

    # robustifiction part, remove higher residuals by weighting them with 0
    residuals = y - yest
    s = np.median(np.abs(residuals))
    delta = np.clip(residuals / (6.0 * s), -1, 1)
    delta = (1 - delta ** 2) ** 2

  if x.shape[1]==1:
    f = interp1d(x.flatten(),yest,fill_value='extrapolate')
  else:
    f = LinearNDInterpolator(x, yest)
  output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
  if sum(np.isnan(output))>0:
    g = NearestNDInterpolator(x,y.ravel()) 
    # output[np.isnan(output)] = g(X[np.isnan(output)])
    output[np.isnan(output)] = g(xnew[np.isnan(output)])
  return output
  ```
