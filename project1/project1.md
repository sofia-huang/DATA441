## Project 1 
# Introduction to Locally Weighted Regression

In this page, I will explain my understanding of locally weighted regression or LOWESS. However first, we must define the preliminaries. 

***Regression***: A technique used to train an algorithm to understand the relationship between independent variables (input features) and a continuous dependent variable (output). The errors in measurement or "noise" are assumed to be normally distributed with a mean of 0 and some unknown standard deviation. We want to use the trained algorithm to predict the expected value of the output as a function of the input features. 

***Linear regression***: One of the basic types of regression technique which plots a straight line of best fit within data points to minimise error between the line and the data points. It is assumed that the relationship between independent and dependent variables is linear when this technique is used. 

When there are multiple independent variables, this is called multiple linear regression and the idea is as follows: 

  * _Predicted Value = weight<sub>1</sub> * Feature<sub>1</sub> + weight<sub>2</sub> * Feature<sub>2</sub> + ... + weight<sub>p</sub> * Feature<sub>p</sub>_

The algorithm learns the optimal weights or coefficients of the input features as an iterative process, usually through a method such as gradient descent. 

Now that we know the foundation, how does an algorithm make predictions when the relationship between the independent and dependent variables are ***non-linear***? We use locally weighted regression (LOWESS).

***Locally weighted regression***: A technique that modifies the linear regression to make it fit non-linear functions. The non-parametric algorithm fits a linear model to localized subsets of the data to build a function that describes the variation in the data, point by point. Basically, there are many small local functions rather than one global function. The weights (parameters) of each training data point are computed locally (for each query data point) based on the distance of the training points from the query point, using a kernel function. This makes minimizing the errors easier and means the parameters are unique to each data point (instead of fitting a fixed set of parameters to the data). 

We use LOWESS when the distribution of data is non-linear and the number of features is smaller, as the process is highly exhaustive and computationally expensive. It also requires a large/dense dataset to produce good results since LOWESS relies on the local data structure when fitting the model.

# Python Code and Visualizations of Locally Weighted Regression

Below is the code for a basic LOWESS regression function.

```Python
# after inserting the proper libraries...
def kernel_function(xi,x0,kern, tau): 
    return kern((xi - x0)/(2*tau))

def weights_matrix(x,kern,tau):
  n = len(x)
  return np.array([kernel_function(x,x[i],kern,tau) for i in range(n)]) 

def lowess(x, y, kern, tau=0.05):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    # tau is a hyper-parameter
    n = len(x)
    yest = np.zeros(n)
    
    #Initializing all weights from the bell shape kernel function       
    #Looping through all x-points
    
    w = weights_matrix(x,kern,tau)    
    
    #Looping through all x-points and fitting the local regression model
    for i in range(n):
        weights = w[:, i]
        lm.fit(np.diag(w[:,i]).dot(x.reshape(-1,1)),np.diag(w[:,i]).dot(y.reshape(-1,1)))
        yest[i] = lm.predict(x[i].reshape(-1,1)) 

    return yest
```
Here are some various kernels we can use.
```Python
# Tricubic Kernel
def tricubic(x):
  return np.where(np.abs(x)>1,0,(1-np.abs(x)**3)**3)   
  
# Epanechnikov Kernel
def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2)) 
  
# Quartic Kernel
def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2) 
```
Next, I created noisy, non linear data and ran the LOWESS function on it, using different kernels. I also ran a weak, linear regression model on the data to show the difference between LOWESS and linear regression on non linear data.

```Python
x = np.linspace(0,2,201)
noise = np.random.normal(loc = 0, scale = .2, size = len(x))
y = np.sin(x**2 * 1.5 * np.pi ) 
y_noise = y + noise
kernel = tricubic
kernel2 = Epanechnikov
kernel3 = Quartic
lm = LinearRegression()
# run lowess with different kernels
yest = lowess(x,y,kernel,0.04)
yest2 = lowess(x,y,kernel2,0.04)
yest3 = lowess(x,y,kernel3,0.04)

# Creating a weak learner using linear regression on nonlinear data
xlr = x.reshape(-1,1)
y_noiselr = y_noise.reshape(-1,1)
lr = LinearRegression()
lr.fit(xlr,y_noiselr)
yhat_lr = lr.predict(xlr)
```
Here is the result. I used Matplotlib to create this graph. Clearly, we cannot use linear regression on data like this and we must use something like locally weighted regression instead. In this example, we can't see much difference between the various kernels, but perhaps that's not a bad thing, it seems that any one we choose will give us good results. 

<img src="project1_graphs/lowess-intro.png" width="1200" height="300" /> 

Now, I did the same process, but used real data in place of the simulated sin function. I used the Cars.csv dataset provided in class. I used a slightly different variation of the LOWESS function that uses interpolation. Again, not mmuch difference bewteen the kernels, but clearly LOWESS does a better job of modeling the data than linear regression does.

```Python
def lowess_reg(x, y, xnew, kern, tau):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    # IMPORTANT: we expect x to the sorted increasingly
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        #theta = linalg.solve(A, b) # A*theta = b
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 
    f = interp1d(x, yest,fill_value='extrapolate')
    return f(xnew)
```

<img src="project1_graphs/lowess-cardata.png" width="1200" height="300" /> 

#### Adapting LOWESS function to be scikit-learn compatible
Next, in class we were shown how to transform the LOWESS function to be a scikit-learn compatible class so we can use .fit() and .predict(). This makes it more simple and easier to use. 

```Python
class Lowess:
    def __init__(self, kernel = Gaussian, tau=0.05):
        self.kernel = kernel
        self.tau = tau
    
    def fit(self, x, y):
        kernel = self.kernel
        tau = self.tau
        self.xtrain_ = x
        self.yhat_ = y

    def predict(self, x_new):
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_

        w = weights_matrix(x,x_new,self.kernel,self.tau)

        if np.isscalar(x_new):
          lm.fit(np.diag(w).dot(x.reshape(-1,1)),np.diag(w).dot(y.reshape(-1,1)))
          yest = lm.predict([[x_new]])[0][0]
        elif len(x.shape)==1:
          n = len(x_new)
          yest_test = np.zeros(n)
          #Looping through all x-points
          for i in range(n):
            lm.fit(np.diag(w[i,:]).dot(x.reshape(-1,1)),np.diag(w[i,:]).dot(y.reshape(-1,1)))
            yest_test[i] = lm.predict(x_new[i].reshape(-1,1))
        else:
          n = len(x_new)
          yest_test = np.zeros(n)
          #Looping through all x-points
          for i in range(n):
            lm.fit(np.diag(w[i,:]).dot(x),np.diag(w[i,:]).dot(y.reshape(-1,1)))
            yest_test[i] = lm.predict(x_new[i].reshape(1,-1))
        return yest_test
```
Using this newly created LOWESS class, I created another noisy, non linear function and ran LOWESS on the data with different kernels. I also computed the mean squared error for each kernel:

Epanechnikov - 0.05732
Gaussian - 0.08125
Tricubic - 0.05939
Quartic - 0.05970

<img src="project1_graphs/lowess-kernels-sin.png" width="1200" height="500" /> 

Finally, I did a k-fold cross validation to choose which kernel to use and also what tau value. Tau is the hyper-parameter that determines the width of the kernel or how big the neighborhood is around the local regression point when calculating weights. The kernel is the function that is used to calculate the weight of each training point based on its distance from the local test point.

```Python
kf = KFold(n_splits=10,shuffle=True,random_state=123)
mse_test_lowess_ep = []
mse_test_lowess_gs = []
mse_test_lowess_tr = []
mse_test_lowess_qc = []

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  xtest = x[idxtest]
  ytrain = ynoisy[idxtrain]
  ytest = ynoisy[idxtest]

  model_lw_ep = Lowess(kernel=Epanechnikov,tau=0.02)
  model_lw_gs = Lowess(kernel=Gaussian,tau=0.02)
  model_lw_tr = Lowess(kernel=Tricubic,tau=0.02)
  model_lw_qc = Lowess(kernel=Quartic,tau=0.02)

  model_lw_ep.fit(xtrain,ytrain)
  model_lw_gs.fit(xtrain,ytrain)
  model_lw_tr.fit(xtrain,ytrain)
  model_lw_qc.fit(xtrain,ytrain)

  mse_test_lowess_ep.append(mse(ytest,model_lw_ep.predict(xtest)))
  mse_test_lowess_gs.append(mse(ytest,model_lw_gs.predict(xtest)))
  mse_test_lowess_tr.append(mse(ytest,model_lw_tr.predict(xtest)))
  mse_test_lowess_qc.append(mse(ytest,model_lw_qc.predict(xtest)))

print('The validated MSE for Lowess with Epanechnikov kernel is : '+str(np.mean(mse_test_lowess_ep)))
print('The validated MSE for Lowess with Gaussian kernel is : '+str(np.mean(mse_test_lowess_gs)))
print('The validated MSE for Lowess with Tricubic kernel is : '+str(np.mean(mse_test_lowess_tr)))
print('The validated MSE for Lowess with Quartic kernel is : '+str(np.mean(mse_test_lowess_qc)))
```
The output was:
  The validated MSE for Lowess with Epanechnikov kernel is : 0.05441352570505785
  The validated MSE for Lowess with Gaussian kernel is : 0.07647261731338811
  The validated MSE for Lowess with Tricubic kernel is : 0.05450102935602012
  The validated MSE for Lowess with Quartic kernel is : 0.05508064325500174
  
I also performed k-fold validation to optimize the tau paramter.

```Python
kf = KFold(n_splits=10,shuffle=True,random_state=123)
mse_test_lowess = []
taus = []
t_range = np.linspace(0.01,1,num=100)

for t in t_range:
  model_lw = Lowess(kernel=Epanechnikov,tau=t)

  for idxtrain, idxtest in kf.split(x):
    xtrain = x[idxtrain]
    xtest = x[idxtest]
    ytrain = ynoisy[idxtrain]
    ytest = ynoisy[idxtest]    

    model_lw.fit(xtrain,ytrain)
    mse_test_lowess.append(mse(ytest,model_lw.predict(xtest)))
    taus.append(t)
idx = np.argmin(mse_test_lowess)
print('The validated MSE for Lowess is : '+str(np.mean(mse_test_lowess)))
print('The optimal tau is ' + str(taus[idx]) + '; and its corresponding MSE is ' + str(np.min(mse_test_lowess)))
```
This was the output: 
  The validated MSE for Lowess is : 0.458179157026943
  The optimal tau is 0.02; and its corresponding MSE is 0.04005951952523736
  
The final graph is using the optimized LOWESS model on the noisy sin function.

<img src="project1_graphs/lowess-optimized.png" width="800" height="400" /> 

The full Python notebook is linked here: [Project 1 Python Notebook](https://colab.research.google.com/drive/1cpUCM7N4mryhDvETJjrGiI5MNH44wlwR#scrollTo=1HGeqENmxPgk) 

[Back to Project Index](https://sofia-huang.github.io/DATA441/)
