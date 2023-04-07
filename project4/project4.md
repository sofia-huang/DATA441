## Project 4
# Generalized Additive Model vs Nadaraya-Watson regressor
#### Goal: Generate 1-D (input) data and simulate a noised output of a rapidly oscillating function on the interval [0,20]; you will fit both regressors on a train set and measure their error on a test set. Show a plot comparing the two reconstructions of the gound truth function. Also, you will compare the 10-fold cross validated root mean square error of the two regressors on the "concrete.csv" data set.


First, I created the 1-D data to simulate a noisy output of a rapidly oscillating function. I also created train and test sets from the function.
```Python
x = np.linspace(0,20,500)
y = np.cos((x**2)/12) + np.random.normal(0, 0.2, len(x))

xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.2,shuffle=True,random_state=123)

xtrain = xtrain.reshape((-1,1))
xtest = xtest.reshape((-1,1))
ytrain = ytrain.reshape((-1,1))
```
Then, I fit the GAM on the training sets and used it to predict on the testing sets. 
```Python
gam = LinearGAM(n_splines=25).gridsearch(xtrain, ytrain)
yhat = gam.predict(xtest)
mse(ytest,yhat)
```
The mean squared error using GAM on the noisy cosine function data was **0.04346660296522676.**

I plotted the GAM's predictions against the true function and the result is below.

<img src="project4graphs/GAM1.png" width="800" height="600" /> 
