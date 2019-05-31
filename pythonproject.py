


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm



def mean_abs_percentage_error(test, forecast): 
    test, forecast = np.array(test), np.array(forecast)
    return np.mean(np.abs((test - forecast) / test)) * 100


def moving_average_method(train, test,value):
    #Moving average approach
    y_hat_avg = test.copy()
    windowsize=np.arange(len(train))
    windowsize=windowsize[1:]
    #try to find the lowest error with all possible windowsizes
    error=10000000000000.0
    for window in windowsize:
        y_hat_avg['moving_avg_forecast'] = train[value].rolling(window).mean().iloc[-1]
        mape = mean_abs_percentage_error(test[value], y_hat_avg.moving_avg_forecast)
        if mape<error:
            error=mape
            optimal_windowsize=window
    return error,optimal_windowsize


def simple_expo_smoothing(train, test,value):
    # Simple Exponential Smoothing
    y_hat_avg = test.copy()
    alphas=np.linspace(0,1,101)
    #try to find the lowest error with all possible alphas with two decimal
    error=100000000000
    for alpha in alphas:
        fit2 = SimpleExpSmoothing(np.asarray(train[value])).fit(smoothing_level=alpha,optimized=False)
        y_hat_avg['SES'] = fit2.forecast(len(test))
        mape = mean_abs_percentage_error(test[value], y_hat_avg.SES)
        if mape<error:
            error=mape
            optimal_alpha=alpha
    return error,optimal_alpha

def holt_winters_method(train, test,value,seasons):
    # Holt-Winters Method
    y_hat_avg = test.copy()
    fit1 = ExponentialSmoothing(np.asarray(train[value]) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
    y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
    mape=mean_abs_percentage_error(test[value], y_hat_avg.Holt_Winter)
    return mape

def holt_method(train,test,value):
    sm.tsa.seasonal_decompose(train[value],freq = 3).plot()
    result = sm.tsa.stattools.adfuller(train[value])
    # plt.show()
    y_hat_avg = test.copy()
    alphas=np.linspace(0,1,101)
    #try to find the lowest error with all possible alphas and slopes with two decimal
    error=1000000000000000
    for alpha in alphas:
        slopes=np.linspace(0,1,101)
        for slope in slopes:
            fit1 = holt_method(np.asarray(train[value])).fit(smoothing_level =alpha,smoothing_slope =slope)
            y_hat_avg['Holt'] = fit1.forecast(len(test))
            mape=mean_abs_percentage_error(test[value], y_hat_avg.Holt)
            if mape<error:
                optimal_alpha=alpha
                error=mape
                optimal_slope=slope
    return error,optimal_alpha,optimal_slope



#*************************** WORLD POPULATION **************************

dfWorlddata=pd.read_csv("World.csv",sep=";")

#In order to get train and test data we use Pareto Principle
size=len(dfWorlddata)
size_train=round(size*0.8)
train=dfWorlddata[0:size_train]
test=dfWorlddata[size_train:]

#moving_averages, simple exponential smoothing, holt-winters,Holt
errors=[0.0,0.0,0.0,0.0]
errors[0]=moving_average_method(train,test,"POPULATION")[0]
errors[1]=simple_expo_smoothing(train,test,"POPULATION")[0]
errors[2]=holt_winters_method(train,test,"POPULATION",seasons=10)
errors[3]=holt_method(train,test,"POPULATION")[0]

optimal_windowsize=moving_average_method(train,test,"POPULATION")[1]
ses_optimal_alpha=simple_expo_smoothing(train,test,"POPULATION")[1]
holt_optimal_alpha=holt_method(train,test,"POPULATION")[1]
holt_optimal_slope=holt_method(train,test,"POPULATION")[2]
 


if errors[0]==min(errors):
    #if moving average method gives the smallest MAPE
    forecast= dfWorlddata['POPULATION'].rolling(optimal_windowsize).mean().iloc[-1]
    print("We applied Holt method for World population data")
    print("Estimation of the last quarter of 2018 for World Population:",forecast[0])
elif errors[1]==min(errors):
    #if Simple Exponential Smoothing method gives  the smallest MAPE
    fit2 = SimpleExpSmoothing(np.asarray(dfWorlddata['POPULATION'])).fit(smoothing_level=ses_optimal_alpha,optimized=False)
    forecast=fit2.forecast(1)
    print("We applied Holt method for World population data")
    print("Estimation of the last quarter of 2018 for World Population:",forecast[0])
elif errors[2]==min(errors):
    #if Holt-Winters method gives  the smallest MAPE
    seasons = 10
    fit = ExponentialSmoothing( np.asarray(dfWorlddata['POPULATION']) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
    forecast= fit.forecast(1)
    print("We applied Holt method for World population data")
    print("Estimation of the last quarter of 2018 for World Population:",forecast[0])
else:
    #if Holt method gives  the smallest MAPE
    fit1 = holt_method(np.asarray(dfWorlddata["POPULATION"])).fit(smoothing_level =holt_optimal_alpha,smoothing_slope =holt_optimal_slope)
    forecast=fit1.forecast(1)
    print("We applied Holt method for World population data")
    print("Estimation of the last quarter of 2018 for World Population:",forecast[0])


growth_population=((forecast-dfWorlddata.POPULATION.iloc[-1])/dfWorlddata.POPULATION.iloc[-1])*100
print("World population's growth rate",growth_population[0])



#*************************** SOCIAL MEDIAS **************************

dfSomediadata=pd.read_csv("social_medias.csv",sep=";")

#In order to get train and test data we use Pareto Principle
size=len(dfSomediadata)
size_train=round(size*0.8)
train=dfSomediadata[0:size_train]
test=dfSomediadata[size_train:]

#*************************** INSTAGRAM **************************


#moving_averages, simple exponential smoothing, holt-winters,Holt
t_errors=[0.0,0.0,0.0,0.0]
t_errors[0]=moving_average_method(train,test,"FACEBOOK")[0]
t_errors[1]=simple_expo_smoothing(train,test,"FACEBOOK")[0]
t_errors[2]=holt_winters_method(train,test,"FACEBOOK",seasons=10)
t_errors[3]=holt_method(train,test,"FACEBOOK")[0]

t_optimal_windowsize=moving_average_method(train,test,"FACEBOOK")[1]
t_ses_optimal_alpha=simple_expo_smoothing(train,test,"FACEBOOK")[1]
t_holt_optimal_alpha=holt_method(train,test,"FACEBOOK")[1]
t_holt_optimal_slope=holt_method(train,test,"FACEBOOK")[2]
 


if t_errors[0]==min(t_errors):
    #if moving average method gives the smallest MAPE
    forecast_INSTAGRAM= dfSomediadata['INSTAGRAM'].rolling(t_optimal_windowsize).mean().iloc[-1]
    print("We applied Holt method for INSTAGRAM data")
    print("Estimation of the last quarter of 2018 for INSTAGRAM:",forecast_INSTAGRAM[0])
elif t_errors[1]==min(t_errors):
    #if Simple Exponential Smoothing method gives  the smallest MAPE
    fit2 = SimpleExpSmoothing(np.asarray(dfSomediadata['INSTAGRAM'])).fit(smoothing_level=t_ses_optimal_alpha,optimized=False)
    forecast_INSTAGRAM=fit2.forecast(1)
    print("We applied Holt method for INSTAGRAM data")
    print("Estimation of the last quarter of 2018 for INSTAGRAM:",forecast_INSTAGRAM[0])
elif t_errors[2]==min(t_errors):
    #if Holt-Winters method gives  the smallest MAPE
    seasons = 10
    fit = ExponentialSmoothing( np.asarray(dfSomediadata['INSTAGRAM']) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
    forecast_INSTAGRAM= fit.forecast(1)
    print("We applied Holt method for INSTAGRAM data")
    print("Estimation of the last quarter of 2018 for INSTAGRAM:",forecast_INSTAGRAM[0])
else:
    #if Holt method gives  the smallest MAPE
    fit1 = holt_method(np.asarray(dfSomediadata["INSTAGRAM"])).fit(smoothing_level =t_holt_optimal_alpha,smoothing_slope =t_holt_optimal_slope)
    forecast_INSTAGRAM=fit1.forecast(1)
    print("We applied Holt method for INSTAGRAM data")
    print("Estimation of the last quarter of 2018 for INSTAGRAM:",forecast_INSTAGRAM[0])


growth_INSTAGRAM=((forecast_INSTAGRAM-dfSomediadata.INSTAGRAM.iloc[-4])/dfSomediadata.INSTAGRAM.iloc[-4])*100
print("INSTAGRAM's growth rate",growth_INSTAGRAM[0])



#*************************** FACEBOOK **************************


#moving_averages, simple exponential smoothing, holt-winters,Holt
f_errors=[0.0,0.0,0.0,0.0]
f_errors[0]=moving_average_method(train,test,"FACEBOOK")[0]
f_errors[1]=simple_expo_smoothing(train,test,"FACEBOOK")[0]
f_errors[2]=holt_winters_method(train,test,"FACEBOOK",seasons=10)
f_errors[3]=holt_method(train,test,"FACEBOOK")[0]

f_optimal_windowsize=moving_average_method(train,test,"FACEBOOK")[1]
f_ses_optimal_alpha=simple_expo_smoothing(train,test,"FACEBOOK")[1]
f_holt_optimal_alpha=holt_method(train,test,"FACEBOOK")[1]
f_holt_optimal_slope=holt_method(train,test,"FACEBOOK")[2]
 


if f_errors[0]==min(f_errors):
    #if moving average method gives the smallest MAPE
    forecast_facebook= dfSomediadata['FACEBOOK'].rolling(f_optimal_windowsize).mean().iloc[-1]
    print("We applied Holt method for FACEBOOK data")
    print("Estimation of the last quarter of 2018 for FACEBOOK:",forecast_facebook[0])
elif f_errors[1]==min(f_errors):
    #if Simple Exponential Smoothing method gives  the smallest MAPE
    fit2 = SimpleExpSmoothing(np.asarray(dfSomediadata['FACEBOOK'])).fit(smoothing_level=f_ses_optimal_alpha,optimized=False)
    forecast_facebook=fit2.forecast(1)
    print("We applied Holt method for FACEBOOK data")
    print("Estimation of the last quarter of 2018 for FACEBOOK:",forecast_facebook[0])
elif f_errors[2]==min(f_errors):
    #if Holt-Winters method gives  the smallest MAPE
    seasons = 10
    fit = ExponentialSmoothing( np.asarray(dfSomediadata['FACEBOOK']) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
    forecast_facebook= fit.forecast(1)
    print("We applied Holt method for FACEBOOK data")
    print("Estimation of the last quarter of 2018 for FACEBOOK:",forecast_facebook[0])
else:
    #if Holt method gives  the smallest MAPE
    fit1 = holt_method(np.asarray(dfSomediadata["FACEBOOK"])).fit(smoothing_level =f_holt_optimal_alpha,smoothing_slope =f_holt_optimal_slope)
    forecast_facebook=fit1.forecast(1)
    print("We applied Holt method for FACEBOOK data")
    print("Estimation of the last quarter of 2018 for FACEBOOK:",forecast_facebook[0])


growth_FACEBOOK=((forecast_facebook-dfSomediadata.FACEBOOK.iloc[-4])/dfSomediadata.FACEBOOK.iloc[-4])*100
print("FACEBOOK's growth rate :",growth_FACEBOOK[0])


#*************************** COMPARING RATIOS **************************

print("Compairing the ratios...")

if growth_population>0:
    #Comparing INSTAGRAM estimate with World Population estimate
    if growth_INSTAGRAM>0:
        if growth_INSTAGRAM>growth_population+5:
            print("INSTAGRAM will have a meaningful growth rate because it is significantly higher than the growth rate of world population")
        elif growth_INSTAGRAM>growth_population+2:
            print("INSTAGRAM will have a meaningful growth rate because it is moderately higher than the growth rate of world population")
        else:
            print("Although INSTAGRAM will have a growth, it has not a meaningful growth rate when compared with world population growth rate")
    else:
        print("Although world population will grow in 2018, INSTAGRAM will lose their users")
    #Comparing Facebook estimate with World Population estimate
    if growth_FACEBOOK>0:
        if growth_FACEBOOK>growth_population+5:
            print("Facebook will have a meaningful growth rate because it is significantly higher than the growth rate of world population")
        elif growth_FACEBOOK>growth_population+2:
            print("Facebook will have a meaningful growth rate because it is moderately higher than the growth rate of world population")
        else:
            print("Although Facebook will have a growth, it has not a meaningful growth rate when compared with world population growth rate")
    else:
        print("Although world population will grow in 2018, Facebook will lose their users")   
else:
    #Comparing INSTAGRAM estimate with World Population estimate
    if growth_INSTAGRAM<0:
        if growth_INSTAGRAM<growth_population-5:
            print("INSTAGRAM will have a decreasing growth rate because it is significantly lower than the growth rate of world population")           
        elif growth_INSTAGRAM<growth_population-2:
            print("INSTAGRAM will have a decreasing growth rate because it is moderately lower than the growth rate of world population")
        else:
            print("Although INSTAGRAM will have a decreasing growth rate, it has not a meaningful growth rate when compared with world population growth rate")
    else:
        print("Although world population will decrease in 2018, INSTAGRAM will grow")
    #Comparing Facebook estimate with World Population estimate
    if growth_FACEBOOK<0:
        if growth_FACEBOOK<growth_population-5:
            print("Facebook will have a decreasing growth rate because it is significantly lower than the growth rate of world population")           
        elif growth_FACEBOOK<growth_population-2:
            print("Facebook will have a decreasing growth rate because it is moderately lower than the growth rate of world population")
        else:
            print("Although Facebook will have a decreasing growth rate, it has not a meaningful growth rate when compared with world population growth rate")
    else:
        print("Although world population will decrease in 2018, Facebook will grow")   