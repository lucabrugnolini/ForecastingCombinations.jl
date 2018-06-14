# NFP.jl
Forecasting using a parallel combinatoric approach.

## Installation
```julia
Pkg.clone("https://github.com/lucabrugnolini/NFP.jl")
```

## Documentation
The procedure is described in [Brugnolini L. (2018)](https://lucabrugnolini.github.io/publication/forecasting_inflation.pdf). The application in the paper is on predicting the probability of having inflation around the European Central Bank's target. 


## Introduction
Given a (balanced) dataset of _K_ macroeconomic variables, the objective is to select the best model to predict future values of a target variable. The selection procedure consists in (i) select the best _iBest_ variables according to several out-of-sample criteria and then use these variables in models that use their combination. More specifically:

1. the procedure selects the best `iBest` variables using two different criteria (mean absolute error (MAE) and root mean squared error (RMSE) included in the model as a vector _fLoss_ of functions in the form _f(Prediction,TrueValue)_). This selection step is univariate, i.e. the variables are chosen by running a simple out-of-sample  regression of the target variable on each variable of the dataset. 

2. the `iBest` variables are combined into set of _2, 3, ..., iBest_ variables. For each of these sets, the model is estimated and then avaluated out-of-sample. The best model is the one with the lowest out-of-sample `MSE`. We also augment each model with the first principal component of all variable in the dataset. Thus, there are a total of _2 (2^iBest)_ models. 

The complexity is _O((T-Ts)*2^iBest)_ where _T_ is the sample size, _Ts_ is the number of observation in the initial estimation window. 

## Example
Forecasting US non-farm-payroll one and two months ahead `H = [1,2]` using a dataset containing 116 US variables taken from [McCracken and Ng (2015)](https://amstat.tandfonline.com/doi/abs/10.1080/07350015.2015.1086655). `iBest` is set to 16. The code below is an example of parallelization on `N_CORE`. 


```julia
addprocs(N_CORE)
@everywhere using NFP
@everywhere using CSV, DataFrames, GLM

@everywhere mae(vX::Vector,vY::Vector) = mean(abs.(vX-vY))      ## MAE loss function 
@everywhere rmse(vX::Vector,vY::Array) = sqrt(mean((vX-vY).^2)) ## RMSE loss function

@everywhere const fLoss = [mae rmse] ## Vector of loss functions 

@everywhere const sStart_s = "01/01/15"                   ## This is the beginning of the out-of-sample window
@everywhere const iSymbol = :NFP                          ## Target variable
@everywhere const vSymbol = [:Date, :NFP, :NFP_bb_median] ## Variables to be removed from the dataset (non-numerical and dep. var.)
@everywhere const H = [1,2]                               ## Out-of-sample horizon
@everywhere const iBest = 16                               ## iBest
@everywhere const ncomb_load = iBest                      ## TODO: remove this option

@everywhere const dfData = CSV.read(joinpath(Pkg.dir("NFP"),"test","data.csv"), header = true)
@everywhere const iStart = find(dfData[:Date] .== sStart_s)[1]

## computes the two steps variable selection
l_plot,r = sforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load,fLoss)
## `fforecast` uses results previously stored with `sforecast`
l_plot,r = fforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load,fLoss)

# Plot the forecasts
l_plot

## Remove process added
rmprocs(2:N_CORE)

```

![Alt Text](/home/lbrugnol/recesion.png)

In case one is interested in predicting probabilities, as the US probability of recession, simply include a link function in `sforecast` or `fforecast`, and use a binary variable as dependent variable.

```julia
@everywhere const l = ProbitLink() # link function for probability model

## computes the two steps variable selection
l_plot,r = sforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load,l,fLoss)
## `fforecast` uses results previously stored with `sforecast`
l_plot,r = fforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load,l,fLoss)
```

## References
Brugnolini L. (2018) "Forecasting Deflation Probability in the EA: A Combinatoric Approach." _Central Bank of Malta Working Paper_, 01/2018.

McCracken, Michael W., and Serena Ng.(2016) "FRED-MD: A Monthly Database for Macroeconomic Research." Journal of Business & Economic Statistics 34.4:574-589.


