# NFP.jl
Forecasting using a parallel combinatoric approach.

## Installation
```julia
Pkg.clone("https://github.com/lucabrugnolini/NFP.jl")
```


## Introduction
Given a (balaanced) dataset of _K_ macroeconomic variables, the objective is to select the best model to predict future values of a target variable. The selection procedure consists in (i) select the best _iBest_ variables according to several out-of-sample criteria and then use these variables in models that use their combination. More specifically:

1. the procedure selects the best `iBest` variables using two different criteria (mean absolute error and root mean squared error). This selection step is univariate, i.e. the variables are chosen by running a simple regression of the target variable on each variable of the dataset. 

2. the `iBest` variables are combined into set of _2, 3, ..., iBest_ variables. For each of these sets, the model is estimated and then avaluated out-of-sample. The best model is the one with the lowest out-of-sample `MSE`. We also augment each model with the first principal component of all variable in the dataset. Thus, there are a total of _2 (2^iBest)_ models. 

The complexity is _O((T-Ts)*2^iBest)_ where _T_ is the sample size, _Ts_ is the number of observation in the initial estimation window. 

## Example
Forecasting US non-farm-payroll one and two months ahead `H = [1,2]` using a dataset has 100 US variables and are taken from McCracke and Ng (2015). `iBest` is set to 16. The code below is an example of parallelization on `N_CORE`. 


```julia
addprocs(N_CORE)
@everywhere using NFP, DataFrames
@everywhere const sStart_s = "01/01/15"                   ## This is the beginning of the out-of-sample window
@everywhere const iSymbol = :NFP                          ## Target variable
@everywhere const vSymbol = [:Date, :NFP, :NFP_bb_median] ## Variables to be removed from the dataset (non-numerical and dep. var.)
@everywhere const H = [1,2]                               ## Out-of-sample horizon
@everywhere const iBest = 16                              ## iBest
@everywhere const ncomb_load = iBest                      ## TODO: remove this option

@everywhere dfData = readtable(joinpath(Pkg.dir("NFP"),"test","data.csv"), header = true)
@everywhere const iStart = find(dfData[:Date] .== sStart_s)[1]


l_plot,r = sforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load)
l_plot,r = fforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load)

# Plot the forecasts
l_plot

## Remove process added
rmprocs(2:N_CORE)

```

There are two primary functions:
1. `sforecast` which computes the two steps variable selection
2. `fforecast` which uses results previously stored (to run after running at least ones `sforecast`)
