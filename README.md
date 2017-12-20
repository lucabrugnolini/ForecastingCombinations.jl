# NFP.jl
Forecasting Variables using a combinatoric approach and exploiting parallel computing in Julia.

## Installation
```julia
Pkg.clone("https://github.com/lucabrugnolini/NFP.jl")
```
The package requires the following packages installed: GLM, StatsBase, DataFrames, MultivariateStats, Combinatorics, Parameters, DataFramesMeta, Lazy, Plots

This can be done via: `Pkg.add("name_of_the_package")`


## Introduction
The package exploits all the variables contained in a balanced dataset. In a first step, the procedure selects the best n-variables using two different criteria (mean absolute error and root mean squared error). 

In the second step, it the code uses all the possible combination of the selected variables and chooses the best available model according to the two criteria. The total number of combinations are <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{C}&space;=&space;2^K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{C}&space;=&space;2^K" title="\mathcal{C} = 2^K" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K" title="K" /></a> is the number of variable selected in the first step of the process. Also, for each model an augmented counterpart is estimated adding the first principal component of the entire dataset. The total number of estimated model are <a href="https://www.codecogs.com/eqnedit.php?latex=\bar{C}&space;=&space;(2H(T-T^s))2^K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bar{C}&space;=&space;(2H(T-T^s))2^K" title="\bar{C} = (2H(T-T^s))2^K" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=H" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H" title="H" /></a> is the horizons and <a href="https://www.codecogs.com/eqnedit.php?latex=T^s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T^s" title="T^s" /></a> is the pre-estimation sample.

## Example
Forecasting US non-farm-payroll one and two months ahead `H = [1,2]` using a dataset which comprises around 100 US variables from McCracke and Ng (2015). The code selects the best 16 variables `iBest` in an out-of-sample forcast starting in January 2015 according to the MAE and RMSE. Then, it exploit a quad-core processor to select the best model among all the possible combinations (<a href="https://www.codecogs.com/eqnedit.php?latex=K=20" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K=20" title="K=20" /></a>). 

```julia
addprocs(3)
@everywhere using NFP, DataFrames
@everywhere const sStart_s = "01/01/15" # start out of sample
@everywhere const iSymbol = :NFP # dependent variable
@everywhere const vSymbol = [:Date, :NFP, :NFP_bb_median] # remove from dataset (non-numerical and dep. var.)
@everywhere const H = [1,2] # horizons
@everywhere const iBest = 16 # Check the best x variables among the two criteria
@everywhere const ncomb_load = 20 # max comb to use for the forecast

@everywhere dfData = readtable(joinpath(Pkg.dir("NFP"),"test","data.csv"), header = true)
@everywhere const iStart = find(dfData[:Date] .== sStart_s)[1]

l_plot,r = sforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load)
l_plot,r = fforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load)

# Plot the forecasts
l_plot

rmprocs(2:4)

```

There are two primary functions:
1. `sforecast` which computes the two steps variable selection
2. `fforecast` which uses results previously stored (to run after running at least ones `sforecast`)
