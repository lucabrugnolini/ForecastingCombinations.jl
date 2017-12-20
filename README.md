# NFP.jl
Forecasting Variables using a combinatoric approach and exploiting parallel computing in Julia.

## Installation
```julia
Pkg.clone("https://github.com/lucabrugnolini/NFP.jl")
```

## Note
The package exploits all the variables contained in a balanced dataset. In a first step, the procedure selects the best n-variables using two different criteria (mean absolute error and root mean squared error). 

In the second step, it the code uses all the possible combination of the selected variables and chooses the best available model according to the two criteria. The total number of combinations are <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{C}&space;=&space;2^K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{C}&space;=&space;2^K" title="\mathcal{C} = 2^K" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K" title="K" /></a> is the number of variable selected in the first step of the process.

## Example
Forecasting US non-farm-payroll one and two months ahead.
```julia
addprocs(3)
@everywhere using NFP, DataFrames
@everywhere const sStart_s = "01/01/15" # start out of sample
@everywhere const iSymbol = :NFP # dependent variable
@everywhere const vSymbol = [:Date, :NFP, :NFP_bb_median] # remove from dataset (non-numerical and dep. var.)
@everywhere const H = [1,2] # horizons
@everywhere const iBest = 3 # Check the best x variables among the two criteria
@everywhere const ncomb_load = 3 # max comb to use for the forecast

@everywhere dfData = readtable(joinpath(Pkg.dir("NFP"),"test","data.csv"), header = true)
@everywhere const iStart = find(dfData[:Date] .== sStart_s)[1]

l_plot = sforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load)

rmprocs(2:4)

```
