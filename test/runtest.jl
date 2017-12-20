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

sforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load)
rmprocs(2:4)