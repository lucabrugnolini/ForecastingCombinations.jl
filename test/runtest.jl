using NFP
const sStart_s = "01/01/15" # start out of sample
const iSymbol = :NFP # dependent variable
const vSymbol = [:Date, :NFP, :NFP_bb_median] # remove from dataset (non-numerical and dep. var.)
const H = [1,2] # horizons
const iBest = 16 # Check the best x variables among the two criteria
const ncomb_load = 20 # max comb to use for the forecast
const iProcs = 3 # number of processors

dfData = readtable(joinpath(Pkg.dir("NFP"),"test","data.csv"), header = true)
const iStart = find(dfData[:Date] .== sStart_s)[1]

sforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load,iProcs)