addprocs(3)
@everywhere using NFP, DataFrames, GLM, CSV
@everywhere const sStart_s = "01/01/15" # start out of sample
@everywhere const iSymbol = :NFP # dependent variable
@everywhere const vSymbol = [:Date, :NFP, :NFP_bb_median] # remove from dataset (non-numerical and dep. var.)
@everywhere const H = [1,2] # horizons
@everywhere const iBest = 3 # Check the best x variables among the two criteria
@everywhere const ncomb_load = 3 # max comb to use for the forecast

@everywhere mae(vX::Vector,vY::Vector) = mean(abs.(vX-vY))      ## MAE loss function 
@everywhere rmse(vX::Vector,vY::Array) = sqrt(mean((vX-vY).^2)) ## RMSE loss function

@everywhere const fLoss = [mae rmse] ## Vector of loss functions 

@everywhere dfData = CSV.read(joinpath(Pkg.dir("NFP"),"test","data.csv"), header = true)
@everywhere const iStart = find(dfData[:Date] .== sStart_s)[1]

# For NFP level forecast
l_plot,r = sforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load,l)
l_plot,r= fforecast(dfData,vSymbol,iSymbol,H,iStart,iBest,ncomb_load,l)

# See the forecasts plot
l_plot

rmprocs(2:4)