addprocs(3)
@everywhere using GLM, StatsBase, DataFrames,MultivariateStats
@everywhere using Combinatorics, Parameters, DataFramesMeta, Lazy, Plots
include("./non_farm_payroll_prediction/code/jcode/functions.jl")

### Run the Exercise
const sFile = "./non_farm_payroll_prediction/data/data.csv" # data location
const sStart_s = "01/01/15" # start out of sample
const iSymbol = :NFP # dependent variable
const vSymbol = [:Date, :NFP, :NFP_bb_median] # remove from dataset (non-numerical and dep. var.)
const bHeader_sd = true # data header
const H = [1,2] # horizons
const iBest = 16 # Check the best x variables among the two criteria

dfData = readtable(sFile, header = bHeader_sd)
const iStart = find(dfData[:Date] .== sStart_s)[1]

mX,vNames = get_independent(dfData,vSymbol)
vY = convert(Array,dfData[iSymbol])

U = UnivariateSelection(mX,vY,vNames,H,iStart,iBest)
writecsv("./non_farm_payroll_prediction/data/univariate_selection_var_names.csv", U.vNames)

# sPath folder is the folder where the results are saved and has to be created
const sPath = "./non_farm_payroll_prediction/data/comb_$(size(U.vVar,1))/"
const vThreshold = moving_average(vY,6)
par_get_best_comb(mX,vY,H,U,sPath,vThreshold)
