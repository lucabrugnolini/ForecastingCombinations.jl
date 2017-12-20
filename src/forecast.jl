using GLM, StatsBase, DataFrames,MultivariateStats
using Combinatorics, Parameters, DataFramesMeta, Lazy, Plots
include("./non_farm_payroll_prediction/code/jcode/functions.jl")

###############################################################################
############# Load results to select the best model
###############################################################################
# Set the path for the data and combinations criteria
const path_d = "./non_farm_payroll_prediction/data/comb_20/"
const sFile = "./non_farm_payroll_prediction/data/data.csv"
const bHeader_sd = true

# Set simulation start
const sStart_s = "01/01/15"

# Set variables toremove from dataset and to use as dependent variable
const vSymbol = [:Date, :NFP, :NFP_bb_median]
const iSymbol = :NFP

# Set horizons to forecast
const H = [1,2]
const rm_var = size(vSymbol,1)

# Set number of combination selected and how many to load (in case one want to check partial results)
const ncomb = 20
const ncomb_load = 20

# Load the dataset
dfData = readtable(sFile, header = bHeader_sd)
const iStart = find(dfData[:Date] .== sStart_s)[1]

# Load selected variable names
const choosen_var = string.(readcsv("./non_farm_payroll_prediction/data/univariate_selection_var_names.csv"))

# Get the clean dataset, the variable names and the independent variables
mX,vNames = get_independent(dfData,vSymbol)
vY = convert(Array,dfData[iSymbol])

# Load scores from the selected variable procedure
# MAE criteria
vMae = H
comb_identifier = "Models" # preallocate comb_identifier for concatenation
for comb = 1:ncomb_load
    comb_index = collect(combinations(1:size(choosen_var,1), comb))
    comb_index = transpose(hcat(comb_index...))
    dfMae = readtable(path_d*"vMae_best_$(ncomb)_comb_$comb.csv", header = true)
    comb_identifier = @>> map(string,names(dfMae)).*"_comb$comb" vcat(comb_identifier)
    vMae = @>> convert(Array,dfMae) hcat(vMae)
    println(size(vMae))
end
best_vMae = [[findmin(vMae[j,2:end])[2]+1 findmin(vMae[j,2:end])[1]] for j in 1:size(vMae,1)] # +1 exclude first column and reset indexes
best_vMae = vcat(best_vMae...)
best_comb_vMae = @>> comb_identifier[convert(Array{Int64},best_vMae[:,1])] map(x -> [Symbol(split(x,"_c")[1]) split(x,"_comb")[2]])
best_comb_vMae = vcat(best_comb_vMae...)
factor_in_vMae = [sum(collect(string(best_comb_vMae[i,1])) .== '_') for i in 1:size(best_comb_vMae,1)]

# RMSE criteria
vRmse = H
comb_identifier = "Models" # preallocate comb_identifier for concatenation
for comb in 1:ncomb_load
    comb_index = collect(combinations(1:size(choosen_var,1), comb))
    comb_index = transpose(hcat(comb_index...))
    dfRmse = readtable(path_d*"vRmse_best_$(ncomb)_comb_$comb.csv", header = true)
    comb_identifier = @>> map(string,names(dfRmse)).*"_comb$comb" vcat(comb_identifier)
    vRmse = @>> convert(Array,dfRmse) hcat(vRmse)
    println(size(vRmse))
end
best_vRmse = [[findmin(vRmse[j,2:end])[2]+1 findmin(vRmse[j,2:end])[1]] for j in 1:size(vRmse,1)] # +1 exclude first column and reset indexes
best_vRmse = vcat(best_vRmse...)
best_comb_vRmse = @>> comb_identifier[convert(Array{Int64},best_vRmse[:,1])] map(x -> [Symbol(split(x,"_c")[1]) split(x,"_comb")[2]])
best_comb_vRmse = vcat(best_comb_vRmse...)
factor_in_vRmse = [sum(collect(string(best_comb_vRmse[i,1])) .== '_') for i in 1:size(best_comb_vRmse,1)]

######################################################################################
############################ Out-of-sample 
#####################################################################################
# Setting up the model:
dict = Dict(1 => 1, 2 => 2)

l_plot = plot(layout = grid(2,1))
l_plot_prob = plot(layout = grid(2,1))

line_st = Dict(best_comb_vMae => (:solid,"green",2), best_comb_vRmse => (:dash, "red",2))
counter = 0
for best_comb = (best_comb_vMae, best_comb_vRmse)
    counter += 1
    if best_comb == best_comb_vMae
        factor_in = factor_in_vMae
    else best_comb == best_comb_vRmse
        factor_in = factor_in_vRmse
    end
    model_index = Array{Vector}(size(best_comb,1))
    variables = Array{Vector}(size(best_comb,1))
    mFore    = zeros(size(vY,1),size(H,1))
    count = 0
    for i = 1:size(best_comb,1)
        comb_index = collect(combinations(1:size(choosen_var,1), parse(Int64,best_comb[i,2])))
        comb_index = transpose(hcat(comb_index...))
        if @> sum(collect(string(best_comb[i,1])) .== '_') == 0
            model_index[i] = comb_index[parse(Int64,split(string(best_comb[i,1]),"x")[2]),:]
            variables[i] = choosen_var[comb_index[parse(Int64,split(string(best_comb[i,1]),"x")[2]),:]]
        else
            model_index[i] = comb_index[parse(Int64,split(string(best_comb[i,1]),r"x|_")[2]),:]
            variables[i] = choosen_var[comb_index[parse(Int64,split(string(best_comb[i,1]),r"x|_")[2]),:]]
        end
        model_index[i] = map(x-> find(x .== map(string,names(dfData)))[1],variables[i])-rm_var
        println("Starting models with $(variables[i]) variables and factor $(Bool(factor_in[i]))")
        h = H[i]
        count += 1
        if factor_in[i] == 1 # test if best use factors
            @inbounds mFore[:,count] = out_of_sample_forecast_pca(mX,vY,iStart,h,model_index[i],1:size(mX,2))
        else
            @inbounds mFore[:,count] = out_of_sample_forecast(mX[:,model_index[i]],vY,iStart,h)
        end
        horiz_col = dict[h] 
        best_comb == best_comb_vMae && plot!(l_plot,dfData[:Date][iStart+h:end],vY[iStart+h:end], color = "black", 
            subplot = horiz_col)
          plot!(l_plot,dfData[:Date][iStart+h:end],mFore[iStart+h:end,horiz_col], line = (line_st[best_comb]), 
            title = "Horizon $h", subplot = horiz_col, legend = false, ylabel = "NFP")
    end
end
plot(l_plot)


