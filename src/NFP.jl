module NFP
using GLM, StatsBase, DataFrames,MultivariateStats
using Combinatorics, Parameters, DataFramesMeta, Lazy, Plots

type Loss
    mVar::Array{Int64}
    mScore::Array{Float64}
    mNames::Array{String}
end

function Loss(m::Array,mScore::Array,iBest::Int64,vNames::Vector)
    mBest = m[:,1:iBest]
    mBscore =  mScore[mBest]
    mBnames = vNames[mBest]
    return Loss(mBest,mBscore,mBnames)
end

type Score
    mae::Loss
    rmse::Loss
end

immutable UnivariateSelection
    vVar::Vector{Int64}
    vNames::Vector{String}
    s::Score
end

function UnivariateSelection(mX::Array,vY::Vector,vNames,H::Vector{Int64},iStart::Int64,iBest::Int64)
    mFore,mMae,mRmse = get_score(mX,vY,H,iStart)
    mMAE  = get_variable_idx(mMae::Array,false)
    mRMSE = get_variable_idx(mRmse::Array,false)
    
    mMAE_b = Loss(mMAE,mMae,iBest,vNames)
    mRMSE_b = Loss(mRMSE,mRmse,iBest,vNames)
    
    s = Score(mMAE_b,mRMSE_b)
    
    vBest = get_best(mMAE_b.mVar,mRMSE_b.mVar,iBest)
    vBest_names = vNames[vBest]
    
    return UnivariateSelection(vBest,vBest_names,s)
end

function get_score(mX::Array{Float64,2},vY::Vector,H::Vector{Int64},iStart::Int64)
    T,K = size(mX)::Tuple{Int64,Int64}
    P = size(H,1)
    # Out of sample
    mFore = zeros(T,P,K)
    mMae  = zeros(P,K)
    mRmse = zeros(P,K)
    @simd for j = 1:K
        println(j)
        mFore_s   = zeros(T,P)
        vMae  = zeros(P)
        vRmse = zeros(P)
        count = 0
        for h = H
            count = count+1
            mFore_s[:,count] = out_of_sample_forecast(mX[:,j:j],vY,iStart,h)
            vMae[count]    = mae(mFore_s[iStart+h:end,count],vY[iStart+h:end])
            vRmse[count]   = rmse(mFore_s[iStart+h:end,count],vY[iStart+h:end])
        end
        mFore[:,:,j] = mFore_s
        mMae[:,j]    = vMae
        mRmse[:,j]   = vRmse
    end
    return mFore,mMae, mRmse
end

function out_of_sample_forecast(mX::Array{Float64,2},vY::Vector,iStart::Int64,h::Int64)
    T,K = size(mX)::Tuple{Int64,Int64}
    vFore = zeros(T)
    for j = iStart:T-h
        mX_n = view(mX,1:j,1:K)
        vY_n = view(vY,(1:j))
        t = size(vY_n,1)
        mXX_n = hcat(ones(t-h,1), mX_n[1:t-h,1:K])
        vYY_n = vY_n[h+1:t]
        md = fit(LinearModel,mXX_n,vYY_n)
        iXX_n = hcat(ones(1,1), view(mX_n,t:t,1:K))
        vFore[j+h] = first(GLM.predict(md,iXX_n))
    end
    return vFore
end

mae(vX::Vector,vY::Vector) = mean(abs.(vX-vY))
rmse(vX::Vector,vY::Array) = sqrt(mean((vX-vY).^2))

function get_variable_idx(mLoss_s::Array,bRev::Bool)
    mLoss = zeros(Int64, size(mLoss_s))
    for i = 1:size(mLoss_s,1)
        mLoss[i,:] = sortperm(mLoss_s[i,:],rev = bRev)
    end
    return mLoss
end

function get_best(mX::Array,mY::Array,iBest::Int64)
    vBest = sort(unique([unique(mX); unique(mY)]))
end

function par_get_best_comb(mX::Array,vY::Vector,H::Vector{Int64},U::UnivariateSelection,iStart::Int64)
    T,K = size(mX)::Tuple{Int64,Int64}
    P = size(H,1)
    N = size(U.vVar,1)
    counter = 0
    for comb = 1:size(U.vVar,1) 
        println("Starting models with $comb variables")
        comb_index = get_comb(U.vVar,comb)
        n = size(comb_index,1)::Int64
        mMae,mRmse = SharedArray{Float64}(P,n),SharedArray{Float64}(P,n)
        mMae_pc,mRmse_pc = SharedArray{Float64}(P,n),SharedArray{Float64}(P,n)
        @time @sync @parallel for j = 1:n
            j%10 == 0 && println("#Combination: $j")
            mFore,mFore_pc = zeros(Float64,T,P),zeros(Float64,T,P)
            vMae,vRmse     = zeros(Float64,P),zeros(Float64,P)
            vMae_pc,vRmse_pc  = zeros(Float64,P),zeros(Float64,P)
            count = 0
            for h = H
                count += 1
                mFore[:,count]    = out_of_sample_forecast(mX[:,comb_index[j,:]],vY,iStart,h)
                vMae[count]       = mae(mFore[iStart+h:end,count],vY[iStart+h:end])
                vRmse[count]      = rmse(mFore[iStart+h:end,count],vY[iStart+h:end])
                mFore_pc[:,count] = out_of_sample_forecast_pca(mX,vY,iStart,h,comb_index[j,:],1:K)
                vMae_pc[count]    = mae(mFore_pc[iStart+h:end,count],vY[iStart+h:end])
                vRmse_pc[count]   = rmse(mFore_pc[iStart+h:end,count],vY[iStart+h:end])
            end
            mMae[:,j],mRmse[:,j] = vMae,vRmse
            mMae_pc[:,j],mRmse_pc[:,j]  = vMae_pc,vRmse_pc
        end
        dfMAE   = get_df_score([mMae mMae_pc], n)
        dfRMSE  = get_df_score([mRmse mRmse_pc],n)
        filename1 = joinpath(Pkg.dir("NFP"),"test","vMae_best_$(size(U.vVar,1))_comb_$comb.csv")
        filename2 = joinpath(Pkg.dir("NFP"),"test","vRmse_best_$(size(U.vVar,1))_comb_$comb.csv")
        writetable(filename1,dfMAE , separator = ',', header = true)
        writetable(filename2,dfRMSE, separator = ',', header = true)
    end
end

function get_comb(vVar::Vector{Int64},comb::Int64)
    N = size(vVar,1)
    aComb_idx = collect(combinations(1:N, comb))
    vComb_idx = hcat(aComb_idx...)'::Array{Int64}
    return vVar[vComb_idx]::Array{Int64}
end

function out_of_sample_forecast_pca(mX::Array{Float64,2},vY::Vector,iStart::Int64,h::Int64,vVar_idx::Vector{Int64},rPC_idx::UnitRange{Int64})
    T,k = size(mX)::Tuple{Int64,Int64}
    K = size(vVar_idx,1)
    vFore = zeros(T)
    for j = iStart:T-h
        mX_n = view(mX,1:j,1:k)
        vY_n = view(vY,(1:j))
        mPC = pca(mX_n[:,rPC_idx])
        t = size(vY_n,1)
        vYY_n = vY_n[h+1:t]
        mXX_n = hcat(ones(t-h,1), mX_n[1:t-h,vVar_idx], mPC[1:t-h,1:1])
        md = GLM.lm(mXX_n,vYY_n)
        iXX_n = hcat(ones(1,1), view(mX_n,t:t,1:K), view(mPC,t:t,1:1))
        vFore[j+h] = first(GLM.predict(md,iXX_n))
    end
    return vFore
end

function pca(data::Array{Float64}; n::Int64 = 1)
    # Need MultivariateStats
    mX = data.-mean(data,1)
    M = fit(PCA, mX, method = :cov)
    return projection(M)[:,1:n]
end

function get_df_score(m::DenseArray,n::Int64)
    mm = convert(Array,m)::Array
    dfM =  DataFrame(mm)
    names!(dfM,repmat(map(Symbol,1:n),2),allow_duplicates = true)
    return dfM
end

function get_independent(dfData::DataFrame, vVar::Vector{Symbol})
    mX = @>> dfData[.~[(x in vVar) for x in names(dfData)]] convert(Array)
    vNames = @> dfData[.~[(x in vVar) for x in names(dfData)]] names()
    return mX, vNames
end

function get_dependent(vY::Vector,fThreshold::Float64)
    T = size(vY,1)
    vY_prob = zeros(Int64,T)
    for i = 1:T
        vY < fThreshold ? (vY_prob[i] = 1) : (vY_prob[i] = 0)
    end
    return vY_prob
end

function get_dependent(vY::Vector,vThreshold::Vector)
    T = size(vY,1)
    vY_prob = zeros(Int64,T)
    for i = 1:T
        vY[i] < vThreshold[i] ? (vY_prob[i] = 1) : (vY_prob[i] = 0)
    end
    return vY_prob
end

function load_score(sCrit::String,ncomb_load::Int64,H::Vector,vNames::Vector{String})
    (sCrit != "vMae" && sCrit != "vRmse") && error("sCrit must be vMae or vRmse")
    ncomb = size(vNames,1)
    vCrit = H
    comb_identifier = "Models" # preallocate comb_identifier for concatenation
    for comb = 1:ncomb_load
        comb_index = collect(combinations(1:size(vNames,1), comb))
        comb_index = transpose(hcat(comb_index...))
        dfCrit = readtable(joinpath(Pkg.dir("NFP"),"test","$(sCrit)_best_$(ncomb)_comb_$comb.csv"), header =  true)
        comb_identifier = @>> map(string,names(dfCrit)).*"_comb$comb" vcat(comb_identifier)
        vCrit = @>> convert(Array,dfCrit) hcat(vCrit)
        println(size(vCrit))
    end
    best = [[findmin(vCrit[j,2:end])[2]+1 findmin(vCrit[j,2:end])[1]] for j in 1:size(vCrit,1)] # +1 exclude first column and reset indexes
    best = vcat(best...)
    best_comb = @>> comb_identifier[convert(Array{Int64},best[:,1])] map(x -> [Symbol(split(x,"_c")[1]) split(x,"_comb")[2]])
    best_comb = vcat(best_comb...)
    factor_in = [sum(collect(string(best_comb[i,1])) .== '_') for i in 1:size(best_comb,1)]
    return best_comb, factor_in
end

function variable_selection(dfData::DataFrame,vSymbol::Array{Symbol,1},iSymbol::Symbol,H::Vector,iStart::Int64,iBest::Int64)
    mX,vNames = get_independent(dfData,vSymbol)
    vY = convert(Array,dfData[iSymbol])
    U = UnivariateSelection(mX,vY,vNames,H,iStart,iBest)
    const ncomb = size(U.vVar,1)
    # sPath folder is the folder where the results are saved and has to be created
    par_get_best_comb(mX,vY,H,U,iStart)
    return mX,vY,U,vNames
end

function sforecast(dfData::DataFrame,vSymbol::Array{Symbol,1},iSymbol::Symbol,H::Vector,iStart::Int64,iBest::Int64,ncomb_load::Int64)
    l_plot = plot(layout = grid(length(H),1))
    rm_var = size(vSymbol,1)
    
    mX,vY,U,vNames = variable_selection(dfData,vSymbol,iSymbol,H,iStart,iBest)
    best_comb_vMae,factor_in_vMae = load_score("vMae",ncomb_load,H,U.vNames)
    best_comb_vRmse,factor_in_vRmse = load_score("vRmse",ncomb_load,H,U.vNames)
    
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
            comb_index = collect(combinations(1:size(U.vNames,1), parse(Int64,best_comb[i,2])))
            comb_index = transpose(hcat(comb_index...))
            if @> sum(collect(string(best_comb[i,1])) .== '_') == 0
                model_index[i] = comb_index[parse(Int64,split(string(best_comb[i,1]),"x")[2]),:]
                variables[i] = U.vNames[comb_index[parse(Int64,split(string(best_comb[i,1]),"x")[2]),:]]
            else
                model_index[i] = comb_index[parse(Int64,split(string(best_comb[i,1]),r"x|_")[2]),:]
                variables[i] = U.vNames[comb_index[parse(Int64,split(string(best_comb[i,1]),r"x|_")[2]),:]]
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
            best_comb == best_comb_vMae && plot!(l_plot,dfData[:Date][iStart+h:end],vY[iStart+h:end], color = "black", 
            subplot = count)
            plot!(l_plot,dfData[:Date][iStart+h:end],mFore[iStart+h:end,count], line = (line_st[best_comb]), 
            title = "Horizon $h", subplot = count, legend = false, ylabel = string(iSymbol))
        end
    end
    return U,best_comb_vMae,factor_in_vMae,best_comb_vRmse,factor_in_vRmse,l_plot    
end

function fforecast(dfData::DataFrame,vSymbol::Array{Symbol,1},iSymbol::Symbol,H::Vector,iStart::Int64,iBest::Int64,ncomb_load::Int64,iProcs::Int64)
    l_plot = plot(layout = grid(length(H),1))
    rm_var = size(vSymbol,1)
    
    mX,vNames = get_independent(dfData,vSymbol)
    vY = convert(Array,dfData[iSymbol])
    U = UnivariateSelection(mX,vY,vNames,H,iStart,iBest)
    
    best_comb_vMae,factor_in_vMae = load_score("vMae",ncomb_load,H,U.vNames)
    best_comb_vRmse,factor_in_vRmse = load_score("vRmse",ncomb_load,H,U.vNames)
    
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
            comb_index = collect(combinations(1:size(U.vNames,1), parse(Int64,best_comb[i,2])))
            comb_index = transpose(hcat(comb_index...))
            if @> sum(collect(string(best_comb[i,1])) .== '_') == 0
                model_index[i] = comb_index[parse(Int64,split(string(best_comb[i,1]),"x")[2]),:]
                variables[i] = U.vNames[comb_index[parse(Int64,split(string(best_comb[i,1]),"x")[2]),:]]
            else
                model_index[i] = comb_index[parse(Int64,split(string(best_comb[i,1]),r"x|_")[2]),:]
                variables[i] = U.vNames[comb_index[parse(Int64,split(string(best_comb[i,1]),r"x|_")[2]),:]]
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
            best_comb == best_comb_vMae && plot!(l_plot,dfData[:Date][iStart+h:end],vY[iStart+h:end], color = "black", 
            subplot = count)
            plot!(l_plot,dfData[:Date][iStart+h:end],mFore[iStart+h:end,count], line = (line_st[best_comb]), 
            title = "Horizon $h", subplot = count, legend = false, ylabel = string(iSymbol))
        end
    end
    return U,best_comb_vMae,factor_in_vMae,best_comb_vRmse,factor_in_vRmse,l_plot
end

export sforecast, fforecast

end # module ends 