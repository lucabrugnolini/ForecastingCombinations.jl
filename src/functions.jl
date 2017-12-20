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

@everywhere function out_of_sample_forecast(mX::Array{Float64,2},vY::Vector,iStart::Int64,h::Int64)
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

@everywhere function out_of_sample_forecast_prob(mX::Array{Float64,2},vY::Vector,iStart::Int64,h::Int64)
  T,K = size(mX)::Tuple{Int64,Int64}
  vFore = zeros(T)
  for j = iStart:T-h
    mX_n = view(mX,1:j,1:K)
    vY_n = view(vY,(1:j))
    t = size(vY_n,1)
    mXX_n = hcat(ones(t-h,1), mX_n[1:t-h,1:K])
    vYY_n = vY_n[h+1:t]
    md = GLM.fit(GeneralizedLinearModel,mXX_n,vYY_n,Binomial(),ProbitLink())
    iXX_n = hcat(ones(1,1), view(mX_n,t:t,1:K))
    vFore[j+h] = first(GLM.predict(md,iXX_n))
  end
  return vFore
end

@everywhere function out_of_sample_forecast_prob_pca(mX::Array{Float64,2},vY::Vector,iStart::Int64,h::Int64,vVar_idx::Vector{Int64},rPC_idx::UnitRange{Int64})
  T,k = size(mX)::Tuple{Int64,Int64}
  K = size(vVar_idx,1)
  vFore = zeros(T)
  for j = iStart:T-h
    mX_n = view(mX,1:j,1:k)
    vY_n = view(vY,(1:j))
    mPC = pca(mX_n[:,rPC_idx])
    t = size(vY_n,1)
    vYY_n = view(vY_n,h+1:t)
    mXX_n = hcat(ones(t-h,1), view(mX_n,1:t-h,vVar_idx), view(mPC,1:t-h,1:1))
    md = GLM.fit(GLM.GeneralizedLinearModel,mXX_n,vYY_n,Binomial(),ProbitLink())
    iXX_n = hcat(ones(1,1), view(mX_n,t:t,1:K), view(mPC,t:t,1:1))
    vFore[j+h] = first(GLM.predict(md,iXX_n))
  end
  return vFore
end

@everywhere mae(vX::Vector,vY::Vector) = mean(abs.(vX-vY))
@everywhere rmse(vX::Vector,vY::Array) = sqrt(mean((vX-vY).^2))

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

function par_get_best_comb(mX::Array,vY::Vector,H::Vector{Int64},U::UnivariateSelection,sPath::String,vThreshold::Vector)
  T,K = size(mX)::Tuple{Int64,Int64}
  P = size(H,1)
  N = size(U.vVar,1)
  counter = 0
  vY_prob = get_dependent(vY,vThreshold)
  for comb = 1:size(U.vVar,1) 
    println("Starting models with $comb variables")
    comb_index = get_comb(U.vVar,comb)
    n = size(comb_index,1)::Int64
    mMae,mRmse = SharedArray{Float64}(P,n),SharedArray{Float64}(P,n)
    mMae_pc,mRmse_pc = SharedArray{Float64}(P,n),SharedArray{Float64}(P,n)
    mMae_prob,mRmse_prob = SharedArray{Float64}(P,n),SharedArray{Float64}(P,n)
    mMae_prob_pc,mRmse_prob_pc = SharedArray{Float64}(P,n),SharedArray{Float64}(P,n)    
    @time @sync @parallel for j = 1:n
      j%10 == 0 && println("#Combination: $j")
      mFore,mFore_pc = zeros(Float64,T,P),zeros(Float64,T,P)
      mFore_prob,mFore_prob_pc = zeros(Float64,T,P),zeros(Float64,T,P)
      vMae,vRmse     = zeros(Float64,P),zeros(Float64,P)
      vMae_pc,vRmse_pc  = zeros(Float64,P),zeros(Float64,P)
      vMae_prob,vRmse_prob     = zeros(Float64,P),zeros(Float64,P)
      vMae_prob_pc,vRmse_prob_pc  = zeros(Float64,P),zeros(Float64,P)      
      count = 0
      for h = H
        count += 1
        mFore[:,count]    = out_of_sample_forecast(mX[:,comb_index[j,:]],vY,iStart,h)
        vMae[count]       = mae(mFore[iStart+h:end,count],vY[iStart+h:end])
        vRmse[count]      = rmse(mFore[iStart+h:end,count],vY[iStart+h:end])
        mFore_pc[:,count] = out_of_sample_forecast_pca(mX,vY,iStart,h,comb_index[j,:],1:K)
        vMae_pc[count]    = mae(mFore_pc[iStart+h:end,count],vY[iStart+h:end])
        vRmse_pc[count]   = rmse(mFore_pc[iStart+h:end,count],vY[iStart+h:end])

        mFore_prob[:,count]    = out_of_sample_forecast_prob(mX[:,comb_index[j,:]],vY_prob,iStart,h)
        vMae_prob[count]       = mae(mFore_prob[iStart+h:end,count],vY_prob[iStart+h:end])
        vRmse_prob[count]      = rmse(mFore_prob[iStart+h:end,count],vY_prob[iStart+h:end])
        mFore_prob_pc[:,count] = out_of_sample_forecast_prob_pca(mX,vY_prob,iStart,h,comb_index[j,:],1:K)
        vMae_prob_pc[count]    = mae(mFore_prob_pc[iStart+h:end,count],vY_prob[iStart+h:end])
        vRmse_prob_pc[count]   = rmse(mFore_prob_pc[iStart+h:end,count],vY_prob[iStart+h:end])
      end
      mMae[:,j],mRmse[:,j] = vMae,vRmse
      mMae_pc[:,j],mRmse_pc[:,j]  = vMae_pc,vRmse_pc

      mMae_prob[:,j],mRmse_prob[:,j] = vMae_prob,vRmse_prob
      mMae_prob_pc[:,j],mRmse_prob_pc[:,j]  = vMae_prob_pc,vRmse_prob_pc
    end
    dfMAE   = get_df_score([mMae mMae_pc], n)
    dfRMSE  = get_df_score([mRmse mRmse_pc],n)

    dfMAE_prob   = get_df_score([mMae_prob mMae_prob_pc], n)
    dfRMSE_prob  = get_df_score([mRmse_prob mRmse_prob_pc],n)
    filename1 = sPath*"vMae_best_$(size(U.vVar,1))_comb_$comb.csv"
    filename2 = sPath*"vRmse_best_$(size(U.vVar,1))_comb_$comb.csv"
    filename3 = sPath*"vMae_prob_best_$(size(U.vVar,1))_comb_$(comb).csv"
    filename4 = sPath*"vRmse_prob_best_$(size(U.vVar,1))_comb_$(comb).csv"
    println(filename2)
    writetable(filename1,dfMAE , separator = ',', header = true)
    writetable(filename2,dfRMSE, separator = ',', header = true)
    writetable(filename3,dfMAE_prob , separator = ',', header = true)
    writetable(filename4,dfRMSE_prob, separator = ',', header = true)
  end
end

function get_comb(vVar::Vector{Int64},comb::Int64)
  N = size(vVar,1)
  aComb_idx = collect(combinations(1:N, comb))
  vComb_idx = hcat(aComb_idx...)'::Array{Int64}
  return vVar[vComb_idx]::Array{Int64}
end
  
  @everywhere function out_of_sample_forecast_pca(mX::Array{Float64,2},vY::Vector,iStart::Int64,h::Int64,vVar_idx::Vector{Int64},rPC_idx::UnitRange{Int64})
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

  @everywhere function pca(data::Array{Float64}; n::Int64 = 1)
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

  function moving_average(y::Array, p::Int64)
    x = zeros(size(y))
    for i = 1:size(y,1)-p+1
      x[p+i-1,:] = mean(y[i:p+i-1,:],1)
    end
    x[1:p-1,:] = kron(x[p,:]',ones(p-1,1))
    return x
  end
  
