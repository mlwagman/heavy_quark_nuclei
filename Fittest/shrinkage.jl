# compute optimal shrinkage parameter
function optimalShrinkage(fitData)
  # normalize data and find normalized correlation matrix
  fitDataSize = size(fitData)
  NMeas = fitDataSize[1]
  Lt = fitDataSize[2]
  fitDataNorm = Array{Float64}(undef, NMeas, Lt)
  yData = Array{Float64}(undef, Lt)
  covData = Array{Float64}(undef, Lt, Lt)
  covNorm = Array{Float64}(undef, Lt, Lt)
  for t in 1:Lt
   yData[t] = 0
   for n in 1:NMeas
     yData[t] += fitData[n,t]
   end
   yData[t] /= NMeas
  end
  for t in 1:Lt, tp in 1:Lt
   covData[t,tp] = 0
   for n in 1:NMeas
     covData[t,tp] += fitData[n,t] * fitData[n,tp]
   end
   covData[t,tp] /= NMeas
   covData[t,tp] -= yData[t]*yData[tp]
   covData[t,tp] *= NMeas/(NMeas-1)
  end
  for t in 1:Lt
   fitDataNorm[:,t] .= (fitData[:,t] .- yData[t])./sqrt(covData[t,t])
  end
  for t in 1:Lt, tp in 1:Lt
   covNorm[t,tp] = covData[t,tp]/sqrt(covData[t,t]*covData[tp,tp])
  end
  # compute optimal shrinkage parameter
  d2 = 0.0
  for t in 1:Lt, tp in 1:Lt
   if t == tp
     d2 += (covNorm[t,tp] - 1)^2
   else
     d2 += covNorm[t,tp]^2
   end
  end
  d2 /= Lt
  b2 = 0.0
  for t in 1:Lt, tp in 1:Lt, n in 1:NMeas
   b2 += (fitDataNorm[n,t] * fitDataNorm[n,tp] - covNorm[t,tp])^2
  end
  b2 /= (NMeas^2 * Lt)
  lambda = b2 / d2
  if lambda > 1
   return 1
  elseif lambda < 0
   return 0
  else
   return lambda
  end
end
