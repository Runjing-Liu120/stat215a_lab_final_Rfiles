# script gets the results for the elastic net
# for all 20 voxels

# elastic net parameter
alpha <- 0.5

library(tidyverse)
library(glmnet)
library(caret)
source('./model_selection_utils.R')

data_path <- '../data/'
source('./load_split_data.R')

elastic_net_penalties_all <- matrix(0, 20, 2)
elastic_net_correlations_all <- matrix(0, 20, 2)

for(voxel in 1:20){
  print(paste('working on voxel', voxel))

  cvfit <- cv.glmnet(feat_train, resp_train[, voxel], 
                     nfolds = 10, type.measure = 'mse', alpha = alpha)
  lambda_cv <- cvfit$lambda.min
  lambdas <- cvfit$lambda
  
  # run ES
  es <- select_lambda_EC(feat_train, resp_train, alpha = alpha, 
                         lambda = lambdas, folds = 10, voxel = voxel)
  
  es_constrained <- es[lambdas >= lambda_cv] # we only consider more regularized models
  lambda_escv <- lambdas[lambdas >= lambda_cv][which.min(es_constrained)]
  
  penalties <- c(lambda_cv, lambda_escv)
  fit <- glmnet(feat_train, resp_train[, voxel], lambda = penalties, alpha = alpha)
  pred_val <- predict(fit, newx = feat_val)
  
  correlations <- apply(pred_val, 2, cor, y = resp_val[, voxel])
  
  elastic_net_penalties_all[voxel, ] <- penalties
  elastic_net_correlations_all[voxel, ] <- correlations
}

rownames(elastic_net_penalties_all) <- paste('voxel', 1:20)
rownames(elastic_net_correlations_all) <- paste('voxel', 1:20)

colnames(elastic_net_penalties_all) <- c('CV', 'ESCV')
colnames(elastic_net_correlations_all) <- c('CV', 'ESCV')

penalties_all <- as.data.frame(elastic_net_penalties_all)
correlations_all <- as.data.frame(elastic_net_correlations_all)

save(penalties_all, file = './results/penalties_elastic_net.RData')
save(correlations_all, file = './results/correlations_elastic_net.RData')
                                       

