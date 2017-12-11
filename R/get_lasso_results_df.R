# script gets the results for the lasso or the elastic net
# for all 20 voxels

# returns two 20 X 5 dataframes, one storing selected lambdas for all 20 voxels using 
# the five model selection techniques; the other gives the correlation scores on the 
# validation set

args <- commandArgs(TRUE)
alpha <- as.double(args[1]) # elastic net paramter; 1 gives LASSO

library(tidyverse)
library(glmnet)
library(caret)
source('./model_selection_utils.R')

data_path <- '../data/'
source('./load_split_data.R')

lasso_penalties_all <- matrix(0, 20, 5)
lasso_correlations_all <- matrix(0, 20, 5)
lasso_df_all <- matrix(0, 20, 5)

# should I parallelize this?
# I'll just go get lunch ...
for(voxel in 1:20){
  print(paste('working on voxel', voxel))
  # get the lasso fit
  fit <- glmnet(feat_train, resp_train[, voxel], alpha = alpha)
  lambdas <- fit$lambda # these are the penalties we will consider
  
  # get penalties for aic, aicc, and bic
  lasso_IC_results <- select_lambda_IC(fit, feat_train, resp_train, voxel = voxel)
  lasso_IC_results <- as.data.frame(lasso_IC_results)
  
  # get optimal lambdas
  lasso_IC_results_sparse <- filter(lasso_IC_results, df < 150)
  lambda_aic <- lasso_IC_results_sparse[which.min(lasso_IC_results_sparse$aic), 
                                        'lambda']
  lambda_bic <- lasso_IC_results_sparse[which.min(lasso_IC_results_sparse$bic), 
                                        'lambda']
  lambda_aicc <- lasso_IC_results_sparse[which.min(lasso_IC_results_sparse$aicc), 
                                         'lambda']
  
  # run CV
  cvfit <- cv.glmnet(feat_train, resp_train[, voxel], lambda = lambdas, 
                     nfolds = 10, type.measure = 'mse', alpha = alpha)
  lambda_cv <- cvfit$lambda.min
  
  # run ES
  es <- select_lambda_EC(feat_train, resp_train, lambda = lambdas, folds = 10, 
                         voxel = voxel, 
                         alpha = alpha)
  
  es_constrained <- es[lambdas >= lambda_cv] # we only consider more regularized models
  lambda_escv <- lambdas[lambdas >= lambda_cv][which.min(es_constrained)]
  
  penalties <- c(lambda_aic, lambda_bic, lambda_aicc, lambda_cv, lambda_escv)
  pred_val <- predict(fit, newx = feat_val, 
                      s = penalties, alpha = alpha)
  
  correlations <- apply(pred_val, 2, cor, y = resp_val[, voxel])
  
  lasso_penalties_all[voxel, ] <- penalties
  lasso_correlations_all[voxel, ] <- correlations
  
}

rownames(lasso_penalties_all) <- paste('voxel', 1:20)
rownames(lasso_correlations_all) <- paste('voxel', 1:20)

colnames(lasso_penalties_all) <- c('AIC', 'BIC', 'AICc', 'CV', 'ESCV')
colnames(lasso_correlations_all) <- c('AIC', 'BIC', 'AICc', 'CV', 'ESCV')

lasso_penalties_all <- as.data.frame(lasso_penalties_all)
lasso_correlations_all <- as.data.frame(lasso_correlations_all)

if(alpha == 1){
  method_name <- 'lasso'
}else{ 
  method_name <- paste('elastic_net_alpha', as.character(alpha*10), sep = '')
}

save(lasso_penalties_all, 
     file = paste('./results/penalties_', method_name, '.RData', sep = ''))
save(lasso_correlations_all, 
     file = paste('./results/correlations_', method_name, '.RData', sep = ''))
                                       

