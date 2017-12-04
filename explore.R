# exploration script

library(tidyverse)
library(glmnet)

data_path <- './data/'
source('./R/load_split_data.R')
n_obs <- dim(fit_feat)[1]

voxel <- 2
# fit lasso
fit <- glmnet(feat_train, resp_train[, voxel])

# exploring information criterion
source('./R/model_selection_utils.R')
lasso_IC_results <- select_lambda_IC(fit, feat_train, resp_train, voxel = voxel)
lasso_IC_results <- as.data.frame(lasso_IC_results)

lasso_IC_results %>% ggplot() + geom_point(aes(x = lambda, y = log(mse)))
lasso_IC_results %>% ggplot() + geom_point(aes(x = lambda, y = df))
lasso_IC_results %>% ggplot() + geom_point(aes(x = lambda, y = aic))
lasso_IC_results %>% ggplot() + geom_point(aes(x = lambda, y = bic))

# lasso_IC_results %>% ggplot() + 
#   geom_point(aes(x = lambda, 
#                  y =  2 * df * (df + 1) / (n_obs - df - 1)))

lasso_IC_results %>% filter(df < 150) %>%
  ggplot() + geom_point(aes(x = lambda, y = aicc))

# get optimal lambdas
lasso_IC_results_sparse <- filter(lasso_IC_results, df < 150)
lambda_aic <- lasso_IC_results_sparse[which.min(lasso_IC_results_sparse$aic), 
                                      'lambda']
lambda_bic <- lasso_IC_results_sparse[which.min(lasso_IC_results_sparse$bic), 
                                      'lambda']
lambda_aicc <- lasso_IC_results_sparse[which.min(lasso_IC_results_sparse$aicc), 
                                      'lambda']

lambdas <- lasso_IC_results$lambda # we'll use this sequence of lambdas in the future too

# cross validation
cvfit <- cv.glmnet(feat_train, resp_train[, voxel], lambda = lambdas, 
                   nfolds = 10, type.measure = 'mse')
lambda_cv <- cvfit$lambda.min

# ES
es <- select_lambda_EC(feat_train, resp_train, lambda = lambdas, folds = 10, voxel = voxel)
ggplot() + geom_point(aes(x = lambdas, y = es)) + 
  geom_vline(xintercept = lambda_cv)

es_constrained <- es[lambdas > lambda_cv] # we only consider more regularized models
lambda_escv <- lambdas[lambdas > lambda_cv][which.min(es_constrained)]

fit <- glmnet(feat_train, resp_train[, voxel])
pred_val <- predict(fit, newx = feat_val, 
                   s = c(lambda_aic, lambda_bic, lambda_aicc, lambda_cv, lambda_escv))

ggplot() + geom_point(aes(x = pred_val[, 4], y = resp_val[, voxel]))

correlations <- apply(pred_val, 2, cor, y = resp_val[, voxel])

# pred_val <- predict(fit, newx = feat_train, 
                    s = c(lambda_aic, lambda_bic, lambda_aicc, lambda_cv, lambda_escv))
# ggplot() + geom_point(aes(x = pred_val[, 4], y = resp_train[, voxel]))

###########
# lets try elastic net
fit_elastnet <- cv.glmnet(feat_train, resp_train[, voxel], alpha = 0.5)
elastnet_pred <- predict(fit_elastnet, newx = feat_val, s = fit_elastnet$lambda.min)
cor(elastnet_pred, pred_val[, voxel])
ggplot() + geom_point(aes(x = pred_val[, 4], y = elastnet_pred))
