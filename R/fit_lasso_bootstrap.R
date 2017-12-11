# this script draws one bootstrap sample and saves a fit
# we fit lasso or elastic net, with penalty parameter selected by ESCV

# we wil call this script from the terminal, and run it in parallel to save time. 

# load libraries
library(tidyverse)
library(glmnet)
library(caret)
source('../../model_selection_utils.R')

data_path <- '../../../data/'
source('../../load_split_data.R')

args <- commandArgs(TRUE)

# elastic net parameter
# alpha = 1 gives lasso
alpha <- as.double(args[2]) 
print(alpha)

# set the seed
bootstrap_no <- as.integer(args[1])
set.seed(425254245 + bootstrap_no)

print(425254245 + bootstrap_no)

# bootstrap sample
n_obs <- dim(feat_train)[1]
sample_indx <- sample(1:n_obs, n_obs, replace = TRUE)

feat_sample <- feat_train[sample_indx, ]
resp_sample <- resp_train[sample_indx, ]
# print(sample_indx)
# we will look at voxel 9
voxel <- 9

# run CV
print('running cv ...')
cvfit <- cv.glmnet(feat_sample, resp_sample[, voxel], nfolds = 10, type.measure = 'mse', 
                   alpha = alpha)
lambda_cv <- cvfit$lambda.min
lambdas <- cvfit$lambda

# run ES
print('running escv ...')
es <- select_lambda_EC(feat_sample, resp_sample, lambda = lambdas, folds = 10, voxel = voxel, 
                       alpha = alpha)

es_constrained <- es[lambdas >= lambda_cv] # we only consider more regularized models
lambda_escv <- lambdas[lambdas >= lambda_cv][which.min(es_constrained)]

# get the fit with the selected lambda
fit <- glmnet(feat_sample, resp_sample[, voxel], lambda = lambda_escv, alpha = alpha)

# save 
filename <- paste('bootstrap_sample_', bootstrap_no, '.RData', sep = '')
save(fit, file = paste('./', filename, sep = ''))

print('done. ')