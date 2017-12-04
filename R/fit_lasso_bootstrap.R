# this script draws one bootstrap sample and saves a fit
# we fit lasso with penalty parameter selected by ESCV

# we call this script from the terminal, and run it in parallel to save time. 

# set the seed
args <- commandArgs(TRUE)
bootstrap_no <- as.integer(args[1])
set.seed(425254245 + bootstrap_no)

# load libraries
library(tidyverse)
library(glmnet)
library(caret)
source('../../model_selection_utils.R')

data_path <- '../../../data/'
source('../../load_split_data.R')

# bootstrap sample
n_obs <- dim(feat_train)[1]
sample_indx <- sample(1:n_obs, n_obs, replace = TRUE)

feat_sample <- feat_train[sample_indx, ]
resp_sample <- resp_train[sample_indx, ]

# we will look at voxel 9
voxel <- 9

# run CV
print('running cv ...')
cvfit <- cv.glmnet(feat_sample, resp_sample[, voxel], nfolds = 10, type.measure = 'mse')
lambda_cv <- cvfit$lambda.min
lambdas <- cvfit$lambda

# run ES
print('running escv ...')
es <- select_lambda_EC(feat_sample, resp_sample, lambda = lambdas, folds = 10, voxel = voxel)

es_constrained <- es[lambdas >= lambda_cv] # we only consider more regularized models
lambda_escv <- lambdas[lambdas >= lambda_cv][which.min(es_constrained)]

# get the fit with the selected lambda
fit <- glmnet(feat_sample, resp_sample[, voxel], lambda = lambda_escv)

# save 
filename <- paste('bootstrap_sample_', bootstrap_no, '.RData', sep = '')
save(fit, file = paste('./', filename, sep = ''))
print('done. ')
