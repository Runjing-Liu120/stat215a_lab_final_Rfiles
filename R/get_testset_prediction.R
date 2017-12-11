# this script gets the predictions on the held out test set
# it saves the results in the results folder, in a file called
# test_set_results.RData

library(tidyverse)
library(glmnet)
library(caret)
source('./model_selection_utils.R')



print('loading data')
data_path <- '../data/'
source('./load_split_data.R')

source('./final_fit_function.R')

final_results <- get_elastic_net_escv_fit(feat_train, resp_train, feat_test)

final_fits <- final_results$fits
final_predictions <- final_results$predictions

filename <- 'test_set_results'
save(final_predictions, final_fits, file = paste('./results/', filename, '.RData', sep = ''))
print('done')
