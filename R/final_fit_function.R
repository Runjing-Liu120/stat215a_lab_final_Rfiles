get_elastic_net_escv_fit <- function(feat_train, resp_train, feat_test, alpha = 0.5){
  # this function fits the elastic net to feat_train and resp_train, 
  # with penalty selection done with ESCV
  # returns the fit for all 20 voxels and 
  # the predictions for feat_train
  
  # we store the predictions for all voxels in a n X 20 matrix
  pred_all <- matrix(0, dim(feat_test)[1], 20)
  
  # save the fits to each voxel in a list
  final_fits <- list() 
  
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
    
    fit <- glmnet(feat_train, resp_train[, voxel], lambda = lambda_escv, alpha = alpha)
    pred_all[, voxel] <- predict(fit, newx = feat_test)
    final_fits[[voxel]] <- fit
  }
  return(list(predictions = pred_all, fits = final_fits))
}