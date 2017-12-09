select_lambda_IC <- function(fit, feat_train, resp_train, voxel = 1){
  # fits l1 penalized least squares for a range of lambda
  # returns the aic and bic
  
  # get MSE
  resp_pred <- predict(fit, newx = feat_train, s = fit$lambda)
  mse <- apply((resp_pred - resp_train[, voxel])^2, 2, sum)
  
  n_obs <- dim(feat_train)[1]
  # compute AIC
  aic <- n_obs * log(mse / n_obs) + 2 * fit$df
  
  # compute BIC
  bic <- n_obs * log(mse / n_obs) + log(n_obs) * fit$df
  
  # compute AICc
  aicc <- aic + 2 * fit$df * (fit$df + 1) / (n_obs - fit$df - 1)
  
  lasso_IC_results <- list(lambda = fit$lambda, 
                           df = fit$df, 
                           mse = mse, 
                           aic = aic, 
                           bic = bic, 
                           aicc = aicc)
  return(lasso_IC_results)
}


select_lambda_EC <- function(feat_train, resp_train, lambdas, folds = 10, voxel = 1, 
                             alpha = 1.0){
  # get estimation stability (ES) measure for specified lambdas
  
  # create folds for CV
  n_obs <- dim(feat_train)[1]
  fold_indx <- createFolds(1:n_obs, k = folds)
  
  # create array where we will store the predicted y for all lambda and all folds
  y_es <- array(0, c(n_obs, length(lambdas), folds))
  for(i in 1:folds){ # loop through folds
    # train set for ith fold
    X_cv_train <- feat_train[-fold_indx[[i]], ]
    y_cv_train <- resp_train[-fold_indx[[i]], voxel]
    
    # test set for ith fold
    X_cv_test <- feat_train[fold_indx[[i]], ]
    y_cv_test <- resp_train[fold_indx[[i]], voxel]
    
    fit_cv <- glmnet(X_cv_train, y_cv_train, alpha = alpha) # train
    y_cv_pred <- predict(fit_cv, newx = feat_train, s = lambdas) # predict on whole dataset
    
    y_es[, , i] <- y_cv_pred
    print(paste('fold =', i))
  }
  
  es <- rep(0, length(lambdas))
  # TODO: remove for loop, use vector operations ...
  for(i in 1:length(lambdas)){
    y_lambda <- y_es[, i, ] # prections at lambda[i]
    y_mean <- apply(y_lambda, 1,mean)
    
    es[i] <- mean(apply((y_lambda - y_mean)^2, 2, sum)) / sum(y_mean^2)
  }
  
  return(es)
}

