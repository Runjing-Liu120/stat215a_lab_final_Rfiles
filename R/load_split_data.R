# this script loads the data, normalizes the features, and 
# splits the data into training, validation, and testing

set.seed(54362089)

load(paste(data_path, 'fMRIdata.RData', sep = ''))
n_obs <- dim(fit_feat)[1]

# some columns are constant; remove them 
constant_colns <- apply(fit_feat, 2, sd) == 0
fit_feat <- fit_feat[, !constant_colns]
val_feat <- val_feat[, !constant_colns]

# normalize features
scale_features <- TRUE
if(scale_features){
  fit_feat <- scale(fit_feat)
  val_feat <- scale(val_feat)
}

propn_train <- 0.6
propn_val <- 0.2
propn_test <- 0.2

indx <- 1:n_obs

# training indices
train_indx <- sample(indx, round(propn_train * n_obs))
indx <- indx[!(indx %in% train_indx)]

# validation indices
val_indx <- sample(indx, round(propn_val * n_obs))
indx <- indx[!(indx %in% val_indx)]

# testing indices
test_indx <- sample(indx, round(propn_test * n_obs))

# assert statment for my sanity, check we got all indices
stopifnot(length(c(train_indx, val_indx, test_indx) %in% 1:n_obs) == n_obs)
stopifnot(all(c(train_indx, val_indx, test_indx) %in% 1:n_obs) & 
            all(1:n_obs %in% c(train_indx, val_indx, test_indx)))

feat_train <- fit_feat[train_indx,]
feat_val <- fit_feat[val_indx,]
feat_test <- fit_feat[test_indx,]

resp_train <- resp_dat[train_indx,]
resp_val <- resp_dat[val_indx,]
resp_test <- resp_dat[test_indx,]

