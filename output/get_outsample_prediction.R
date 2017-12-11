library(glmnet)

# get data
data_path <- '../data/'
source('../R/load_split_data.R')

# get fit from first voxel
load('../R/results/test_set_results.RData')
vox1_final_fit <- final_fits[[1]]

final_prediction <- predict(vox1_final_fit, newx =  val_feat)

write.table(final_prediction, file="predv1_runjing_liu.txt", row.names=FALSE, col.names=FALSE)
