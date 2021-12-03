#' @title xgb.rmse
#' @description xgb.rmse is an automatic parameter adjustment function which is applied to the regression task by xgboost.
#' @param data training dataset. xgb.rmse accepts only an \code{xgb.DMatrix} as the input.
#' @param n the number of cycles.
#' @param cvround max number of boosting iterations.
#' @param cvfold the original dataset is randomly partitioned into \code{nfold} equal size subsamples.
#' @param early_stopping_rounds If NULL, the early stopping function is not triggered. If set to an integer k,
#' training with a validation set will stop if the performance doesn't improve for k rounds.
#' @param seed.number random number seed.
#' @param nthread  number of thread used in training.
#'
#' @return a \code{list} contains the best_param,the best_rmse and best_rmse_index.
#' @export
#'
#' @importFrom stats runif
#' @importFrom xgboost xgb.cv
#' @importFrom xgboost xgb.DMatrix
#' @importFrom Matrix sparse.model.matrix
#' @import xgboost
#' @import Matrix
#' @examples
#'
#'library(xgboost)
#'data(agaricus.train, package='xgboost')
#'data(agaricus.test, package='xgboost')
#'dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
#'dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)
#'fit <-xgb.rmse(dtrain,10,cvround=100,cvfold=5,
#'                   early_stopping_rounds =10,
#'                   seed.number = 12345,nthread = 8)
 xgb.rmse <- function(data,n,cvround = cvround,cvfold = cvfold,
                     early_stopping_rounds = early_stopping_rounds,
                     seed.number = seed.number,nthread = nthread){
  test_rmse_mean = NULL
  best_param = list()
  best_rmse = Inf
  best_rmse_index = 0

  for (iter in 1:n) {
    param <- list(objective = "reg:squarederror",
                  eval_metric = "rmse",
                  max_depth = sample(6:10, 1),
                  eta = runif(1, .01, .3),
                  gamma = runif(1, 0.0, 0.2),
                  subsample = runif(1, .6, .9),
                  colsample_bytree = runif(1, .5, .8),
                  min_child_weight = sample(1:40, 1),
                  max_delta_step = sample(1:10, 1)
    )

    cv.nround = cvround
    cv.nfold = cvfold
    set.seed(seed.number)

    mdcv <- xgb.cv(data = data, params = param, nthread = nthread,
                   nfold = cv.nfold, nrounds = cv.nround,
                   verbose = T, early_stopping_rounds = early_stopping_rounds,
                   maximize = FALSE)
    as.matrix(mdcv$evaluation_log)
    min_rmse = min(mdcv$evaluation_log[,4])
    min_rmse_index = which.min(mdcv$evaluation_log[,4])

    if (min_rmse < best_rmse) {
      best_rmse = min_rmse
      best_rmse_index = min_rmse_index
      best_param = param
    }
  }
  result <-list(best_param = best_param,
                best_rmse = best_rmse,
                best_rmse_index = best_rmse_index)
  return(result)
}




