#' @title xgb.auc
#' @description xgb.auc is an automatic parameter adjustment function which is applied to the Classification task by xgboost.
#' @param data training dataset. xgb.auc accepts only an \code{xgb.DMatrix} as the input.
#' @param n the number of cycles.
#' @param cvround max number of boosting iterations.
#' @param cvfold the original dataset is randomly partitioned into \code{nfold} equal size subsamples.
#' @param early_stopping_rounds If NULL, the early stopping function is not triggered. If set to an integer k,
#' training with a validation set will stop if the performance doesn't improve for k rounds.
#' @param seed.number random number seed.
#' @param nthread  number of thread used in training.
#'
#' @return a \code{list} contains the best_param,the best_auc and best_auc_index.
#' @export
#'
#' @importFrom stats runif
#' @importFrom xgboost xgb.cv
#' @importFrom xgboost xgb.DMatrix
#' @importFrom Matrix sparse.model.matrix
#' @import xgboost
#' @import Matrix
#'
#' @examples
#'
#'library(xgboost)
#'data(redwine,package="XGBparsel")
#'redwine$grade[redwine$quality <= 5] <- "0"
#'redwine$grade[redwine$quality >= 6] <- "1"
#'myvars <- names(redwine) %in% "quality"
#'redwine <- redwine[!myvars]
#'redwine_s <- Matrix::sparse.model.matrix( grade~.-1 ,data = redwine)
#'train <- xgb.DMatrix( data = redwine_s , label = redwine$grade )
#'fit <-xgb.auc(train,10,cvround=100,cvfold=5,
#'                   early_stopping_rounds =10,
#'                   seed.number = 12345,nthread = 8)
 xgb.auc <- function(data,n,cvround = cvround,cvfold = cvfold,
                    early_stopping_rounds = early_stopping_rounds,
                    seed.number = seed.number,nthread = nthread){

  test_auc_mean = NULL
  best_param = list()
  best_auc = -Inf
  best_auc_index = 0

  for (iter in 1:n) {
    param <- list(objective = "binary:logistic",
                  eval_metric = "auc",
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

    mdcv <- xgb.cv(data = data,params = param,nthread = nthread,
                   nfold = cv.nfold, nrounds = cv.nround,
                   verbose = T, early_stopping_rounds = early_stopping_rounds,
                   maximize = TRUE)
    as.matrix(mdcv$evaluation_log)
    max_auc = max(mdcv$evaluation_log[, 4])
    max_auc_index = which.max(mdcv$evaluation_log[, 4])

    if (max_auc > best_auc) {
      best_auc = max_auc
      best_auc_index = max_auc_index
      best_param = param
    }
  }
  result <-list(best_param=best_param,
                best_auc=best_auc,
                best_auc_index=best_auc_index)
  return(result)
}

