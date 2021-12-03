#' @title conf.matrix
#' @description conf.matrix is a function which is used to generate a confusion matrix.
#' @param p the probability which is the threshold of the pred.If pred exceeds,it will be assigned to 1.otherwise,it will be assigned to 0.
#' @param x the vector of the prediction of the xgboost model.
#' @param tarvar the target variable which is compared to the prediction.
#'
#' @return a confusion matrix
#' @export
#'
#' @examples conf.matrix(0.5,x=c(0:1),tarvar=c(0:1))
  conf.matrix <- function(p,x,tarvar = tarvar){
  prediction <- as.numeric(x>p)
  return(table(prediction,tarvar))
}
