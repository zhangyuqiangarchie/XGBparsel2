# XGBparsel2
## An automatic parameter tuning method for xgboost

About the **xgb.auc** function in **xgbparsel** package: 

>The cross validation function **xgb.cv** in xgboost package only performs cross validation and does not select relatively excellent parameters. In order to overcome this problem, **xgb.auc** is an automatic parameter adjustment function suitable for binary classification. The function creates a cycle with the number of N and the parameter **max_ Depth** is randomly selected from *6-10*. Parameter **ETA** follows a uniform distribution of *0.01-0.3*, parameter gamma follows a uniform distribution of *0-0.2*, parameter **subsample** follows a uniform distribution of *0.6-0.9*, and parameter colsample follows a uniform distribution_ Bytree obeys the uniform distribution of *0.5-0.8*, parameter **min_ child_ Weight** randomly selected from 1-40, parameter **Max_ delta_ Step** is randomly selected from *1-10*. The value range of all parameters is the typical parameter value in xgboost algorithm. Through cross validation, the parameters corresponding to the maximum **AUC** on the test set are selected, and the optimal parameters are obtained through continuous iteration.

The xgbparsel package also comes with the function **xgb.rmse**. The principle of this function is similar to **xgb.auc**. But the difference is that **xgb.rmse** is used for regression rather than classification.
