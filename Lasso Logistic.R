rm(list = ls())    #delete objects
cat("\014")
library(class)
library(ggplot2)
library(dplyr)
library(glmnet)
#library(alocv)
library(rmutil)
library(tictoc)
library(latex2exp)
#install.packages("naniar")
library(naniar)

# import dataset
library(readxl)
csv_result_5year <- read_excel("Downloads/STA 9891/final project data/csv_result-5year.xlsx")
#View(csv_result_5year)
# checking which columns have null values
allmisscols <- sapply(csv_result_5year, function(x) all(is.na(x) | x == '' ))
# replace missing value with average value
csv_result_5year[is.na(csv_result_5year)] = 0


###### repeat modeling 50 times ######
# create vectors to save AUC for training and testing
auc_train_lasso = c()
auc_test_lasso = c()
period_lasso = c()
p = dim(csv_result_5year)[2] - 1
lambda_min_lasso = c()
num_positive_train_lasso = c()
num_positive_test_lasso = c()
coef_beta0 = c()
coef_beta = c()

for (i in c(1:50)){
  # split dataset into training and testing
  dt = sort(sample(nrow(csv_result_5year),nrow(csv_result_5year)*0.9))
  train = csv_result_5year[dt,]
  test = csv_result_5year[-dt,]
  # get predictors and label
  X.train = subset(train,select = -c(class))
  X.train = data.matrix(X.train)
  y.train = train$class
  num_positive_train_lasso = append(num_positive_train_lasso,sum(y.train==1))
  
  X.test = subset(test,select = -c(class))
  X.test = data.matrix(X.test)
  y.test = test$class
  num_positive_test_lasso = append(num_positive_test_lasso,sum(y.test==1))
  
  n.train             =        length(y.train)
  n.P                 =        sum(y.train)
  n.N                 =        n.train - n.P
  ww                  =        rep(1,n.train)
  ww[y.train==1]      =        n.N/n.P
  ##### Lasso Logistic #######
  
  # train logistic elastic-net model with 10-fold
  start_time = Sys.time()
  cv.lasso    =     cv.glmnet(X.train, y.train, family = "binomial", alpha = 1, nfolds = 10, type.measure="auc",
                              weights = ww,nlambda=100,lambda.min.ratio=1e-3,maxit= 500,thresh=1e-3,parallel = TRUE)
                              # 4
                              #nlambda=50,lambda.min.ratio=1e-1,
                              #parallel = TRUE,maxit=100,
                              #thresh=1e-1
                              # 5
                              # nlambda=100,lambda.min.ratio=1e-3,
                              # parallel = TRUE,maxit=500,
                              # thresh=1e-3
            
  
  end_time = Sys.time()
  t = end_time - start_time
  lambda_min_lasso = append(lambda_min_lasso,cv.lasso$lambda.min)
  # plot first cv curve
  if (i == 1){
    #par("mar")
    #par(mar=c(1,1,1,1))
    plot(cv.lasso)
  }
  # using min lambda to train model
  lasso       =     glmnet(X.train, y.train, lambda = cv.lasso$lambda.min, family = "binomial", alpha = 1,weights = ww,
                           nlambda=100,lambda.min.ratio=1e-3,maxit= 500,thresh=1e-3,parallel = TRUE)
  # get model coefficients
  beta0.hat  = lasso$a0
  coef_beta0 = append(coef_beta0,beta0.hat)
  beta.hat   = as.vector(lasso$beta)
  coef_beta  = append(coef_beta,beta.hat)
  
  prob.train = exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))
  prob.test  = exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
  
  dt                      =        0.01
  thta                    =        1-seq(0,1, by=dt)
  thta.length             =        length(thta)
  FPR.train               =        matrix(0, thta.length)
  TPR.train               =        matrix(0, thta.length)
  FPR.test                =        matrix(0, thta.length)
  TPR.test                =        matrix(0, thta.length)
  # calculate FPR and TPR for train and test data
  for (i in c(1:thta.length)){
    # calculate the FPR and TPR for train data 
    y.hat.train             =        ifelse(prob.train > thta[i], 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train[i]            =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train[i]            =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    # calculate the FPR and TPR for test data 
    y.hat.test              =        ifelse(prob.test > thta[i], 1, 0)
    FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test                  =        sum(y.test==1) # total positives in the data
    N.test                  =        sum(y.test==0) # total negatives in the data
    FPR.test[i]             =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
    TPR.test[i]             =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity  
  }
  
  # plot ROC curve and calculate AUC score
  auc.train     =       sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
  auc.test      =       sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))
  
  auc_train_lasso = append(auc_train_lasso,auc.train)
  auc_test_lasso = append(auc_test_lasso,auc.test)
  period_lasso = append(period_lasso,t)
}

#####################################################################################################
#####################################################################################################

df_lasso = data.frame(lambda_min_lasso,num_positive_test_lasso,num_positive_train_lasso,auc_test_lasso,auc_train_lasso,period_lasso)
df_lasso_coeff_beta0 = data.frame(coef_beta0)
df_lasso_coeff_beta  = data.frame(coef_beta)
write.csv(df_lasso, file = "df_lasso_newest.csv")
write.csv(df_lasso_coeff_beta0,file = 'df_lasso_coeff_beta0_newest.csv')
write.csv(df_lasso_coeff_beta,file = 'df_lasso_coeff_beta_newest.csv')

#####################################################################################################
#####################################################################################################
alldata = csv_result_5year
df_x = subset(alldata,select = -c(class))
df_x = data.matrix(df_x)
df_y = alldata$class 

n.all               =        length(df_y)
n.P                 =        sum(df_y)
n.N                 =        n.all - n.P
ww                  =        rep(1,n.all)
ww[df_y==1]         =        n.N/n.P

start_time  = Sys.time()
cv.lasso    =    cv.glmnet(df_x, df_y, family = "binomial", alpha = 1,  nfolds = 10, type.measure="auc",
                           weights = ww,nlambda=100,lambda.min.ratio=1e-3,maxit= 500,
                           thresh=1e-3) # 5
end_time = Sys.time()
alllasso = end_time - start_time

# using min lambda to train model
lasso       =     glmnet(df_x, df_y, lambda = cv.lasso$lambda.min, family = "binomial", alpha = 1,weights = ww,
                         nlambda=100,lambda.min.ratio=1e-3,maxit= 500,thresh=1e-3)
lasso_coeff_beta4 = lasso$beta
lasso_coeff_beta04 = lasso$a0
plot(cv.lasso)
test = lasso_coeff_beta4
b = as.data.frame(summary(test))
write.csv(b,file = 'df_all_data_lasso_coeff.csv')


