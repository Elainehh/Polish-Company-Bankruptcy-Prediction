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
#install.packages('randomForest')
library(randomForest)

# import dataset
library(readxl)
csv_result_5year <- read_excel("Downloads/STA 9891/final project data/csv_result-5year.xlsx")
#View(csv_result_5year)
# checking which columns have null values
allmisscols <- sapply(csv_result_5year, function(x) all(is.na(x) | x == '' ))
# replace missing value with average value
csv_result_5year[is.na(csv_result_5year)] = 0


###### repeat modeling 50 times #####
# create vectors to save AUC for training and testing
auc_train_tree = c()
auc_test_tree = c()
period_tree = c()
p = dim(csv_result_5year)[2] - 1
num_positive_train_tree = c()
num_positive_test_tree = c()

for (i in c(1:50)){
  # split dataset into training and testing
  dt = sort(sample(nrow(csv_result_5year),nrow(csv_result_5year)*0.9))
  train = csv_result_5year[dt,]
  test = csv_result_5year[-dt,]
  y.train = train$class
  num_positive_train_tree = append(num_positive_train_tree,sum(y.train==1))
  y.test  = test$class
  num_positive_test_tree = append(num_positive_test_tree,sum(y.test==1))
  
  train$class <- as.character(train$class)
  train$class <- as.factor(train$class)
  test$class <- as.character(test$class)
  test$class <- as.factor(test$class)
  # get predictors and label
  # X.train = subset(train,select = -c(id,class))
  # y.train = train$class
  # X.test = subset(test,select = -c(id,class))
  # y.test = test$class
  ##### random forest tree #######
  start_time =    Sys.time()
  rf.fit     =    randomForest(class~., data = train, mtry = sqrt(p),
                               samplesize = c("0" = 200, "1" = 100), 
                               strata = as.factor(train$class))
  end_time   =    Sys.time()
  t = end_time - start_time
  # predict probability for train and test
  prob.train      =    predict(rf.fit, train,type = 'prob')
  prob.test      =    predict(rf.fit, test, type = 'prob')
  
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
    y.hat.train             =        ifelse(prob.train[,2] > thta[i], 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train[i]            =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train[i]            =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    # calculate the FPR and TPR for test data 
    y.hat.test              =        ifelse(prob.test[,2] > thta[i], 1, 0)
    FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test                  =        sum(y.test==1) # total positives in the data
    N.test                  =        sum(y.test==0) # total negatives in the data
    FPR.test[i]             =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
    TPR.test[i]             =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity  
  }
  
  # calculate AUC score
  auc.train     =       sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
  auc.test      =       sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))
  
  auc_train_tree = append(auc_train_tree,auc.train)
  auc_test_tree = append(auc_test_tree,auc.test)
  period_tree = append(period_tree,t)
}

#####################################################################################################
#####################################################################################################
df_tree = data.frame(num_positive_test_tree,num_positive_train_tree,auc_test_tree,auc_train_tree,period_tree)
df_tpr_fpr_tree = data.frame(TPR.train,FPR.train,TPR.test,FPR.test)  
write.csv(df_tree, file = "df_tree_NEWEST.csv")
write.csv(df_tpr_fpr_tree,file = 'df_tpr_fpr_tree_NEWEST.csv')

#####################################################################################################
#####################################################################################################

alldata = csv_result_5year

alldata$class <- as.character(alldata$class)
alldata$class <- as.factor(alldata$class)
p = dim(csv_result_5year)[2] - 1

start_time  = Sys.time()
rf.fit     =    randomForest(class~., data = alldata, mtry = sqrt(p))
end_time = Sys.time()
allttree = end_time - start_time
i_scores <- rf.fit$importance
b = data.frame(i_scores)
write.csv(b,file = 'df_all_data_tree_parameters.csv')
