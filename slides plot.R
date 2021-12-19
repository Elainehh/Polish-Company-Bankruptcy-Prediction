library(class)
library(ggplot2)
library(dplyr)
library(glmnet)
#library(alocv)
library(rmutil)
library(tictoc)
library(latex2exp)
#nstall.packages("naniar")
#install.packages('randomForest')
library(naniar)


##############################################################################################################
# auc boxplot for ridge
library(readxl)
X50_auc_test <- read_excel("Downloads/STA 9891/final project data/50_auc_test.xlsx")
View(X50_auc_test)
boxplot(X50_auc_test$auc_test~X50_auc_test$model, main = 'AUC of Test Dataset by Model',ylab = 'AUC',
        ylim = c(0.60,1),las = 1,xlab = 'Model')

library(readxl)
X50_auc_train <- read_excel("Downloads/STA 9891/final project data/50_auc_train.xlsx")
View(X50_auc_train)
boxplot(X50_auc_train$auc_train~X50_auc_train$model, main = 'AUC of Train Dataset by Model',ylab = 'AUC',
        ylim = c(0.60,1),las = 1,xlab = 'Model')

###################################################################################################################
### barplot of coefficients and importance
install.packages('patchwork')
library(patchwork)
library(readxl)
barplot_coefficients <- read_excel("Downloads/STA 9891/final project data/all data/barplot_coefficients.xlsx")
View(barplot_coefficients)

# standardize coefficients and importance of tree
barplot_coefficients$ElasticNet = (barplot_coefficients$ElasticNet - mean(barplot_coefficients$ElasticNet))/sd(barplot_coefficients$ElasticNet)
barplot_coefficients$Lasso = (barplot_coefficients$Lasso - mean(barplot_coefficients$Lasso))/sd(barplot_coefficients$Lasso)
barplot_coefficients$Ridge = (barplot_coefficients$Ridge - mean(barplot_coefficients$Ridge))/sd(barplot_coefficients$Ridge)
barplot_coefficients$Tree = (barplot_coefficients$Tree - mean(barplot_coefficients$Tree))/sd(barplot_coefficients$Tree)

p1 = ggplot(barplot_coefficients, aes(x = reorder(Feature,-ElasticNet), y = ElasticNet, width=0.5)) + 
  labs(y = "Elastic-Net", x = NULL, fill = NULL) +
  geom_bar(stat = "identity") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ggtitle("Coefficients/Importance for 4 Models")

p2 = ggplot(barplot_coefficients, aes(x = reorder(Feature,-ElasticNet), y = Lasso, width=0.5)) + 
  labs(y = "Lasso", x = NULL, fill = NULL) +
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) 

p3 = ggplot(barplot_coefficients, aes(x = reorder(Feature,-ElasticNet), y = Ridge, width=0.5)) + 
  labs(y = "Ridge", x = NULL, fill = NULL) +
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) 

p4 = ggplot(barplot_coefficients, aes(x = reorder(Feature,-ElasticNet), y = Tree, width=0.5)) + 
  labs(y = "Tree", x = NULL, fill = NULL) +
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  labs(x = "Features")

p1 / p2 /p3 /p4

