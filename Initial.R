
library(caret)
library(tidyr)
library(plyr)
library(dplyr)
library(caTools)
library(reshape2)
library(gbm)
library(caTools)
library(randomForest)
library(ggplot2)
library(data.table)
library(xgboost)
library(Matrix)
library(lightgbm)
library(TeachingDemos)


##Notations off and clear all objects
options(scipen = 999)
rm(list = ls())
gc()

startime=Sys.time()



empdata_orig=fread("E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\ibm-hr-analytics-employee-attrition-performance\\WA_Fn-UseC_-HR-Employee-Attrition.csv",
                    data.table = FALSE,
                    colClasses =c("integer","factor","factor","integer","factor","integer","integer","factor","integer","integer",                                  
                                  "integer","factor","factor","factor","factor","factor","integer","factor","integer","factor",
                                   "factor","factor","factor","factor","integer","factor","integer","factor","integer","integer",
                                  "factor","integer","integer","integer","integer" ))



##breaking data into Train+validate and TEST sets
splitt1=sample.split(empdata_orig$Attrition,SplitRatio = 0.6)
TV_set=subset(empdata_orig,splitt1==TRUE)
Test_set=subset(empdata_orig,splitt1==FALSE)


splitt2=sample.split(TV_set$Attrition,SplitRatio = 0.7)
Train_Set=subset(TV_set,splitt2==TRUE)
Validate_set=subset(TV_set,splitt2==FALSE)


dim(Train_Set)
dim(Validate_set)
dim(Test_set)


nrow(Train_Set)+nrow(Validate_set)+nrow(Test_set)
## should be = 1470


