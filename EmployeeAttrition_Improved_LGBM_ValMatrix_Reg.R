
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
library(Hmisc)
library(pROC)


##Notations off and clear all objects
options(scipen = 999)
rm(list = ls())
gc()

startime=Sys.time()

flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

empdata_orig=fread("E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\ibm-hr-analytics-employee-attrition-performance\\WA_Fn-UseC_-HR-Employee-Attrition.csv",
                   data.table = FALSE,
                   colClasses =c("integer","factor","factor","integer","factor","integer","integer","factor","integer","integer",                                  
                                 "integer","factor","integer","factor","factor","factor","factor","factor","integer","integer",
                                 "integer","factor","factor","integer","factor","factor","integer","factor","integer","integer",
                                 "factor","integer","integer","integer","integer" ))


empdata=empdata_orig


##converting factor columms to numeric for feeding to various alogorithms 

empdata$Attrition=as.numeric(factor(empdata$Attrition),levels=levels(empdata$Attrition))-1
empdata$BusinessTravel=as.numeric(factor(empdata$BusinessTravel),levels=levels(empdata$BusinessTravel))-1
empdata$Department=as.numeric(factor(empdata$Department),levels=levels(empdata$Department))-1

empdata$EducationField=as.numeric(factor(empdata$EducationField),levels=levels(empdata$EducationField))-1
empdata$Gender=as.numeric(factor(empdata$Gender),levels=levels(empdata$Gender))-1
empdata$HourlyRate=as.numeric(factor(empdata$HourlyRate),levels=levels(empdata$HourlyRate))-1

empdata$JobInvolvement=as.numeric(factor(empdata$JobInvolvement),levels=levels(empdata$JobInvolvement))-1
empdata$JobLevel=as.numeric(factor(empdata$JobLevel),levels=levels(empdata$JobLevel))-1
empdata$JobRole=as.numeric(factor(empdata$JobRole),levels=levels(empdata$Jobrole))-1

empdata$JobSatisfaction=as.numeric(factor(empdata$JobSatisfaction),levels=levels(empdata$JobSatisfaction))-1
empdata$MaritalStatus=as.numeric(factor(empdata$MaritalStatus),levels=levels(empdata$MaritalStatus))-1
empdata$Over18=as.numeric(factor(empdata$Over18),levels=levels(empdata$over18))-1
empdata$OverTime=as.numeric(factor(empdata$OverTime),levels=levels(empdata$OverTime))-1

empdata$PerformanceRating=as.numeric(factor(empdata$PerformanceRating),levels=levels(empdata$PerformanceRating))-1
empdata$RelationshipSatisfaction=as.numeric(factor(empdata$RelationshipSatisfaction),levels=levels(empdata$RelationshipSatisfaction))-1
empdata$StockOptionLevel=as.numeric(factor(empdata$StockOptionLevel),levels=levels(empdata$StockOptionLevel))-1
empdata$WorkLifeBalance=as.numeric(factor(empdata$WorkLifeBalance),levels=levels(empdata$WorkLifeBalance))-1

## multicollinearity check

res=cor(empdata)
a=round(res,2)
write.csv(a,"E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\a.csv")

res2=rcorr(as.matrix(Train_set))
flattenCorrMatrix(res2$r,res2$P)

tmp=cor(empdata)
tmp[upper.tri(tmp)]<- 0
diag(tmp)<-0
remove=names(which(sapply(as.data.frame(tmp),function(x) any(abs(x)>0.90))=="TRUE")) 
remove
dim(empdata)
empdata=empdata[,-c(which(names(empdata) %in% remove))]
dim(empdata)
##should be less columns now than before


##Check for missing values 

##apply(is.na(empdata),2,sum)

a=colnames(empdata)[colSums(is.na(empdata)>0)]
a
rowSums(is.na(empdata))




##breaking data into Train+validate and TEST sets
splitt1=sample.split(empdata$Attrition,SplitRatio = 0.6)
TV_set=subset(empdata,splitt1==TRUE)
Test_set=subset(empdata,splitt1==FALSE)

splitt2=sample.split(TV_set$Attrition,SplitRatio = 0.7)
Train_set=subset(TV_set,splitt2==TRUE)
Validate_set=subset(TV_set,splitt2==FALSE)

dim(Train_set)
dim(Validate_set)
dim(Test_set)

nrow(Train_set)+nrow(Validate_set)+nrow(Test_set)
## should be = 1470




tr_labels=Train_set[,"Attrition"]
Train_set$Attrition <- NULL
x=as.matrix(Train_set)
x=matrix(as.numeric(x),nrow(x),ncol(x))
dtrain=xgb.DMatrix(data=x,label=tr_labels,missing = NA)

tv_labels=Validate_set[,"Attrition"]
Validate_set$Attrition<-NULL
v=as.matrix(Validate_set)
v=matrix(as.numeric(v),nrow(v),ncol(y))
dvalidate=xgb.DMatrix(data=v,label=tv_labels,missing = NA)


actual_labels=Test_set[,"Attrition"]
Test_set$Attrition <-NA
head(Test_set)
ts_labels=Test_set[,"Attrition"]
Test_set$Attrition <-NULL

dtest=xgb.DMatrix(data=as.matrix(Test_set),label=ts_labels,missing = NA)

param=list(booster="gblinear",
           objective="binary:logistic",
           eval_metric="auc",
           eta=0.01,
           lambda=5,
           lambda_bias=0,
           alpha=2)

watch=list(train=dtrain,test=dvalidate)
set.seed(115)
##fitcv=xgb.cv(params = param,
            ## data=dtrain,
             ##nrounds = 100000,
            ## watchlist=watch,
             ##nfold = 5,
            ## early_stopping_rounds = 10,
             ##verbose = 2)

##xgboost main model


xgb_mod1=xgb.train(params = param,
                   data=dtrain,
                   watchlist = watch,
                   nrounds = 600,
                   verbose = 1)


##making predictions on unseen data

pred=predict(xgb_mod1,newdata=dvalidate)

length(pred)

## creating roc /auc and making class predictions based on best threshold
res.roc=roc(tv_labels,pred)

plot.roc(res.roc,print.auc = TRUE,print.thres = "best")
t=coords(res.roc,"best","threshold",transpose = FALSE)
thresh=t[1]
auc(res.roc)

pred_classes=ifelse(pred>thresh,1,0)

finaldf=data.frame(predicted=pred_classes,actuals=tv_labels)
table(finaldf$predicted,finaldf$actuals)






