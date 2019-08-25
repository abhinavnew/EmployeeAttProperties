
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
library(DMwR)
library(e1071)


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
Train_set=subset(empdata,splitt1==TRUE)
Test_set=subset(empdata,splitt1==FALSE)

##splitt2=sample.split(TV_set$Attrition,SplitRatio = 0.7)
##Train_set=subset(TV_set,splitt2==TRUE)
##Validate_set=subset(TV_set,splitt2==FALSE)

##dim(Train_set)
##dim(Validate_set)
##dim(Test_set)

nrow(Train_set)+nrow(Test_set)
## should be = 1470

table(Train_set$Attrition)
table(Test_set$Attrition)

##SMOTE'd trainset 

newtrain=Train_set

Classcount=table(newtrain$Attrition)

# Over Sampling
over = ( (0.6 * max(Classcount)) - min(Classcount) ) / min(Classcount)
# Under Sampling
under = (0.4 * max(Classcount)) / (min(Classcount) * over)

over = round(over, 1) * 100
under = round(under, 1) * 100
#Generate the balanced data set

newtrain$Attrition <- as.factor(newtrain$Attrition)

newtrain=SMOTE(Attrition ~ .,data=newtrain,perc.over = over,perc.under = under,k=5)


prop.table(table(newtrain$Attrition))

rftrain=newtrain
logtrain=newtrain

newtrain$Attrition=as.numeric(factor(newtrain$Attrition),levels=levels(newtrain$Attrition))-1

dim(newtrain)

glimpse(newtrain)

object.size(newtrain)


##xgboost trainset prep

tr_labels=newtrain[,"Attrition"]
newtrain$Attrition <- NULL
x=as.matrix(newtrain)
x=matrix(as.numeric(x),nrow(x),ncol(x))
dtrain=xgb.DMatrix(data=x,label=tr_labels,missing = NA)

dim(dtrain)


actual_labels=Test_set[,"Attrition"]
Test_set$Attrition <-NA
head(Test_set)
rftest=Test_set
lrtest=Test_set


##xgboost testset prep

ts_labels=Test_set[,"Attrition"]
Test_set$Attrition <-NULL
dtest=xgb.DMatrix(data=as.matrix(Test_set),label=ts_labels,missing = NA)

dim(dtest)

param=list(booster="gblinear",
           objective="binary:logistic",
           eval_metric="auc",
           eta=0.01,
           lambda=5,
           lambda_bias=0,
           alpha=2)

watch=list(train=dtrain,test=dtest)
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

out=output.capture(xgb_mod1)
write.csv(out,"E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\xgbmodelOut.csv")

##making predictions on unseen data
finalpred=predict(xgb_mod1,newdata=dtest)
length(finalpred)
res.roc2=roc(actual_labels,finalpred)
plot.roc(res.roc2,print.auc = TRUE,print.thres = "best")
auc(res.roc2)
t2=coords(res.roc2,"best","threshold",transpose=FALSE)
thresh=t2[1,1]
finalpred_classes=ifelse(finalpred>thresh,1,0)
mydf=data.frame(predicted=finalpred_classes,actuals=actual_labels)
table(mydf$predicted,mydf$actuals)


##random forest prep

rftrain$Attrition=make.names(rftrain$Attrition)
trcontrolobj=trainControl(method="cv",verboseIter = TRUE,classProbs = TRUE,summaryFunction = twoClassSummary)
tgrid=expand.grid(.mtry=c(2,4,8,15))

rf_mod1=train(Attrition ~.,
              data=rftrain,
              method="rf",
              metric="ROC",
              trControl=trcontrolobj,
              verbose=T)

pred_rf=predict(rf_mod1,newdata=rftest,type="raw")
length(pred_rf)
mydf=data.frame(predicted=pred_rf,actuals=actual_labels)
table(mydf$predicted,mydf$actuals)
pred_rfRoc=ifelse(pred_rf=="X1",1,0)
res.roc3=roc(actual_labels,pred_rfRoc)
auc(res.roc3)


##logistic regression for 0/1 classification

logi_mod1=glm(Attrition ~ .,data=logtrain,family = binomial)
summary(logi_mod1)

pred_lr=predict(logi_mod1,newdata = lrtest,type="response")
length(pred_lr)


res.roc4=roc(actual_labels,pred_lr)
plot.roc(res.roc3,print.auc = TRUE,print.thres = "best")
auc(res.roc4)
t3=coords(res.roc4,"best","threshold",transpose=FALSE)
thresh=t3[1,1]
finalpred_classes_lr=ifelse(finalpred>thresh,1,0)
mydf=data.frame(predicted=finalpred_classes_lr,actuals=actual_labels)
table(mydf$predicted,mydf$actuals)










