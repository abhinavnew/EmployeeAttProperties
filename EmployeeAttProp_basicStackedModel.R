
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
library(DMwR)
library(e1071)
library(glmnet)
library(pROC)
library(funModeling)


##Notations off and clear all objects
options(scipen = 999)
rm(list = ls())
gc()

starttime=Sys.time()

flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

zerovari <- function(dat) {
  out <- lapply(dat, function(x) length(unique(x)))
  want <- which(!out > 1)
  unlist(want)
}


lift <- function(depvar, predcol, groups=10) {
  if(!require(dplyr)){
    install.packages("dplyr")
    library(dplyr)}
  if(is.factor(depvar)) depvar <- as.integer(as.character(depvar))
  if(is.factor(predcol)) predcol <- as.integer(as.character(predcol))
  helper = data.frame(cbind(depvar, predcol))
  helper[,"bucket"] = ntile(-helper[,"predcol"], groups)
  gaintable = helper %>% group_by(bucket)  %>%
    summarise_at(vars(depvar), funs(total =n(),
                                    totalresp=sum(., na.rm = TRUE))) %>%
    mutate(Cumresp = cumsum(totalresp),
           Gain=Cumresp/sum(totalresp)*100,
           Cumlift=Gain/(bucket*(100/groups)))
  return(gaintable)
}

costfunc <- function(FN,FP,Weight=10){
  cal_cost=round((FP+(Weight*FN))/nrow(Test_set),3)
  return(cal_cost)
}

empdata_orig=fread("E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\ibm-hr-analytics-employee-attrition-performance\\WA_Fn-UseC_-HR-Employee-Attrition.csv",
                   data.table = FALSE,
                   colClasses =c("integer","factor","factor","integer","factor","integer","factor","factor","integer","integer",                                  
                                 "factor","factor","integer","factor","factor","factor","factor","factor","integer","integer",
                                 "integer","factor","factor","integer","factor","factor","integer","factor","integer","integer",
                                 "factor","integer","integer","integer","integer" ))


empdata=empdata_orig
dim(empdata)
##chisq.test(empdata$Attrition,empdata$BusinessTravel.Non.Travel)

empdata=na.omit(empdata)
dim(empdata)


##Check for unique values ie Zero variance check -drop columns where values are constant
k=lapply(empdata,function(x) {length(unique(x))})
w=which(!k>1)
names(w)
empdata=empdata[,-which(names(empdata) %in% names(w))]
empdata=empdata[,-which(names(empdata) %in% c("EmployeeNumber"))]

##******Feature engineering********   

##Adding age group 
empdata$AgeGroups=as.factor(ifelse(empdata$Age<=24,"Young",ifelse((empdata$Age>24 & empdata$Age<=54),"Middle Aged","Senior Citizen")))
table(empdata$AgeGroups)
empdata$AgeGroups=as.numeric(empdata$AgeGroups,levels=levels(empdata$AgeGroups))-1
##Adding total satisfaction 
empdata$TotalSatisfaction=as.numeric(empdata$EnvironmentSatisfaction)+as.numeric(empdata$JobInvolvement)+as.numeric(empdata$JobSatisfaction)+as.numeric(empdata$RelationshipSatisfaction)+as.numeric(empdata$WorkLifeBalance)
summary(empdata$TotalSatisfaction)
##Adding Low or High Income indicator if Monthly income less or greater than mean
empdata$IncomeInd=as.factor(ifelse(empdata$MonthlyIncome<mean(empdata$MonthlyIncome),"Low","High"))
empdata$IncomeInd=as.numeric(factor(empdata$IncomeInd),levels=levels(empdata$IncomeInd))-1

##LABEL ENCODING-converting factor columms to numeric for feeding to various alogorithms 
empdata$Gender=as.numeric(factor(empdata$Gender),levels=levels(empdata$Gender))-1
empdata$OverTime=as.numeric(factor(empdata$OverTime),levels=levels(empdata$OverTime))-1

##Create onehot encoded variables using dummyvars
dmy=dummyVars("Attrition ~ . ",data=empdata)
tnsf=data.frame(predict(dmy,newdata=empdata))
head(tnsf,1)
write.csv(tnsf,"E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\Onehotencoded.csv")
empdata$Attrition=as.numeric(factor(empdata$Attrition),levels=levels(empdata$Attrition))-1
tnsf$Attrition=empdata$Attrition
glimpse(tnsf)
## multicollinearity check

res=cor(tnsf)
a=round(res,2)
##write.csv(a,"E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\a.csv")

res2=rcorr(as.matrix(empdata))
flattenCorrMatrix(res2$r,res2$P)

tmp=cor(tnsf)
tmp[upper.tri(tmp)]<- 0
diag(tmp)<-0
remove=names(which(sapply(as.data.frame(tmp),function(x) any(abs(x)>0.90))=="TRUE")) 
remove
dim(tnsf)
tnsf=tnsf[,-c(which(names(tnsf) %in% remove))]
dim(tnsf)
rm(empdata)
empdata=tnsf
dim(empdata)
##should be less columns now than before

##Check for missing values 
a=colnames(empdata)[colSums(is.na(empdata)>0)]
a
rowSums(is.na(empdata))
rowSums((is.na(lrdata)))

set.seed(115)
##breaking data into Train+validate and TEST sets
splitt1=sample.split(empdata$Attrition,SplitRatio = 0.6)
Trainfull_set=subset(empdata,splitt1==TRUE)
Test_set=subset(empdata,splitt1==FALSE)
nrow(Trainfull_set)+nrow(Test_set)
## should be = 1470

splitt2=sample.split(Trainfull_set$Attrition,SplitRatio = 0.6)
Train_set=subset(Trainfull_set,splitt2==TRUE)
Val_set=subset(Trainfull_set,splitt2==FALSE)

##*********************************************************

##using lasso regression for feature selection 

##lassotrain=Train_set
##lassotest=Test_set 

##x.train=data.matrix(lassotrain[,1:ncol(lassotrain)-1])
##y.train=data.matrix(lassotrain$Attrition)

##lambda_seq=10^seq(2,-2,by=-.1)


##cv.out=cv.glmnet(x.train,y.train,family="binomial",alpha=0.5,type.measure = "auc",lambda = lambda_seq)
##plot(cv.out)

##co=coef(cv.out,s="lambda.min")

##best.lambda=cv.out$lambda.1se
##best.lambda

##lasso.mod=glmnet(x.train,y.train,family = "binomial",alpha = 0.5,lambda = best.lambda)
##lasso.mod$beta[,1]

##test_lasso=data.matrix(lassotest[,-which(names(lassotest) %in% c("Attrition"))])
##p<-predict(lasso.mod, s=best.lambda, newx =test_lasso )
##length(p)
##variables of importance 
##coef(lasso.mod)
##plot(lasso.mod,xvar="lambda",label=TRUE)
##*************************************************************************

## using recursive feature elimination RFE for feature selection 

##subsets <- c(1:5, 10, 15, 18)

##ctrl <- rfeControl(functions = rfFuncs,
##            method = "repeatedcv",
##           repeats = 5,
##           verbose = FALSE)

##lmProfile <- rfe(x=x.train, y=y.train,
##             sizes = subsets,
##             rfeControl = ctrl)

##Important=lmProfile$optVariables

##**************************************************************
newtrain=Train_set
newval=Val_set
rftrain=Train_set
rfval=Val_set
logtrain=Train_set
logval=Val_set

##newtrain$Attrition=as.numeric(factor(newtrain$Attrition),levels=levels(newtrain$Attrition))-1

actual_labels=Test_set[,"Attrition"]
Test_set$Attrition <-NA
head(Test_set)
rftest=Test_set
lrtest=Test_set
# 
# ##SMOT the trainset
# dim(newtrain)
# glimpse(newtrain)
# object.size(newtrain)
# Classcount=table(newtrain$Attrition)
# # Over Sampling
# over = ( (0.6 * max(Classcount)) - min(Classcount) ) / min(Classcount)
# # Under Sampling
# under = (0.4 * max(Classcount)) / (min(Classcount) * over)
# 
# over = round(over, 1) * 100
# under = round(under, 1) * 100
# #Generate the balanced data set using SMOT 
# newtrain$Attrition <- as.factor(newtrain$Attrition)
# newtrain=SMOTE(Attrition ~ .,data=newtrain,perc.over = over,perc.under = under,k=5)
# prop.table(table(newtrain$Attrition))
# dim(newtrain)
# newtrain$Attrition=as.numeric(factor(newtrain$Attrition),levels=levels(newtrain$Attrition))-1


###2 Layered stacking--Initial layer of models####

##xgboost trainset prep
tr_labels=newtrain[,"Attrition"]
newtrain$Attrition <- NULL
x=as.matrix(newtrain)
x=matrix(as.numeric(x),nrow(x),ncol(x))
dtrain=xgb.DMatrix(data=x,label=tr_labels,missing = NA)
dim(dtrain)
##xgboost validateset prep
vl_labels=newval[,"Attrition"]
newval$Attrition<- NULL
y=as.matrix(newval)
y=matrix(as.numeric(y),nrow(y),ncol(y))
dval=xgb.DMatrix(data=y,label=vl_labels,missing = NA)
dim(dval)
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
##fitcv=xgb.cv(params = param,
## data=dtrain,
##nrounds = 100000,
## watchlist=watch,
##nfold = 5,
## early_stopping_rounds = 10,
##verbose = 2)
##xgboost main model
set.seed(111)
xgb_mod1=xgb.train(params = param,
                   data=dtrain,
                   watchlist = watch,
                   nrounds = 600,
                   verbose = 1)
##making predictions on unseen data
pred_xg_val=predict(xgb_mod1,newdata=dval,type="prob")
length(pred_xg_val)
meta_trainset=as.data.frame(cbind(vl_labels,pred_xg_val))
dim(meta_trainset)
res.rocxg_val=roc(response=vl_labels,predictor=pred_xg_val,positive=1)
pROC::auc(res.rocxg_val)
##with different weights: best.weights = c(Cost of FN, Prevalence) default-prevalence is 0.5 and cost is 1 so that no weight is applied in effect.
t2=coords(res.rocxg_val,x="best",ret="threshold",transpose=FALSE,best.weights = c(5,0.16))
thresh=t2
finalpred_classes=ifelse(pred_xg_val>thresh,1,0)
cm=confusionMatrix(as.factor(finalpred_classes),as.factor(vl_labels))
fnxg=cm$table[1,2]
fpxg=cm$table[2,1]
accxg=cm$overall[1]
costxg=costfunc(fnxg,fpxg)
plot.roc(res.rocxg_val,print.auc = TRUE,print.thres = "best")

pred_xg=predict(xgb_mod1,newdata=dtest,type="prob")
length(pred_xg)
meta_testset=as.data.frame(cbind(pred_xg))
dim(meta_testset)

##random forest prep
dim(rftrain)
glimpse(rftrain)
rftrain$Attrition=make.names(rftrain$Attrition)
rftrain$Attrition=as.factor(rftrain$Attrition)

vl_rf_labels=rfval$Attrition
rfval$Attrition=as.factor(make.names(rfval$Attrition))


trcontrolobj=trainControl(method="cv",verboseIter = TRUE,classProbs = TRUE,summaryFunction = twoClassSummary)
tgrid=expand.grid(.mtry=c(2,4,8,15))
set.seed(111)
rf_mod1=train(Attrition ~.,
              data=rftrain,
              method="rf",
              metric="ROC",
              trControl=trcontrolobj,
              verbose=T)

pred_rf=predict(rf_mod1,newdata=rfval,type="prob")
length(pred_rf)
##pred_rfRoc=ifelse(pred_rf$X1>pred_rf$X0,1,0)
res.rocrf=roc(response=vl_rf_labels,predictor=pred_rf$X1)
meta_trainset=cbind(meta_trainset,pred_rf_val=pred_rf$X1)
dim(meta_trainset)
##with different weights: best.weights = c(Cost of FN, Prevalence) default-prevalence is 0.5 and cost is 1 so that no weight is applied in effect.
t3=coords(res.rocrf,x="best",ret = "threshold",transpose=FALSE,best.weights = c(5,0.16))
thresh=t3
finalpred_classes_rf=ifelse(pred_rf$X1>thresh,1,0)
cm=confusionMatrix(as.factor(finalpred_classes_rf),as.factor(vl_rf_labels))
accrf=cm$overall[1]
fnrf=cm$table[1,2]
fprf=cm$table[2,1]
costrf=costfunc(fnrf,fprf)
print(costrf)
plot.roc(res.rocrf,print.auc = TRUE,print.thres = "best")
pROC::auc(res.rocrf)

pred_rf_test=predict(rf_mod1,newdata=rftest,type="prob")
meta_testset=cbind(meta_testset,pred_rf=pred_rf_test$X1)
dim(meta_testset)

##logistic regression for 0/1 classification
set.seed(111)
dim(empdata)
dim(logtrain)
##refine trainset based on significant variables -feature selection as per significance shown
prop.table(table(logtrain$Attrition))
logi_mod1=glm(Attrition ~ .,data=logtrain,family = binomial)
summary(logi_mod1)
imp=varImp(logi_mod1)
print(imp)

logtrain_refined=logtrain[,which(names(logtrain) %in% c(rownames(imp),"Attrition"))]
dim(logtrain_refined)
dim(lrtest)
lrtest_refined=lrtest[,which(names(lrtest) %in% c(rownames(imp)))]
dim(logval)
logval_refined=logval[,which(names(logval) %in% c(rownames(imp)))]
dim(logval_refined)

logi_mod2=glm(Attrition ~.,data=logtrain_refined,family = binomial)
summary(logi_mod2)
pred_lr_val=predict(logi_mod2,newdata = logval_refined,type="response")
length(pred_lr_val)
##Back to finding threshold as per customer cost function FP=1 and FN=10 Cost func=(FP+10FN)/N
res.roclr=roc(response=vl_labels,predictor=pred_lr_val,positive=1)
pROC::auc(res.roclr)
##with different weights: best.weights = c(Cost of FN, Prevalence) default-prevalence is 0.5 and cost is 1 so that no weight is applied in effect.
t4=coords(res.roclr,x="best",ret = "threshold",transpose=FALSE,best.weights = c(5,0.16))
thresh=t4
print(thresh)
finalpred_classes_lr=ifelse(pred_lr_val>thresh,1,0)
cm=confusionMatrix(as.factor(finalpred_classes_lr),as.factor(vl_labels))
acclr=cm$overall[1]
fnlr=cm$table[1,2]
fplr=cm$table[2,1]
costlr=costfunc(fnlr,fplr)
print(costlr)
plot.roc(res.roclr,print.auc = TRUE,print.thres = "best")
meta_trainset=cbind(meta_trainset,pred_lr_val)
head(meta_trainset)

pred_lr_test=predict(logi_mod2,newdata = lrtest_refined,type="response")
length(pred_lr_test)
meta_testset=cbind(meta_testset,pred_lr_test)
head(meta_testset)

# ##Lift analysis on the predictions from logistic regression 
# dt=lift(actual_labels,pred_lr,groups = 10)
# print(dt)
# plot(dt$bucket, dt$Cumlift, type="l", ylab="Cumulative lift", xlab="Bucket")
# write.csv(dt,"E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\liftanalysis.csv",row.names=FALSE)
# 
# ##Weighthed average of previous models 
# pred_wtd_avg=((0.6*pred_lr)+ (0.3*(pred_rf$X1))+(0.1*pred_xg))/3
# 
# res.rocavg=roc(actual_labels,pred_wtd_avg)


##2nd level of MODEL STACK -GBM/caret
# passing prev predictions and test set to 2nd level of gbm model 

trctrlobj=trainControl(method="cv",number=5,verboseIter=FALSE,classProbs = TRUE,summaryFunction = twoClassSummary)
tgrid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))
meta_trainset$vl_labels=as.factor(make.names(meta_trainset$vl_labels))
lev=levels(meta_trainset$vl_labels)
print(lev)
meta_trainset$vl_labels=relevel(meta_trainset$vl_labels,"X1")
gbmmod1=train(vl_labels ~.,
              data=meta_trainset,
              method="gbm",
              metric="ROC",
              trControl=trctrlobj,
              tuneGrid=tgrid,
              verbose=FALSE
)
summary(gbmmod1)
colnames(meta_testset)<-c("pred_xg_val","pred_rf_val","pred_lr_val")
pred_final_gbm=predict(gbmmod1,newdata=meta_testset,type='prob')
fin=pred_final_gbm$X1
res.rocfin=roc(response=actual_labels,predictor=fin,positive=1)
pROC::auc(res.rocfin)



cat("cost of xgboost model is =",costxg)
cat("Accuracy of xgboost model is=",accxg)
cat("AUC of xgboost model is=",pROC::auc(res.rocxg))

cat("cost of RF model is =",costrf)
cat("Accuracy of RF model is=",accrf)
cat("AUC of RF model is=",pROC::auc(res.rocrf))

cat("cost of LR model is =",costlr)
cat("Accuracy of LR model is=",acclr)
cat("AUC of LR model is=",pROC::auc(res.roclr))

cat("AUC of stacked model is =",pROC::auc(res.rocfin))

plot(res.rocxg_val,ylim = c(0,1),print.thres=T,print.thres.cex=0.8,main="ROC Curves",col="blue")
plot(res.rocrf,ylim = c(0,1),print.thres=T,print.thres.cex=0.8,col="green",add=T)
plot(res.roclr,ylim = c(0,1),print.thres=T,print.thres.cex=0.8,col="red",add=T)
plot(res.rocfin,ylim = c(0,1),print.thres=T,print.thres.cex=0.8,col="magenta",add=T)


endtime=Sys.time()
timetaken=endtime-starttime
print(timetaken)











