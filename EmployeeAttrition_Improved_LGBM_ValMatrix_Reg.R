
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

empdata_orig=fread("E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\ibm-hr-analytics-employee-attrition-performance\\WA_Fn-UseC_-HR-Employee-Attrition.csv",
                   data.table = FALSE,
                   colClasses =c("integer","factor","factor","integer","factor","integer","factor","factor","integer","integer",                                  
                                 "factor","factor","integer","factor","factor","factor","factor","factor","integer","integer",
                                 "integer","factor","factor","integer","factor","factor","integer","factor","integer","integer",
                                 "factor","integer","integer","integer","integer" ))


empdata=empdata_orig
dim(empdata)
chisq.test(empdata$Attrition,empdata$BusinessTravel.Non.Travel)

empdata=na.omit(empdata)
dim(empdata)


##Check for unique values ie Zero variance check -drop columns where values are constant
k=lapply(empdata,function(x) {length(unique(x))})
w=which(!k>1)
names(w)
empdata=empdata[,-which(names(empdata) %in% names(w))]

##Create onehot encoded variables using dummyvars
dmy=dummyVars("Attrition ~ . ",data=empdata)
tnsf=data.frame(predict(dmy,newdata=empdata))
head(tnsf,1)
write.csv(tnsf,"E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\Onehotencoded.csv")
empdata$Attrition=as.numeric(factor(empdata$Attrition),levels=levels(empdata$Attrition))-1
tnsf$Attrition=empdata$Attrition
glimpse(tnsf)




##LABEL ENCODING-converting factor columms to numeric for feeding to various alogorithms 

##empdata$Attrition=as.numeric(factor(empdata$Attrition),levels=levels(empdata$Attrition))-1
##empdata$BusinessTravel=as.numeric(factor(empdata$BusinessTravel),levels=levels(empdata$BusinessTravel))-1
##empdata$Department=as.numeric(factor(empdata$Department),levels=levels(empdata$Department))-1
##empdata$Education=as.numeric(factor(empdata$Education),levels=levels(empdata$Education))-1
##empdata$EnvironmentSatisfaction=as.numeric(factor(empdata$EnvironmentSatisfaction),levels=levels(EnvironmentSatisfaction))-1

##empdata$EducationField=as.numeric(factor(empdata$EducationField),levels=levels(empdata$EducationField))-1
##empdata$Gender=as.numeric(factor(empdata$Gender),levels=levels(empdata$Gender))-1
##empdata$HourlyRate=as.numeric(factor(empdata$HourlyRate),levels=levels(empdata$HourlyRate))-1

##empdata$JobInvolvement=as.numeric(factor(empdata$JobInvolvement),levels=levels(empdata$JobInvolvement))-1
##empdata$JobLevel=as.numeric(factor(empdata$JobLevel),levels=levels(empdata$JobLevel))-1
##empdata$JobRole=as.numeric(factor(empdata$JobRole),levels=levels(empdata$Jobrole))-1

##empdata$JobSatisfaction=as.numeric(factor(empdata$JobSatisfaction),levels=levels(empdata$JobSatisfaction))-1
##empdata$MaritalStatus=as.numeric(factor(empdata$MaritalStatus),levels=levels(empdata$MaritalStatus))-1
##empdata$Over18=as.numeric(factor(empdata$Over18),levels=levels(empdata$over18))-1
##empdata$OverTime=as.numeric(factor(empdata$OverTime),levels=levels(empdata$OverTime))-1

##empdata$PerformanceRating=as.numeric(factor(empdata$PerformanceRating),levels=levels(empdata$PerformanceRating))-1
##empdata$RelationshipSatisfaction=as.numeric(factor(empdata$RelationshipSatisfaction),levels=levels(empdata$RelationshipSatisfaction))-1
##empdata$StockOptionLevel=as.numeric(factor(empdata$StockOptionLevel),levels=levels(empdata$StockOptionLevel))-1
##empdata$WorkLifeBalance=as.numeric(factor(empdata$WorkLifeBalance),levels=levels(empdata$WorkLifeBalance))-1

       ## multicollinearity check

res=cor(tnsf)
a=round(res,2)
write.csv(a,"E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\a.csv")

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

##******Feature engineering********   

##Adding age group 
empdata$AgeGroups=as.factor(ifelse(empdata$Age<=24,"Young",ifelse((empdata$Age>24 & empdata$Age<=54),"Middle Aged","Senior Citizen")))
table(empdata$AgeGroups)
empdata$AgeGroups=as.numeric(empdata$AgeGroups,levels=levels(empdata$AgeGroups))-1
##Adding total satisfaction 
empdata$TotalSatisfaction=empdata$EnvironmentSatisfaction+empdata$JobInvolvement+empdata$JobSatisfaction+empdata$RelationshipSatisfaction+empdata$WorkLifeBalance
summary(empdata$TotalSatisfaction)
##Adding Low or High Income indicator if Monthly income less or greater than mean
empdata$IncomeInd=as.factor(ifelse(empdata$MonthlyIncome<mean(empdata$MonthlyIncome),"Low","High"))
empdata$IncomeInd=as.numeric(factor(empdata$IncomeInd),levels=levels(empdata$IncomeInd))-1
##dropping unique column as its not useful for modelling
empdata=empdata[,-which(names(empdata) %in% c("EmployeeNumber"))]
glimpse(empdata)


set.seed(115)

##breaking data into Train+validate and TEST sets
splitt1=sample.split(empdata$Attrition,SplitRatio = 0.6)
Train_set=subset(empdata,splitt1==TRUE)
Test_set=subset(empdata,splitt1==FALSE)
nrow(Train_set)+nrow(Test_set)
## should be = 1470

table(Train_set$Attrition)
table(Test_set$Attrition)

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
rftrain=Train_set
logtrain=Train_set
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

##xgboost trainset prep

tr_labels=newtrain[,"Attrition"]
newtrain$Attrition <- NULL
x=as.matrix(newtrain)
x=matrix(as.numeric(x),nrow(x),ncol(x))
dtrain=xgb.DMatrix(data=x,label=tr_labels,missing = NA)
dim(dtrain)

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

##out=output.capture(xgb_mod1)
##write.csv(out,"E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\xgbmodelOut.csv")

##making predictions on unseen data
finalpred=predict(xgb_mod1,newdata=dtest,type="prob")
length(finalpred)
res.rocxg=roc(actual_labels,finalpred)


t2=coords(res.rocxg,"best","threshold",transpose=FALSE)
thresh=t2[1,1]
finalpred_classes=ifelse(finalpred>thresh,1,0)
mydf=data.frame(predicted=finalpred_classes,actuals=actual_labels)
table(mydf$predicted,mydf$actuals)
confusionMatrix(as.factor(finalpred_classes),as.factor(actual_labels))
pROC::auc(res.rocxg)
plot.roc(res.rocxg,print.auc = TRUE,print.thres = "best")

##random forest prep

rftrain$Attrition=make.names(rftrain$Attrition)
rftrain$Attrition=as.factor(rftrain$Attrition)
trcontrolobj=trainControl(method="cv",verboseIter = TRUE,classProbs = TRUE,summaryFunction = twoClassSummary)
tgrid=expand.grid(.mtry=c(2,4,8,15))
set.seed(111)
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
res.rocrf=roc(actual_labels,pred_rfRoc)

plot.roc(res.rocrf,print.auc = TRUE,print.thres = "best")
pROC::auc(res.rocrf)

##logistic regression for 0/1 classification
set.seed(111)
dim(empdata)
logtrain=Train_set
##SMOTE the trainset so that it becomes balanced in the ratio we want 


prop.table(table(logtrain$Attrition))
logi_mod1=glm(Attrition ~ .,data=logtrain,family = binomial)
summary(logi_mod1)
imp=varImp(logi_mod1)
print(imp)

logtrain_refined=logtrain[,which(names(logtrain) %in% c(rownames(imp),"Attrition"))]
dim(logtrain_refined)
dim(lrtest)
lrtest_refined=lrtest[,which(names(lrtest) %in% c(rownames(imp)))]

logi_mod2=glm(Attrition ~.,data=logtrain_refined,family = binomial)
summary(logi_mod2)
pred_lr=predict(logi_mod2,newdata = lrtest_refined,type="response")
length(pred_lr)

##Lift analysis on the predictions from logistic regression 
dt=lift(actual_labels,pred_lr,groups = 10)
print(dt)
plot(dt$bucket, dt$Cumlift, type="l", ylab="Cumulative lift", xlab="Bucket")
write.csv(dt,"E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\liftanalysis.csv",row.names=FALSE)

res.roclr=roc(actual_labels,pred_lr)

pROC::auc(res.roclr)
t3=coords(res.roclr,"best","threshold",transpose=FALSE)
thresh=t3[1,1]
finalpred_classes_lr=ifelse(finalpred>thresh,1,0)
mydf=data.frame(predicted=finalpred_classes_lr,actuals=actual_labels)
table(mydf$predicted,mydf$actuals)
plot.roc(res.roclr,print.auc = TRUE,print.thres = "best")

pROC::auc(res.rocxg)
pROC::auc(res.rocrf)
pROC::auc(res.roclr)

plot(res.rocxg,ylim = c(0,1),print.thres=T,print.thres.cex=0.8,main="ROC Curves",col="blue")
plot(res.rocrf,ylim = c(0,1),print.thres=T,print.thres.cex=0.8,col="green",add=T)
plot(res.roclr,ylim = c(0,1),print.thres=T,print.thres.cex=0.8,col="red",add=T)











