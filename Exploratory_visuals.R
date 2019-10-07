## Data visualization on Employee attrition data set 

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


empdata_orig=fread("E:\\AbhinavB\\Kaggle\\IBM HR Analytics Employee Attrition\\ibm-hr-analytics-employee-attrition-performance\\WA_Fn-UseC_-HR-Employee-Attrition.csv",
                   data.table = FALSE,
                   colClasses =c("integer","factor","factor","integer","factor","integer","factor","factor","integer","integer",                                  
                                 "factor","factor","integer","factor","factor","factor","factor","factor","integer","integer",
                                 "integer","factor","factor","integer","factor","factor","integer","factor","integer","integer",
                                 "factor","integer","integer","integer","integer" ))

glimpse(empdata_orig)

empdata=empdata_orig
empdata=empdata_orig
dim(empdata)



empdata$AgeGroups=as.factor(ifelse(empdata$Age<=24,"Young",ifelse((empdata$Age>24 & empdata$Age<=54),"Middle Aged","Senior Citizen")))
table(empdata$AgeGroups)

##Total count age group wise 
empdata %>% ggplot(aes(x=empdata$AgeGroups))+geom_histogram(stat = "count")

## Age group wise count with Attrition and non attrition employees

empdata %>% ggplot(aes(x=empdata$AgeGroups))+geom_histogram(aes(color=empdata$Attrition),stat = "count",fill="white")

## Age group wise count with Attrition and non attrition employees with stacked bar graph

empdata %>% ggplot(aes(x=empdata$AgeGroups))+geom_bar(aes(fill=empdata$Attrition),position = position_dodge(),color="black")

##Business travel wise count with attrition and non attrition employees 

table(empdata$BusinessTravel)
e2=empdata %>% group_by(empdata$BusinessTravel,empdata$Attrition) %>% summarize(count=n())
e2
empdata %>% ggplot(aes(x=empdata$BusinessTravel))+geom_bar(aes(fill=empdata$Attrition),position=position_dodge(),color="black")


##Education level wise breakup 
unique(empdata$Education)
empdata %>% ggplot(aes(x=empdata$EducationField))+geom_bar(aes(fill=empdata$Attrition),position = position_dodge(),color="black")+coord_flip()

##department wise breakup of attrition of employees 
table(empdata$Department)
rm(e3)
e3=empdata %>% group_by(empdata$Department,empdata$Attrition) %>% summarise(count=n()) %>% mutate(grp_pct=count/sum(count)*100)
e3
empdata %>% ggplot(aes(x=empdata$Department))+geom_bar(aes(fill=empdata$Attrition),position = position_dodge(),color="blue")+coord_flip()

##overtime and attrition relationship ;with facet of gender hence 2 graphs
empdata %>% ggplot(aes(x=empdata$OverTime))+geom_bar(aes(fill=empdata$Attrition),position = position_dodge(),color="grey")+facet_grid(empdata$Gender~.)


###Age group wise of Yes no with stacked bar chart

e1=empdata %>% group_by(empdata$AgeGroups,empdata$Attrition) %>% summarise(count=n()) %>% mutate(Grppct=count/sum(count)*100)

e1 %>% ggplot(aes(x=e1$`empdata$AgeGroups`))+geom_bar(aes(fill=e1$`empdata$Attrition`),position = position_dodge())





