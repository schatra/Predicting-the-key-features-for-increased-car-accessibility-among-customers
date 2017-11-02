#Required Libraries
rm(list=ls(all=TRUE))
library(ISLR)
library(e1071)
library(caret)
library(MASS)
library(nnet)

#Loading the Data

Cars <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",header=F)
colnames(Cars) <- c("buying", "maint", "doors", "persons", "lug_boot", "safety", "class.values")
View(Cars)



#EDA on each attribute
#reordering data
Cars$buying <- factor(Cars$buying, levels=c("low", "med", "high", "vhigh"), ordered=TRUE)
Cars$maint <- factor(Cars$maint, levels=c("low", "med", "high", "vhigh"), ordered=TRUE)
Cars$doors <- factor(Cars$doors, levels=c("2", "3", "4", "5more"), ordered=TRUE)
Cars$persons <- factor(Cars$persons, levels=c("2", "4", "more"), ordered=TRUE)
Cars$lug_boot <- factor(Cars$lug_boot, levels=c("small", "med", "big"), ordered=TRUE)
Cars$safety <- factor(Cars$safety, levels=c("low", "med", "high"), ordered=TRUE)
Cars$class.values <- factor(Cars$class, levels=c("unacc", "acc", "good", "vgood"), ordered=TRUE)

barplot(summary(Cars$class.values))
counts10 <- table(Cars$class.values, Cars$safety)
barplot(counts10, main="Car Safety Vs Car Accebility",
        xlab="Safety",ylab="count",ylim=c(0, 500), col=c("pink","orange","blue","red"),
        legend = rownames(counts10))

counts2 <- table(Cars$class.values, Cars$persons)
barplot(counts2, main="Seating capacity Vs Car Accebility",
        xlab="persons",ylab="count",ylim=c(0, 500), col=c("pink","orange","blue","red"),
        legend = rownames(counts2))

counts3 <- table(Cars$class.values, Cars$buying)
barplot(counts3, main="Buying Price Vs Car Accebility",
        xlab="Buying Price",ylab="count",ylim=c(0, 400), col=c("pink","orange","blue","red"),
        legend = rownames(counts3))


counts4 <- table(Cars$class.values, Cars$maint)
barplot(counts4, main="Maintanace Price Vs Car Accebility",
        xlab="Maintaince Price",ylab="count",ylim=c(0, 400), col=c("pink","orange","blue","red"),
        legend = rownames(counts4))

counts5 <- table(Cars$class.values, Cars$doors)
barplot(counts5, main="No of doors Vs Car Accebility",
        xlab="No.of Doors",ylab="count",ylim=c(0, 400), col=c("pink","orange","blue","red"),
        legend = rownames(counts5))

counts6 <- table(Cars$class.values, Cars$lug_boot)
barplot(counts6, main="Luggage boot size Vs Car Accebility",
        xlab="Luggage boot space",ylab="count",ylim=c(0, 500), col=c("pink","orange","blue","red"),
        legend = rownames(counts6))


#Sampling the data into training data(80%) and test data(20%)
N<-nrow(Cars)
trainIndex = sample(1:N, size = round(0.8*N), replace=FALSE)
train = Cars[trainIndex ,]
test = Cars[-trainIndex ,]

#Best subset selection
install.packages("leaps")
library(leaps)

subset.best <- regsubsets(class.values~buying+maint+doors+persons+lug_boot+safety,
                          data=train,nbest=1)
summary(subset.best)
plot(subset.best,scale="adjr2")
plot(subset.best, scale="bic")


#Build the models
#Using multinomial logistic regression

##predicting the model for each predictor individually
model1<-multinom(class.values~safety,data=train) 
pred_svm1<-predict(model1,test) #Accuracy of the model 
mtab1<-table(pred_svm1,test$class.values) 
confusionMatrix(mtab1)
mean(pred_svm1!=test$class.values)

model2<-multinom(class.values~persons,data=train) 
pred_svm2<-predict(model2,test) #Accuracy of the model 
mtab2<-table(pred_svm2,test$class.values) 
confusionMatrix(mtab2)
mean(pred_svm2!=test$class.values)

model3<-multinom(class.values~doors,data=train) 
pred_svm3<-predict(model3,test) #Accuracy of the model 
mtab3<-table(pred_svm3,test$class.values) 
confusionMatrix(mtab3)
mean(pred_svm3!=test$class.values)

model4<-multinom(class.values~buying,data=train) 
pred_svm4<-predict(model4,test) #Accuracy of the model 
mtab4<-table(pred_svm4,test$class.values) 
confusionMatrix(mtab4)
mean(pred_svm4!=test$class.values)

model5<-multinom(class.values~maint,data=train) 
pred_svm5<-predict(model5,test) #Accuracy of the model 
mtab5<-table(pred_svm5,test$class.values) 
confusionMatrix(mtab5)
mean(pred_svm5!=test$class.values)

model6<-multinom(class.values~lug_boot,data=train) 
pred_svm6<-predict(model6,test) #Accuracy of the model 
mtab6<-table(pred_svm6,test$class.values) 
confusionMatrix(mtab6)
mean(pred_svm6!=test$class.values)



## predicting the model for multiple combination of predictors

model11<-multinom(class.values~safety+persons,data=train) 
pred_svm11<-predict(model11,test) #Accuracy of the model 
mtab11<-table(pred_svm11,test$class.values) 
confusionMatrix(mtab11)
mean(pred_svm11!=test$class.values)



model35<-multinom(class.values~doors+safety+persons,data=train) 
pred_svm35<-predict(model35,test) #Accuracy of the model 
mtab35<-table(pred_svm35,test$class.values) 
confusionMatrix(mtab35)
mean(pred_svm35!=test$class.values)

model45<-multinom(class.values~maint+safety+persons,data=train) 
pred_svm45<-predict(model45,test) #Accuracy of the model 
mtab45<-table(pred_svm45,test$class.values) 
confusionMatrix(mtab45)
mean(pred_svm45!=test$class.values)

model55<-multinom(class.values~lug_boot+safety+persons,data=train) 
pred_svm55<-predict(model55,test) #Accuracy of the model 
mtab55<-table(pred_svm55,test$class.values) 
confusionMatrix(mtab55)
mean(pred_svm55!=test$class.values)


model65<-multinom(class.values~buying+safety+persons,data=train) 
pred_svm65<-predict(model65,test) #Accuracy of the model 
mtab65<-table(pred_svm65,test$class.values) 
confusionMatrix(mtab65)
mean(pred_svm65!=test$class.values)

model75<-multinom(class.values~buying+safety+persons+maint,data=train) 
pred_svm75<-predict(model75,test) #Accuracy of the model 
mtab75<-table(pred_svm75,test$class.values) 
confusionMatrix(mtab75)
mean(pred_svm75!=test$class.values)



model85<-multinom(class.values~buying+safety+persons+lug_boot,data=train) 
pred_svm85<-predict(model85,test) #Accuracy of the model 
mtab85<-table(pred_svm85,test$class.values) 
confusionMatrix(mtab85)
mean(pred_svm85!=test$class.values)


## model with all predictors except doors

model29<-multinom(class.values~buying+maint+persons+safety+lug_boot,data=train) 
pred_svm29<-predict(model29,test) #Accuracy of the model 
mtab29<-table(pred_svm29,test$class.values) 
confusionMatrix(mtab29)
summary(model29)
mean(pred_svm29!=test$class.values)



## model with all predictors
model28<-multinom(class.values~doors+buying+maint+persons+safety+lug_boot,data=train) 
pred_svm28<-predict(model28,test) #Accuracy of the model 
length(pred_svm28)
mtab128<-table(pred_svm28,test$class.values) 
print(addmargins(mtab128))
confusionMatrix(mtab128)
mean(pred_svm28!=test$class.values)

summary(model28)


