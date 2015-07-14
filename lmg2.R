## R version 3.1.0 (2014-04-10) -- "Spring Dance"
## Copyright (C) 2014 The R Foundation for Statistical Computing
## Platform: x86_64-w64-mingw32/x64 (64-bit)


##Loading Libraries

library(gbm)
library(e1071)
library(randomForest)
library(caret)

##set the seed

set.seed(123)

### Load train and test
setwd("C:/Users/pawlusm/Desktop/decTree")

train <- read.csv('lmg/train.csv')
test <- read.csv('lmg/test.csv')


gbm1= gbm(Hazard~.-Id, 
          data=train,
          distribution = "gaussian",
          n.trees = 50,
          interaction.depth = 9,
          n.minobsinnode = 1,
          shrinkage = 0.1,
          bag.fraction = 0.9)   # tuned to best model

pred=predict(gbm1,test[,-1],n.trees=50,type="response")
pred.t=predict(gbm1,train[,-1],n.trees=50,type="response")

mod.rf<-randomForest(Hazard~.-Id, 
                     data=train,
                     ntree = 200,
                     sampsize=3000,
                     mtry=15,
                     do.trace=50)   # tuned

pred.rf=predict(mod.rf,test)
pred.rf.t=predict(mod.rf,train)

mod.lm <- lm(Hazard~.-Id,
             data=train)

pred.lm=predict(mod.lm,test)
pred.lm.t=predict(mod.lm,train)

# mod.svm <- svm(Hazard~.-Id,
#                data=train, 
#                cost = 1000, 
#                gamma = .00001)
# 
# pred.svm=predict(mod.svm,test)

comb.pred <- ((pred*.65) + (pred.rf*.35))*.65 + (pred.lm*.35)

Submission1=data.frame("Id"=test$Id,"Hazard"=comb.pred)
write.csv(Submission1,"lmg18b.csv",row.names=FALSE,quote=FALSE)

all.preds <- data.frame("Id"=test$Id,"GBM1"=pred, "RF1" = pred.rf, "LM1" = pred.lm)
write.csv(all.preds,"lmg_preds1.csv",row.names=FALSE,quote=FALSE)

all.preds.t  <- data.frame("Id"=train$Id,"Hazard" = train$Hazard, "GBM1"=pred.t, "RF1" = pred.rf.t, "LM1" = pred.lm.t)
all.preds.ts <- data.frame("Id"=train$Id, "GBM1"=pred.t, "RF1" = pred.rf.t, "LM1" = pred.lm.t)
all.pred.t2 <- merge(train,all.preds.ts,by="Id",all.x=TRUE)
write.csv(all.preds.t ,"lmg_preds1_t.csv",row.names=FALSE,quote=FALSE)

mod.rf.e<-randomForest(Hazard~., 
                     data=all.preds.t,
                     ntree = 50,
                     do.trace=TRUE)

pred.rf.e=predict(mod.rf.e,all.preds)
Submission1=data.frame("Id"=all.preds$Id,"Hazard"=pred.rf.e)
write.csv(Submission1,"lmg19b.csv",row.names=FALSE,quote=FALSE)


#### try models on small sample


# using parallel mode to save time.
cl <- makeCluster(2)
registerDoParallel(cl)

# ctrl <- trainControl(method = "repeatedcv",
#                      number = 10,
#                      repeats = 3,
#                      allowParallel=TRUE)
# 
# gbmGrid <-  expand.grid(interaction.depth = 9,
#                         n.trees = c(10,30,50),
#                         shrinkage = .2,  # surprisingly .2 works best
#                         n.minobsinnode = 1)
# 
# fit.gbm <-train(Hazard~.-Id, 
#                 data=train,
#                 method="gbm",
#                 distribution = "gaussian",
#                 trControl = ctrl,
#                 tuneGrid = gbmGrid,
#                 verbose=FALSE
# )
# 
# # plots
# fit.gbm
# plot(fit.gbm, metric = "Kappa")
# ggplot(fit.gbm, metric = "Kappa")
# summary(fit.gbm,cBars=5)

strain <- train[sample(nrow(train), 5000), ]

## create some models
set.seed(325)
inTrain <- createDataPartition(y=strain$Hazard,
                               p=0.7, list=FALSE)
training <- strain[inTrain,]
testing <- strain[-inTrain,]

gbm1= gbm(Hazard~.-Id, 
          data=training,
          distribution = "gaussian",
          n.trees = 50,
          interaction.depth = 9,
          n.minobsinnode = 1,
          shrinkage = 0.1,
          bag.fraction = 0.9)   # tuned to best model

pred=predict(gbm1,testing[,-1],n.trees=500,type="response")

mod.rf<-randomForest(Hazard~.-Id, 
                     data=training,
                     ntree = 200,
                     sampsize=3000,
                     mtry=15,
                     do.trace=50)   # tuned

pred.rf=predict(mod.rf,testing)

# mod.lm <- lm(Hazard~.-Id,
#              data=training)
# 
# pred.lm=predict(mod.lm,testing)

# lm2 

# set the cv
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3,
                     allowParallel=TRUE)

# glm model
fit.lm <- train(Hazard~.-Id, 
                data=training,
                 method = "lm", 
                 trControl = ctrl
)

pred.lm<-predict(fit.lm,testing)



# glm 

# set the cv
# ctrl <- trainControl(method = "repeatedcv",
#                      number = 10,
#                      repeats = 3,
#                      allowParallel=TRUE)

# glm model
fit.glm <- train(Hazard~.-Id, 
                 data=training,
                 method = "glm", 
                 trControl = ctrl
)


pred.glm<-predict(fit.glm,testing)

# mod.svm <- svm(Hazard~.-Id,
#                data=training, 
#                cost = 1000, 
#                gamma = .00001)
# 
# pred.svm=predict(mod.svm,testing)


# build Gini functions for use in custom xgboost evaluation metric
SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df$random = (1:nrow(df))/nrow(df)
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}

NormalizedGini(testing$Hazard,pred)
NormalizedGini(testing$Hazard,pred.rf)
NormalizedGini(testing$Hazard,pred.lm)
# NormalizedGini(testing$Hazard,pred.svm)
NormalizedGini(testing$Hazard,pred.glm)

comb.pred <- ((pred*.65) + (pred.rf*.35))*.65 + (pred.lm*.35)
NormalizedGini(testing$Hazard,comb.pred)
