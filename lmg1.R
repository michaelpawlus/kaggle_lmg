# You can write R code here and then click "Run" to run it on our platform

# The readr library is the best way to read and write CSV files in R
library(randomForest)
library(gbm)
library(caret)
library(parallel)
library(doParallel)

### Load train and test
setwd("C:/Users/pawlusm/Desktop/decTree")

train <- read.csv('lmg/train.csv')
test <- read.csv('lmg/test.csv')

# remove IDs from the train set
train<-train[,-1]


# using parallel mode to save time.
cl <- makeCluster(2)
registerDoParallel(cl)


# rf
mod.rf<-randomForest(train[2:33],log(train[,1]+1),ntree = 20,sampsize=20000,do.trace=TRUE)

pred.rf<-exp(predict(mod.rf,test))-1

range(train$Hazard)
table(train$Hazard)
range(pred.rf)
table(round(pred.rf))

write.csv(data.frame(Id=test$Id,Hazard=pred.rf),"lmg4b.csv",row.names=FALSE)


# gbm
mod.gbm <-gbm(log(train[,1]+1)~., 
          data=train[,2:33], 
          distribution="gaussian",
          cv.folds = 3,             # current best did not have
          n.trees=500,             # current best with 500
          shrinkage=0.05,           # current best with 0.05
          interaction.depth=5,       # current best with 20
#           bag.fraction = 0.5,
#           train.fraction = 0.5,
#           n.minobsinnode = 25,
          n.cores = 2
          )

best.gbm <- gbm.perf(mod.gbm, method="cv")
best.gbm
plot.gbm(mod.gbm)
pretty.gbm.tree(mod.gbm)
plot(pretty.gbm.tree(mod.gbm))
relative.influence(mod.gbm)
summary(mod.gbm, n.trees = 500)
print.gbm(mod.gbm)
summary.gbm(mod.gbm)

qplot(Hazard, T1_V11,data=train)
qplot(Hazard, T1_V16,data=train)
qplot(Hazard, T1_V8,data=train)
qplot(Hazard, T1_V1,data=train)
qplot(Hazard, T1_V2,data=train)
qplot(Hazard, T1_V15,data=train)
qplot(Hazard, T1_V12,data=train)



pred.gbm<-exp(predict(mod.gbm,test,n.trees=451))-1

range(train$Hazard)
table(round(train$Hazard))
range(pred.gbm)
table(round(pred.gbm))

write.csv(data.frame(Id=test$Id,Hazard=pred.gbm),"lmg13b.csv",row.names=FALSE)

pred.comb <- ((pred.gbm*0.9)+(pred.rf*0.1))

write.csv(data.frame(Id=test$Id,Hazard=pred.comb),"lmg6b.csv",row.names=FALSE) # current best score = 0.375419



# glm 

# set the cv
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3,
                     allowParallel=TRUE)

# glm model
fit.glm <- train(log(train[,1]+1)~., 
                data=train[,2:33], 
                method = "glm", 
                trControl = ctrl, 
                preProc = c("center","scale")
)

fit.glm

pred.glm<-exp(predict(fit.glm,test))-1
pred[pred.gbm<0]<-0

range(train$Hazard)
range(pred.glm)
table(round(pred.glm))

## nnet

## Train the network and tuning the number of nodes and decay
maxout= max(train[,1]) # to scale the output
mygrid <- expand.grid(.decay=c(0.5, 0.1), .size=c(3,4,5))
nnetfit <- train(Hazard/maxout ~ ., 
                 data=train, 
                 method="nnet", 
                 maxit=50, 
                 tuneGrid=mygrid, 
                 trace=T
                 )
nnetfit

pred.nn<-predict(nnetfit,test)
pred[pred.gbm<0]<-0
pred.nn <- pred.nn*69

range(train$Hazard)
range(pred.nn)
table(round(pred.nn))


# write csv

pred.comb <- ((pred.gbm*0.5)+(pred.glm*0.4)+(pred.rf*0.1))
table(round(pred.comb))

write.csv(data.frame(Id=test$Id,Hazard=pred.comb),"lmg11b.csv",row.names=FALSE) # current best score = 0.377456 (gbm 80 - rf 10 - glm 10)

ctest <- data.frame(Id=test$Id,Hazard=pred.gbm)

ftrain <- ftrain[order(ftrain$Hazard, decreasing = TRUE),]
ctest <- ctest[order(ctest$Hazard, decreasing = TRUE),]

ftest <- cbind(ctest,ftrain$Hazard)
ftest <- ftest[,c(1,3)]
names(ftest) <- c("Id","Hazard")
write.csv(ftest,"lmg14b.csv",row.names=FALSE)
