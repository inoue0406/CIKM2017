#
# prediction by RandomForest
# used 2d image feature by autoencoder as independent variable
# use alltime data
#
library("h5")

setwd("C:\\home\\CIKM2017")

#library(randomForest)
library(ranger)

RMSE <- function(x){
  return(sqrt(mean(x^2)))
}

D <- read.csv("processed/train/gauge_ts_train.csv")

file <- h5file("res/autoencoder_3d_alltime/auto_3d_feature128_0515_train_alltime.hdf5")
# Save testvec in group 'test' as DataSet 'testvec'
MR <- file["MR"]

# only use 15th time
id <- rep(c(rep(FALSE,14),TRUE),10000)
MR_t <- MR[,]
MR_t <- MR_t[id,]

Dt <- cbind(rain=D$rain,as.data.frame(MR_t[,]))
h5close(file)

# set formula
vnames <- names(Dt)
vnames <- vnames[2:129]
vnames1 <- paste(vnames,collapse="+")
rf.form <- as.formula(paste("rain",vnames1,sep=" ~ "))

# 5-fold cross-validation
nf <- 5
N  <- 10000
id <- c(1:N)
#i  <- 1
for(i in 1:nf){
  id5 <- id[id %% nf==(i-1)]
  Dt.tr <- Dt[-id5,]
  Dt.te <- Dt[id5,]
  cv.rf <- ranger(rf.form,
                  Dt.tr,mtry=12, # 2-84
                  num.trees=500)
  Etrain <- sqrt(mean(cv.rf$prediction.error))
  # prediction error
  pred.y <- predict(cv.rf,data=Dt.te)
  Etest  <- RMSE(Dt.te$rain-pred.y$predictions)
  cat("fold=",i,",Training RMSE=",Etrain,", Test RMSE=",Etest,"\n")
}

# run model
oss.sell.rf <- ranger(rf.form,
                      Dt,mtry=12, # 2-84
                      num.trees=500)

# RMSE
sqrt(mean(oss.sell.rf$prediction.error))

# ---------------------------------
# (2) prediction
file <- h5file("res/autoencoder_3d_alltime/auto_3d_feature128_0515_testA_alltime.hdf5")
# Save testvec in group 'test' as DataSet 'testvec'
MR <- file["MR"]

# only use 15th time
id <- rep(c(rep(FALSE,14),TRUE),2000)
MR_t <- MR[,]
MR_t <- MR_t[id,]

Dt <- as.data.frame(MR_t)
h5close(file)

pred.y <- predict(oss.sell.rf,data=Dt)

# write output
dout <- data.frame(round(pred.y$predictions,digits=1))
write.table(dout,"res/autoencoder_3d_alltime/rf_result_CV5.csv",
            row.names=FALSE,col.names = FALSE,eol="\n")
