#
# prediction by RandomForest
# used 2d image feature by autoencoder as independent variable
# use 2 time slot data
#
library("h5")

setwd("C:\\home\\CIKM2017")

#library(randomForest)
library(ranger)

RMSE <- function(x){
  return(sqrt(mean(x^2)))
}

D <- read.csv("processed/train/gauge_ts_train.csv")

layers <- c(1,2,3,4)

Dt <- cbind(rain=D$rain)

for(lno in layers){
  #lno <- 3 #layer number
  fname <- sprintf("res/autoencoder_3d_2time/auto_2d_feature256_0626_train_2time_lyr%d.hdf5",lno-1)
  file <- h5file(fname)
  # Save testvec in group 'test' as DataSet 'testvec'
  MR <- file["MR"]
  MR_t <- MR[,]
  Dt <- cbind(Dt,as.data.frame(MR_t[,]))
  h5close(file)
}

# set formula
vnames <- names(Dt)
vnames <- vnames[-1]
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
                  Dt.tr,mtry=80, # 2-84
                  num.trees=500)
  Etrain <- sqrt(mean(cv.rf$prediction.error))
  # prediction error
  pred.y <- predict(cv.rf,data=Dt.te)
  Etest  <- RMSE(Dt.te$rain-pred.y$predictions)
  cat("fold=",i,",Training RMSE=",Etrain,", Test RMSE=",Etest,"\n")
}

# run model
oss.sell.rf <- ranger(rf.form,
                      Dt,mtry=5, # 2-84
                      num.trees=500)

# RMSE
sqrt(mean(oss.sell.rf$prediction.error))

# ---------------------------------
# (2) prediction
file <- h5file("res/autoencoder_3d_alltime/auto_3d_feature128_0515_testA_alltime.hdf5")
# Save testvec in group 'test' as DataSet 'testvec'
MR <- file["MR"]
MR_t <- MR[,]
Dt <- data.frame(row.names=1:2000)

Dt <- as.data.frame(MR_t)
h5close(file)

pred.y <- predict(oss.sell.rf,data=Dt)

# write output
dout <- data.frame(round(pred.y$predictions,digits=1))
write.table(dout,"res/autoencoder_3d_alltime/rf_result_CV5_time2.csv",
            row.names=FALSE,col.names = FALSE,eol="\n")
