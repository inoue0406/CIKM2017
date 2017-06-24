#
# Dimension Reduction by PCA 
# prediction by LASSO
#
library("h5")

nx <- 34
ny <- 34
nz <- 4

setwd("C:\\home\\CIKM2017")

#library(randomForest)
library(ranger)

RMSE <- function(x){
  return(sqrt(mean(x^2)))
}

D <- read.csv("processed/train/gauge_ts_train.csv")

# read hd5 data
file <- h5file("processed/for_python/radar_train_3d_ds3.hdf5")
# Save testvec in gr  oup 'test' as DataSet 'testvec'
MR <- file["MR"]
Dt <- cbind(rain=D$rain,as.data.frame(MR[,]))
h5close(file)

# set formula
vnames <- names(Dt)
vnames <- vnames[-1]
vnames1 <- paste(vnames,collapse="+")
rf.form <- as.formula(paste("rain",vnames1,sep=" ~ "))

# 0:32 start 0:53 end
oss.sell.rf <- ranger(rf.form,
                      Dt,mtry=3, # 2-84
                      num.trees=500)

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
                  Dt.tr,mtry=2, # 2-84
                  num.trees=500)
  Etrain <- sqrt(mean(cv.rf$prediction.error))
  # prediction error
  pred.y <- predict(cv.rf,data=Dt.te)
  Etest  <- RMSE(Dt.te$rain-pred.y$predictions)
  cat("fold=",i,",Training RMSE=",Etrain,", Test RMSE=",Etest,"\n")
}

# run model
oss.sell.rf <- ranger(rf.form,
                      Dt,mtry=3, # 2-84
                      num.trees=500)

# RMSE
sqrt(mean(oss.sell.rf$prediction.error))

# ---------------------------------
# (2) prediction
file <- h5file("processed/for_python/radar_testA_3d_ds3.hdf5")
# Save testvec in group 'test' as DataSet 'testvec'
MR <- file["MR"]
Dtest <- cbind(rain=D$rain,as.data.frame(MR[,]))
h5close(file)

pred.y <- predict(oss.sell.rf,data=Dtest)

# write output
dout <- data.frame(round(pred.y$predictions,digits=1))
write.table(dout,"res/ranger/rf_result_0619.csv",
            row.names=FALSE,col.names = FALSE,eol="\n")


