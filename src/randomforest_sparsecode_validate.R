#
# prediction by RandomForest
# used 2d image feature by sparse coding as independent variable
#
library("h5")

setwd("C:\\home\\CIKM2017")

library(ranger)

RMSE <- function(x){
  return(sqrt(mean(x^2)))
}

D <- read.csv("processed/train/gauge_ts_train.csv")

#file <- h5file("res/sparsecoding/auto_2d_feature512_0629_train_nt80.hdf5")
#file <- h5file("res/sparsecoding/auto_2d_feature64_0629_train_nt80.hdf5")
file <- h5file("res/sparsecoding/auto_2d_feature256_0629_train_nt80.hdf5")
#file <- h5file("res/sparsecoding/auto_2d_feature128_0629_train_nt80.hdf5")
#file <- h5file("res/sparsecoding/auto_2d_feature128_0629_train_nt4.hdf5")
# Save testvec in group 'test' as DataSet 'testvec'
MR <- file["MR"]
MR_t <- MR[,]
Dt <- cbind(rain=D$rain,as.data.frame(MR_t[id,]))
h5close(file)

# split into training and validation set
N <- 10000
N.tr <- N*0.8
id.tr <- c(1:N.tr)
id.va <- c((N.tr+1):N)
Dt.tr <- Dt[id.tr,] # 1-8000s–Ú
Dt.va <- Dt[id.va,] # 8001-10000s–Ú

# set formula
vnames <- names(Dt)
vnames <- vnames[-1]
vnames1 <- paste(vnames,collapse="+")
rf.form <- as.formula(paste("rain",vnames1,sep=" ~ "))

# run model
oss.sell.rf <- ranger(rf.form,
                      Dt.tr,mtry=2, # 2-84
                      num.trees=500)

# RMSE
sqrt(mean(oss.sell.rf$prediction.error))

# ---------------------------------
# (2) prediction
pred.tr <- predict(oss.sell.rf,data=Dt.tr)
pred.y  <- predict(oss.sell.rf,data=Dt.va)

# RMSE
RMSE(pred.tr$predictions-Dt.tr$rain)
RMSE(pred.y$predictions-Dt.va$rain)

