#
# prediction by RandomForest
# used 2d image feature by autoencoder as independent variable
#
library("h5")

setwd("C:\\home\\CIKM2017")

library(randomForest)

RMSE <- function(x){
  return(sqrt(mean(x^2)))
}

D <- read.csv("processed/train/gauge_ts_train.csv")

file <- h5file("res/autoencoder_3d/auto_3d_feature128_0514_train.hdf5")
# Save testvec in group 'test' as DataSet 'testvec'
MR <- file["MR"]
Dt <- cbind(rain=D$rain,as.data.frame(MR[,]))
h5close(file)

# set formula
vnames <- names(Dt)
vnames <- vnames[2:129]
vnames1 <- paste(vnames,collapse="+")
rf.form <- as.formula(paste("rain",vnames1,sep=" ~ "))

# run model
oss.sell.rf <- randomForest(rf.form,
                            Dt,
                            ntree=500,
                            importance=T)
# åvéZéûä‘ÇTï™Ç≠ÇÁÇ¢
# 0:45 -> 1:01Ç…ÇÕÇ®ÇÌÇ¡ÇƒÇ¢ÇΩÅB

# RMSE
sqrt(mean(oss.sell.rf$mse))
# 12.83509

# ---------------------------------
# (2) prediction
file <- h5file("res/autoencoder_3d/auto_3d_feature128_0514_testA.hdf5")
# Save testvec in group 'test' as DataSet 'testvec'
MR <- file["MR"]
Dt <- as.data.frame(MR[,])
h5close(file)

pred.y <- predict(oss.sell.rf,newdata=Dt)

# write output
dout <- data.frame(round(pred.y,digits=1))
write.table(dout,"res/autoencoder_3d/rf_result.csv",
            row.names=FALSE,col.names = FALSE,eol="\n")
