#
# prediction by RandomForest
# used 2d image feature by autoencoder as independent variable
#
setwd("C:\\home\\CIKM2017")

library(randomForest)

RMSE <- function(x){
  return(sqrt(mean(x^2)))
}


D <- read.csv("processed/train/gauge_ts_train.csv")

file <- h5file("res/autoencoder_2d/auto_2d_feature128_0513.hdf5")
# Save testvec in group 'test' as DataSet 'testvec'
MR <- file["MR"]
Dt <- cbind(D$rain,as.data.frame(MR[,]))
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
# �v�Z���ԂT�����炢

# RMSE
sqrt(mean(oss.sell.rf$mse))

# ---------------------------------
# (2) prediction
DT <- read.csv("processed/testA/gauge_ts_testA.csv")

xmat <- as.matrix(DT[,c(5:23)])
# set 0.0 if not available
xmat[is.na(xmat)] <- 0.0

D <- data.frame(xmat)
pred.y <- predict(oss.sell.rf,newdata=D)

# write output
dout <- data.frame(round(pred.y,digits=1))
#write.table(data.frame(pred.y),file("res/simple_model/simplemodel_result.csv","wb"),
#            row.names=TRUE,col.names = FALSE,eol="\n")
write.table(dout,"res/simple_model/simple_rf_result.csv",
            row.names=FALSE,col.names = FALSE,eol="\n")