#
# Dimension Reduction by PCA 
# prediction by LASSO
#
library("h5")
library("glmnet")

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
MR_t <- MR[,]
h5close(file)

G2 <- array(MR_t[1,],dim=c(nz,nx,ny))
image(G2[1,,])

# -----------------------------------------------
# Apply PCA
# takes around 20min
pca <- prcomp(MR_t,
              center = TRUE,
              scale. = TRUE) ## using correlation matrix

plot(pca, type = "l")
summary(pca)

nd <- 128 # reduced dimension
MPC <- pca$x[,1:nd]
# prep data
Dt <- cbind(rain=D$rain,as.data.frame(MPC))

# check the approximation
i <- 20
nn <- 50
G <- t(pca$x[i,1:nn] %*% t(pca$rotation[,1:nn])) * pca$scale + pca$center
# plot
par(mfrow=c(1,2)) 
G1 <- array(MR_t[i,],dim=c(nz,nx,ny))
image(G1[2,,],main="original")
G2 <- array(G,dim=c(nz,nx,ny))
image(G2[2,,],main=sprintf("approximation by %d PCs",nn))

# -----------------------------------------------
# fit by lasso
fitLasso1 <- glmnet( x=MPC, y=Dt$rain, family="gaussian", alpha=1 )
fitLassoCV1 <- cv.glmnet( x=MPC, y=Dt$rain, family="gaussian", alpha=1 )

plot(fitLassoCV1)

# predict by min lambda case
pred_fitLassoCV1 <- predict(fitLassoCV1, s="lambda.min", newx=MPC)

RMSE(pred_fitLassoCV1-Dt$rain)

# 00
# temp

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
                  Dt.tr,mtry=4, # 2-84
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
MRp <- file["MR"]
Dtest <- as.data.frame(MRp[,])
h5close(file)

# first apply pca
Dt.pca <- predict(pca, newdata=Dtest)

# then apply regression by RF
pred.y <- predict(oss.sell.rf,data=Dt.pca)

# write output
dout <- data.frame(round(pred.y$predictions,digits=1))
write.table(dout,"res/ranger/rf_result_0620.csv",
            row.names=FALSE,col.names = FALSE,eol="\n")


