#
# check leakage of data
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
MR_t <- MR[,]
h5close(file)

# 
R <- matrix(0,nrow = 10000,ncol=10000)
for(i in 1:10000){
  cat("i=",i)
  for(j in 1:10000){
    R[i,j] <-RMSE(MR_t[i,]-MR_t[j,])
  }   
}

fout <- sprintf("res/chk_leakage/RMSE_allpairs.rdata",n)
saveRDS(R,file=fout)

for(i in 1:10000){
  R[i,i] <- -999.0
}

id_small <- which(R<10 & R>=0)
