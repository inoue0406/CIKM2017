#
# Dimension Reduction by PCA 
# prediction by RandomForest
# Use 2 time steps
#
rm(list = ls())
library("h5")

library("ranger")

setwd("C:\\home\\CIKM2017")

N <- 10000
nx <- 34
ny <- 34
nz <- 4
nt <- 5

RMSE <- function(x){
  return(sqrt(mean(x^2)))
}

D <- read.csv("processed/train/gauge_ts_train.csv")

# (1) Read Data

# Gauge Measurement (dependent variable)
R <- read.csv("processed/train/gauge_ts_train.csv")

# read hd5 data
file <- h5file("processed/for_python/radar_train_3d_ds3_dt3.hdf5")
# Save testvec in gr  oup 'test' as DataSet 'testvec'
fMR <- file["MR"]
MR <- fMR[,,,,] # n,nt,nx,ny,nz
h5close(file)

# select vertical layer
t.slct <- 5 

pred.PCA.RF <- function(t.slct,MR,R){
  
  tmp <- matrix(MR[,t.slct,,,],nrow=N,ncol=nx*ny*nz)
  D <- cbind(rain=R$rain,
             as.data.frame(tmp))
  
  #G1 <- array(tmp[1,],dim=c(nx,ny,nz))
  #image(G1[,,4])
  
  #par(mfrow=c(2,2)) 
  #image(MR[2,1,,,2])
  #image(MR[2,2,,,2])
  #image(MR[2,3,,,2])
  #image(MR[2,4,,,2])
  
  # split into training and validation set
  N <- 10000
  N.tr <- N*0.8
  id.tr <- c(1:N.tr)
  id.va <- c((N.tr+1):N)
  D.tr <- D[id.tr,] # 1-8000行目
  D.va <- D[id.va,] # 8001-10000行目
  
  # -----------------------------------------------
  # (2) Apply PCA
  # takes around 20min 
  pca <- prcomp(D.tr[,-1],
                center = TRUE,
                scale. = TRUE)
  
  plot(pca, type = "l")
  summary(pca)
  
  nd <- 128 # reduced dimension
  MPC <- pca$x[,1:nd]
  # prep data
  Dp.tr <- cbind(rain=D.tr$rain,
                 as.data.frame(MPC))
  
  # check the approximation
  i <- 20
  nn <- 50
  G <- t(pca$x[i,1:nn] %*% t(pca$rotation[,1:nn])) * pca$scale + pca$center
  # plot
  par(mfrow=c(1,2)) 
  G1 <- array(as.numeric(D.tr[i,-1]),dim=c(nz,nx,ny))
  image(G1[2,,],main="original")
  G2 <- array(G,dim=c(nz,nx,ny))
  image(G2[2,,],main=sprintf("approximation by %d PCs",nn))
  
  # (3) fit by RF
  vnames <- names(Dp.tr)
  vnames <- vnames[-1]
  vnames1 <- paste(vnames,collapse="+")
  rf.form <- as.formula(paste("rain",vnames1,sep=" ~ "))
  
  oss.sell.rf <- ranger(rf.form,
                        Dp.tr,mtry=3, # 2-84
                        num.trees=500)
  
  # (4) prediction
  
  # first apply pca
  Dp.va <- predict(pca, newdata=D.va[,-1])
  
  # predict by min lambda case
  pred_tr <- predict(oss.sell.rf,data=Dp.tr[,-1])
  pred_va <- predict(oss.sell.rf,data=Dp.va[,1:nd])
  
  # assess error
  r.tr <- RMSE(pred_tr$predictions - D.tr$rain)
  r.va <- RMSE(pred_va$predictions - D.va$rain)
  
  return(c(r.tr,r.va))
  
}

# 2 times
pred.PCA.2times.RF <- function(t.slct,MR,R,nd=128,mtry=3){
  
  nz.slct <- 2
  #tmp <- matrix(MR[,c(5,t.slct),,,],nrow=N,ncol=nx*ny*nz*2)# all vertical layers
  tmp <- matrix(MR[,c(5,t.slct),,,nz.slct],nrow=N,ncol=nx*ny*2)# n,nt,nx,ny,nz
  D <- cbind(rain=R$rain,
             as.data.frame(tmp))
  
  #G1 <- array(tmp[1,],dim=c(nx,ny,nz))
  #image(G1[,,4])
  
  #par(mfrow=c(2,2)) 
  #image(MR[2,1,,,2])
  #image(MR[2,2,,,2])
  #image(MR[2,3,,,2])
  #image(MR[2,4,,,2])
  
  # split into training and validation set
  N <- 10000
  N.tr <- N*0.8
  id.tr <- c(1:N.tr)
  id.va <- c((N.tr+1):N)
  D.tr <- D[id.tr,] # 1-8000行目
  D.va <- D[id.va,] # 8001-10000行目
  
  # -----------------------------------------------
  # (2) Apply PCA
  # takes around 20min 
  pca <- prcomp(D.tr[,-1],
                center = TRUE,
                scale. = TRUE)
  
  plot(pca, type = "l")
  summary(pca)
  
  #nd <- 128 # reduced dimension
  MPC <- pca$x[,1:nd]
  # prep data
  Dp.tr <- cbind(rain=D.tr$rain,
                 as.data.frame(MPC))
  
  # check the approximation
  i <- 20
  nn <- 50
  G <- t(pca$x[i,1:nn] %*% t(pca$rotation[,1:nn])) * pca$scale + pca$center
  # plot
  par(mfrow=c(1,2)) 
  G1 <- array(as.numeric(D.tr[i,-1]),dim=c(nz,nx,ny))
  image(G1[2,,],main="original")
  G2 <- array(G,dim=c(nz,nx,ny))
  image(G2[2,,],main=sprintf("approximation by %d PCs",nn))
  
  # (3) fit by RF
  vnames <- names(Dp.tr)
  vnames <- vnames[-1]
  vnames1 <- paste(vnames,collapse="+")
  rf.form <- as.formula(paste("rain",vnames1,sep=" ~ "))
  
  oss.sell.rf <- ranger(rf.form,
                        Dp.tr,mtry=mtry, # 2-84
                        num.trees=500)
  
  # (4) prediction
  
  # first apply pca
  Dp.va <- predict(pca, newdata=D.va[,-1])
  
  # predict by min lambda case
  pred_tr <- predict(oss.sell.rf,data=Dp.tr[,-1])
  pred_va <- predict(oss.sell.rf,data=Dp.va[,1:nd])
  
  # assess error
  r.tr <- RMSE(pred_tr$predictions - D.tr$rain)
  r.va <- RMSE(pred_va$predictions - D.va$rain)
  
  return(c(r.tr,r.va))
  
}

# run validation
#pred.PCA.RF(1,MR,R)
#pred.PCA.RF(2,MR,R)
#pred.PCA.RF(3,MR,R)
#pred.PCA.RF(4,MR,R)
#pred.PCA.RF(5,MR,R)

# run validation
#pred.PCA.2times.RF(1,MR,R)
#pred.PCA.2times.RF(2,MR,R)
#pred.PCA.2times.RF(3,MR,R)
#pred.PCA.2times.RF(4,MR,R)

# tweak params
pred.PCA.2times.RF(3,MR,R,nd=64 ,mtry=3)
pred.PCA.2times.RF(3,MR,R,nd=128,mtry=3)
pred.PCA.2times.RF(3,MR,R,nd=256,mtry=3)
pred.PCA.2times.RF(3,MR,R,nd=512,mtry=3)
pred.PCA.2times.RF(3,MR,R,nd=1024,mtry=3)
pred.PCA.2times.RF(3,MR,R,nd=128,mtry=2)
pred.PCA.2times.RF(3,MR,R,nd=128,mtry=3)
pred.PCA.2times.RF(3,MR,R,nd=128,mtry=5)
pred.PCA.2times.RF(3,MR,R,nd=128,mtry=10)




