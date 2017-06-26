
# 
# convert radar echo data into python-readable format
# 

library("h5")

setwd("C:\\home\\CIKM2017")

for(n in c(1:10000)){
  #n <- 1
  cat("n=",n,"\n")
  fin <- sprintf("processed/train/radar_train_%05d.rdata",n)
  fout <- sprintf("processed/train_h5/radar_train_%05d.hdf5",n)
  
  MR <- readRDS(fin)
  # fill NA with zero
  if(sum(is.na(MR))>0){
    cat("NA found \n")
    MR[is.na(MR)] <- 0
  }
  
  if(file.exists(fout)){
    file.remove(fout) 
  }
  file <- h5file(fout)
  # Save testvec in group 'test' as DataSet 'testvec'
  file["MR"] <- MR
  h5close(file)
}

for(n in c(1:2000)){
  #n <- 1
  cat("n=",n,"\n")
  fin <- sprintf("processed/testA/radar_testA_%05d.rdata",n)
  fout <- sprintf("processed/testA_h5/radar_testA_%05d.hdf5",n)
  
  MR <- readRDS(fin)
  # fill NA with zero
  if(sum(is.na(MR))>0){
    cat("NA found \n")
    MR[is.na(MR)] <- 0
  }
  
  if(file.exists(fout)){
    file.remove(fout) 
  }
  file <- h5file(fout)
  # Save testvec in group 'test' as DataSet 'testvec'
  file["MR"] <- MR
  h5close(file)
}


for(n in c(1:2000)){
  #n <- 1
  cat("n=",n,"\n")
  fin <- sprintf("processed/testB/radar_testB_%05d.rdata",n)
  fout <- sprintf("processed/testB_h5/radar_testB_%05d.hdf5",n)
  
  MR <- readRDS(fin)
  # fill NA with zero
  if(sum(is.na(MR))>0){
    cat("NA found \n")
    MR[is.na(MR)] <- 0
  }
  
  if(file.exists(fout)){
    file.remove(fout) 
  }
  file <- h5file(fout)
  # Save testvec in group 'test' as DataSet 'testvec'
  file["MR"] <- MR
  h5close(file)
}

