#
# read the raw data 
# 

setwd("C:\\home\\CIKM")

NX <- 101
NY <- 101
NZ <- 4
NT <- 15

NMAX <- 10000

RES <- data.frame(matrix(0,nrow = NMAX,ncol=3+15+4))
colnames(RES) <- c("id","label","rain",
                   paste("rts",c(1:15),sep=""),
                   paste("rz",c(1:4),sep=""))

FI <- file("data_new/CIKM2017_train/train.txt", "r")

n <- 1
for(n in 1:NMAX){
  cat("n=",n,"\n")
  RES$id[n] <- n
  #l <- readLines("data_new/CIKM2017_train/data_sample.txt",n=1)
  l <- readLines(FI,n=1)
  if ( length(l) == 0 ) {
    break
  }
  
  x <- strsplit(l,",")
  RES$label[n] <- x[[1]][1]
  # rainnfall gauge
  RES$rain[n] <- as.numeric(x[[1]][2])
  # radar reflectivity
  M <- strsplit(x[[1]][3]," ")
  M2 <- as.numeric(M[[1]])
  # SET NA if negative
  M2[M2 < -0.0001] <- NA
  MR <- array(M2,dim=c(NX,NY,NZ,NT))
  # 
  rts <- apply(MR,c(4),mean)
  rz <- apply(MR,c(3),mean)
  RES[n,c(4:22)] <- c(rts,rz)
  
  # save as RDS file
  fout <- sprintf("processed/train/radar_train_%05d.rdata",n)
  saveRDS(MR,file=fout)

}

close(FI)

# write time series data
write.csv(RES,"processed/train/gauge_ts.csv")
