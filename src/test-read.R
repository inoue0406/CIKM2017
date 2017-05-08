#
# test for reading of the data
# 

setwd("C:\\home\\CIKM")

NX <- 101
NY <- 101
NZ <- 4
NT <- 15

l <- readLines("data_new/CIKM2017_train/data_sample.txt",n=1)
#l <- readLines("data_new/CIKM2017_train/data_sample.txt",n=2)

x <- strsplit(l,",")
label <- x[[1]][1]
rain <- x[[1]][2]
M <- strsplit(x[[1]][3]," ")
M2 <- as.numeric(M[[1]])
MR <- array(M2,dim=c(NX,NY,NZ,NT))

# 
image(MR[,,4,7])
