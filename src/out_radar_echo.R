
# print radar echo data

setwd("C:\\home\\CIKM")

library("fields")

# select
D <- read.csv("processed/train/gauge_ts_train.csv")

# 
id_slct <- c()
for(ra in c(0,10,20,30,40,50,60)){
  for(rz in c(0,10,20,30,40,50,60)){
    cat("rain:radar=",ra,rz,"\n")    
    id <- D$rain > ra & D$rain < (ra+10) & D$rz1 > rz & D$rz1 < (rz+10)
    n <- which(id)[1]
    id_slct <- c(id_slct,n)
    print(D[n,])
  }
}

# read radar data
for(n in id_slct){
  #n <- id_slct[30]
  cat("n=",n)
  
  fin <- sprintf("processed/train/radar_train_%05d.rdata",n)
  
  MR <- readRDS(fin)
  
  it <- 1
  for(it in c(1:15)){
    png(filename=sprintf("res/radar_png/radar_r%03d_%05d_t%02d.png",round(D$rain[n]),n,it)
        , width=1000, height=800)
    par(mfrow=c(2,2),mar=c(0.2,0.2,2,0.2))
    for(iy in c(1:4)){
      image.plot(MR[,,iy,it],zlim=c(0,200),col=rev(topo.colors(20)),
                 xaxt="n", yaxt="n",main=paste("time:",it,", ",(it-15)*6,"min, layer:",iy,"rain:",D$rain[n])) 
      points(0.5,0.5,pch=8,cex=2)
      grid()
    }
    dev.off()
  }
}


