#
# test prediction by simple model
#
setwd("C:\\home\\CIKM")

RMSE <- function(x){
  return(sqrt(mean(x^2)))
}


D <- read.csv("processed/train/gauge_ts_train.csv")

plot(D$rain)

plot(D$rain,D$rz1)
plot(D$rain,D$rz2)
plot(D$rain,D$rz3)
plot(D$rain,D$rz4)
plot(D$rain,D$rts1)
plot(D$rain,D$rts15)

cor(D$rain,D$rz1)

model <- lm(D$rain~D$rz1+D$rz2+D$rz3+D$rz4)
RMSE(model$residuals)

model2<- lm(D$rain~D$rts1+D$rts2+D$rts3+D$rts4+D$rts5+D$rts6+D$rts7+D$rts8+
              D$rts9+D$rts10+D$rts11+D$rts12+D$rts13+D$rts14+D$rts15)
RMSE(model2$residuals)

DT <- read.csv("processed/testA/gauge_ts_testA.csv")

xmat <- as.matrix(DT[,c(5:19)])
# set 0.0 if not available
xmat[is.na(xmat)] <- 0.0

D <- data.frame(xmat)
pred.y <- predict(model2,newdata=D)

sum(c(1,xmat[1,]) * model2$coefficients)

str(c(1,xmat[1,]))
str(model2$coefficients)

# write output
dout <- data.frame(round(pred.y,digits=1))
#write.table(data.frame(pred.y),file("res/simple_model/simplemodel_result.csv","wb"),
#            row.names=TRUE,col.names = FALSE,eol="\n")
write.table(dout,"res/simple_model/simplemodel_result.csv",
            row.names=FALSE,col.names = FALSE,eol="\n")
