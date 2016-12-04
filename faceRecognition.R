install.packages('doMC', repos='https://cran.ma.imperial.ac.uk/')
library(doMC)
registerDoMC()

install.packages('reshape2',  repos='https://cran.ma.imperial.ac.uk/')
library(reshape2)
#declaring var
#looking at choosen image
#imageN <- 5



###reading train
train.file <- paste0(data.dir, 'training.csv')
d.train    <- read.csv(train.file, stringsAsFactors=F)
#put image pixels in  another variable
im.train   <- foreach(im = d.train$Image, .combine=rbind) %dopar% {
    as.integer(unlist(strsplit(im, " ")))
}
#delete column with image pixels
d.train$Image <- NULL
###

###reading test
test.file  <- paste0('test.csv')
d.test  <- read.csv(test.file, stringsAsFactors=F)
#put image pixels in  another variable
im.test <- foreach(im = d.test$Image, .combine=rbind) %dopar% {
    as.integer(unlist(strsplit(im, " ")))
}
#delete column with image pixels
d.test$Image <- NULL
###

#save(d.train, im.train, d.test, im.test, file='data.Rd')

### image representation
#im <- matrix(data=rev(im.train[imageN,]), nrow=96, ncol=96)
#image(1:96, 1:96, im, col=gray((0:255)/255))

#points(96-d.train$nose_tip_x[imageN],         96-d.train$nose_tip_y[imageN],         col="red")
#points(96-d.train$left_eye_center_x[imageN],  96-d.train$left_eye_center_y[imageN],  col="blue")
#points(96-d.train$right_eye_center_x[imageN], 96-d.train$right_eye_center_y[imageN], col="green")


#predictions based on means
colMeans(d.train, na.rm=T)

p           <- matrix(data=colMeans(d.train, na.rm=T), nrow=nrow(d.test), ncol=ncol(d.train), byrow=T)
colnames(p) <- names(d.train)
predictions <- data.frame(ImageId = 1:nrow(d.test), p)
head(predictions)

submission <- melt(predictions, id.vars="ImageId", variable.name="FeatureName", value.name="Location")

example.submission <- read.csv(paste0('SampleSubmission.csv'))
sub.col.names      <- names(example.submission)
example.submission$Location <- NULL
submission <- merge(example.submission, submission, all.x=T, sort=F)
submission <- submission[, sub.col.names]
write.csv(submission, file="submission_means.csv", quote=F, row.names=F)
