library(dplyr)
library(ggplot2)
library(rpart)
library(randomForest)
library(xgboost)
library(ggrepel)
set.seed(12345)

load('w_imp.rda')

x <- w_imp %>%
  select(-prov,-municipio,-w_unscaled,-w_scaled,-poor_unscaled,-poor_scaled)

d_scaled <- w_imp %>% select(poor_scaled) %>% 
  rename(y=poor_scaled) %>% cbind(x)
d_unscaled <- w_imp %>% select(poor_unscaled) %>% 
  rename(y=poor_unscaled) %>% cbind(x)

# Explain what tree-based models are. Start with simplest case of single tree.

## Single tree

tree <- rpart(y~.,data=d_scaled,method='class')
par(xpd=TRUE)
plot.new()
plot(tree,uniform=TRUE)
text(tree,use.n=TRUE,fancy=TRUE,bg='olivedrab2')

# This plot can be a little bit confusing. For some reason, the `rpart` package re-classified our
# country (`pais') labels from numbers (which were already confusing) to letters. The split at the top of the 
# tree sends all entries with a `pais` value of "o" to the right, and everything else to the left. "o" is the 15th
# letter in the alphabet, and the 15th label in our set of countries is 22, which corresponds to Haiti. So the
# first assumption our tree is making is that everyone in Haiti is not poor. This judgement was correct for 
# 61,180 cases and incorrect for 4,795, as we can read from the "6.118e+04/4795" on the left-hand split.

# That doesn't mean that all Haitians were deemed poor. Those who never use the internet (`www1 >= 4.5`) were
# considered to be poor. Out of Haitians who do not use the internet, 1,364 actually were poor, and 340 were not.
# Among at least occasional internet users, those with less than 13.5 years of education (which seems like a lot) 
# were deemed poor. 

# This tree is easy to interpret and highlight some of the same important variables we saw in logistic
# regression. How good are these classifications? We can read the correct classifications directly off the tree:

(6.118e4 + 342 + 671 + 1364) / nrow(d_scaled)

# This accuracy outperforms the naive expectation of 90%, but doesn't do as well as regression or KNN.

# How does our unscaled version look?
tree2 <- rpart(y~.,data=d_unscaled,method='class')
summary(tree2)

# In this case, the single-tree algorithm did not find any significant splits at all, and wasn't able to generate a tree.

## Random forest

# Explain what random forests are

# The downside to `randomForest` is that it's kind of slow. Let's try benchmarking
# it with subsets of the data to see how it does.

rf_xval <- function(d,niter=10,ntree=NULL,mtry=NULL,sampsize=NULL) {
  sapply(1:niter,function(i) {
    s <- runif(nrow(d)) > 0.1
    train <- d[s,] %>% select(-y)
    test <- d[!s,] %>% select(-y)
    y <- as.factor(d$y)
    train_y <- y[s]
    test_y <- y[!s]
    if (is.null(ntree)) {
      rf_i <- randomForest(x=train,y=train_y,xtest=test,ytest=test_y)
    } else {
      rf_i <- randomForest(x=train,y=train_y,xtest=test,ytest=test_y,
                           ntree=ntree,mtry=mtry,sampsize=sampsize)
    }
    mean(test_y == rf_i$test$predicted)
  })
}

bench <- function(nrow) {
  ti <- Sys.time()
  rf_xval(head(d_scaled,nrow),1)
  dt <- as.double(Sys.time()-ti,units='secs')
  paste(nrow,dt) %>% print
  dt
}

n <- c(500,1000,2000,5000,10000)
t <- sapply(n,bench)
qplot(log(n),log(t))
bm <- lm(log(t) ~ log(n))
a <- exp(summary(bm)$coef[1,1])
b <- summary(bm)$coef[2,1]

# Based on this initial benchmark, each round of cross-validation should take about
# `r round(a*nrow(x)^b/60,2)` minutes.
# 2.7 min on laptop

# Random forests have a few different hyperparameters we might want to tweak:
# `ntree`: Number of trees grown (default=500)
# `mtry`: Number of variables randomly sampled as candidates at each split. 
# (default=`sqrt(ncol(x))`=`r floor(sqrt(ncol(x)))`)
# sampsize = size of sample to draw (default = `nrow(x)`)

ntree_vals <- round(100*1.3^(0:9))
mtry_vals <- 2*1:8
sampsize_vals <- round(0.1*nrow(x)*1:8)

rsearch <- function(d,nsamp,niter) {
  hyper <- data.frame(ntree=base::sample(ntree_vals,2*nsamp,replace=TRUE),
                      mtry=base::sample(mtry_vals,2*nsamp,replace=TRUE),
                      sampsize=base::sample(sampsize_vals,2*nsamp,replace=TRUE)) %>%
    unique %>%
    head(nsamp)
  plyr::ldply(1:nrow(hyper),function(j) {
    ntree <- hyper[j,'ntree']
    mtry <- hyper[j,'mtry']
    sampsize <- hyper[j,'sampsize']
    ti <- Sys.time()
    res <- rf_xval(d,niter=niter,ntree=ntree,mtry=mtry,sampsize=sampsize) %>% mean
    dt <- as.double(Sys.time()-ti,units='secs')
    paste(ntree,mtry,sampsize,res,round(dt,2)) %>% print
    data.frame(ntree=ntree,mtry=mtry,sampsize=sampsize,
               acc=res,dt=dt)
  })
}

quick <- rsearch(d_scaled,5,1)

# Based on this quick benchmarking, each iteration should take about 
# `r round(mean(quick$dt)/60,2)` minutes. That means that if we want 60
# hyperparameter combinations with 5-fold cross validation of each one, then
# we're looking at `r round(mean(quick$dt)*60*5/3600,2)` hours of computation.
# This is another one to run overnight and cache the results.

t1 <- Sys.time()
h_scaled <- rsearch(d_scaled,60,5)
t_scaled <- Sys.time()-t1 # finished in 8.82h on AWS
save(h_scaled,file='rf_hscaled.rda')
t2 <- Sys.time()
h_unscaled <- rsearch(d_unscaled,60,5)
t_unscaled <- Sys.time()-t2
save(h_unscaled,file='rf_hunscaled.rda')
# Run these lines instead
load('rf_hscaled.rda')
load('rf_hunscaled.rda')

### Running those on AWS now.

# analysis: more than 2 HPs, so use lm

lm(acc~ntree+mtry+sampsize,data=h_scaled) %>% summary
# It looks like ntree doesn't matter very much, and sampsize
# matters a little bit. mtry is the most important.
# Does this intuition hold up with which 
# hyperparameter combinations actually came out at the top?
h_scaled %>% arrange(desc(acc)) %>% head

# Compare that result to default parameters:
# mtry=8, ntree=500, sampsize=69321
default_scaled <- rf_xval(d_scaled,5)
mean(h_scaled$acc >= mean(default_scaled))
# So about 22% of the hyperparameter combinations we testetd
# in our random search outperformed the default hyperparameters.
# It seems that the search was worth it.

# What about execution time?
lm(dt~ntree+mtry+sampsize,data=h_scaled) %>% summary
# For execution time, it's ntree and sampsize that really 
# dominate, and mtry (which affects accuracy the most)
# doesn't matter much.

# 10x xval on top solutions, save and rank

newrow_rf <- function(d,label,scaled) {
  tt_acc <- t.test(d$acc)
  data.frame(label=label,scaled=scaled,
             acc_mean=mean(d$acc),
             acc_lo=tt_acc$conf.int[1],
             acc_hi=tt_acc$conf.int[2],
             ll_mean=NA,ll_lo=NA,ll_hi=NA)
}

best_rf <- function(hparams,d,scaled) {
  set.seed(12345)
  hparams <- hparams %>% arrange(desc(acc))
  ntree_best <- hparams[1,'ntree']
  mtry_best <- hparams[1,'mtry']
  sampsize_best <- hparams[1,'sampsize']
  final <- rf_xval(d,10,ntree=ntree_best,mtry=mtry_best,sampsize=sampsize_best)
  label <- paste0('RF: (ntree,mtry,sampsize)=(',
                  ntree_best,',',mtry_best,',',sampsize_best,')')
  newrow_rf(final,label,scaled)
}

results <- read.csv('results.csv') %>%
  rbind(best_rf(h_scaled,d_scaled,TRUE)) %>%
  rbind(best_rf(h_unscaled,d_unscaled,FALSE)) 

# Train just one model with all data, get variable importance
importance_rf <- function(h,d,title,label_cutoff=0.002) {
  h <- arrange(h,desc(acc))
  y <- as.factor(d$y)
  rf <- randomForest(x=x,y=y_scaled,
                     ntree=h[1,'ntree'],
                     mtry=h[1,'mtry'],
                     sampsize=h[1,'sampsize'],
                     importance=TRUE)
  imp <- rf$importance %>% as.data.frame
  names(imp) <- c('false','true','accuracy_decrease','gini_decrease')
  imp <- imp %>%
    mutate(label=row.names(imp),
           label=ifelse(accuracy_decrease > label_cutoff,label,NA))
  ggplot(imp,aes(x=false,y=true,label=label)) +
    geom_point(size=3,color='darkorange3') +
    geom_text(vjust=1) +
    #geom_text_repel() +
    xlab('Accuracy loss for FALSE class') +
    ylab('Accuracy loss for TRUE class') +
    ggtitle(title)
}
importance_rf(h_scaled,d_scaled,'Scaled wealth index')
importance_rf(h_unscaled,d_unscaled,'Unscaled wealth index')

# Explain how variable importance calculations are done
# For the most part, the variables that look important here are similar to the ones that were highly significant
# in the logistic regression. We're seeing geographic factors (`pais`,`ur`,`tamano`), demographics (`leng1`,`etid`,`q2`),
# household economics (`q10d`) and variables related to education or information consumption (`ed`,`www1`,`gi0`).

# Note that these effects are asymmetric (note the difference in scale of the x and y axes). 
# Most variables have a stronger impact on the accuracy of TRUE predictions, simply because the TRUE class is smaller. 

##############################

# Next, on to xgboost!

logloss <- function(p,y,epsilon=0.01) {
  p[p==0] <- epsilon
  p[p==1] <- 1-epsilon
  -mean(y*log2(p)) - mean((1-y)*log2(1-p))
}

# Comment on early stopping -- effectively gives us automatic tuning of one hyperparameter

xgb_xval <- function(d,niter=10,params=list(),verbose=0) {
  plyr::ldply(1:niter,function(i) {
    mm <- model.matrix(~ .+0, data=select(d,-y)) 
    s <- runif(nrow(d)) > 0.1
    train <- xgb.DMatrix(mm[s,],label = d$y[s])
    test <- xgb.DMatrix(mm[!s,],label = d$y[!s])
    watchlist=list(train=train,test=test)
    # parameters: eta, gamma, max_depth, min_child_weight, subsample, colsample_bytree
    xgb_i <- xgb.train(params=params,
                       data = train,
                       watchlist=watchlist,
                       objective = "binary:logistic",
                       nthread=2,
                       nrounds=200,
                       early_stopping_rounds=10,
                       verbose=verbose)
    data.frame(acc=as.numeric(1 - xgb_i$best_score),
               ll=logloss(predict(xgb_i,newdata=test),d$y[!s]))
  })
}

# hyperparameter tuning
eta_vals <- 0.025*1:16
gamma_vals <- 0.1*1.7^(0:14)
max_depth_vals <- 2:10
min_child_weight_vals <- 0.1*1.7^(0:14)
subsample_vals <- 0.1*1:10
colsample_bytree_vals <- 0.1*1:10

rsearch_xgb <- function(d,nsamp,niter) {
  hyper <- data.frame(eta=base::sample(eta_vals,2*nsamp,replace=TRUE),
                      gamma=base::sample(gamma_vals,2*nsamp,replace=TRUE),
                      max_depth=base::sample(max_depth_vals,2*nsamp,replace=TRUE),
                      min_child_weight=base::sample(min_child_weight_vals,2*nsamp,replace=TRUE),
                      subsample=base::sample(subsample_vals,2*nsamp,replace=TRUE),
                      colsample_bytree=base::sample(colsample_bytree_vals,2*nsamp,replace=TRUE)) %>%
    unique %>%
    head(nsamp)
  plyr::ldply(1:nrow(hyper),function(j) {
    params <- list(eta=hyper[j,'eta'],
                   gamma=hyper[j,'gamma'],
                   max_depth=hyper[j,'max_depth'],
                   min_child_weight=hyper[j,'min_child_weight'],
                   subsample=hyper[j,'subsample'],
                   colsample_bytree=hyper[j,'colsample_bytree'])
    ti <- Sys.time()
    res <- xgb_xval(d,niter,params) %>% summarize(acc=mean(acc),ll=mean(ll))
    dt <- as.double(Sys.time()-ti,units='secs')
    paste(j,res,Sys.time()) %>% print
    cbind(data.frame(params,dt=dt),res)
  })
}

quick <- rsearch_xgb(d_scaled,5,1)

# On average, each model fitting took `r round(mean(quick$dt),2)` seconds. This means that if we want
# 60 hyperparameter combinations with 5-fold cross-validation we'll need about `r 60*5*mean(quick$dt)/60`
# minutes. 

t1 <- Sys.time()
h_scaled_xgb <- rsearch_xgb(d_scaled,60,5)
t_scaled_xgb <- Sys.time()-t1 
save(h_scaled_xgb,file='xgb_hscaled.rda')
t2 <- Sys.time()
h_unscaled_xgb <- rsearch_xgb(d_unscaled,60,5)
t_unscaled_xgb <- Sys.time()-t2
save(h_unscaled_xgb,file='xgb_hunscaled.rda')

# The hyperparameter search with the scaled wealth index took `r round(as.double(t_scaled_xgb,units='mins'),2)`
# minutes, while the search with the unscaled index took `r round(as.double(t_unscaled_xgb,units='mins'),2)` minutes.

h_scaled_xgb %>% arrange(desc(acc)) %>% head

# It's striking how top-ranked solutions seem to have very different combinations of hyperparameters.
# One possible reason for this could be that many parameters have similar (or opposing) effects, so that
# the hyperparameter landscape has multiple nearly-optimal points. What about the unscaled wealth index?

h_unscaled_xgb %>% arrange(desc(acc)) %>% head

# Again, the performance with the unscaled wealth index lags behind.

# 10x xval on top solutions, save and rank

newrow_xgb <- function(d,label,scaled) {
  tt_acc <- t.test(d$acc)
  tt_ll <- t.test(d$ll)
  data.frame(label=label,scaled=scaled,
             acc_mean=mean(d$acc),
             acc_lo=tt_acc$conf.int[1],
             acc_hi=tt_acc$conf.int[2],
             ll_mean=mean(d$ll),
             ll_lo=tt_ll$conf.int[1],
             ll_hi=tt_ll$conf.int[2])
}

best_xgb <- function(hparams,d,scaled) {
  set.seed(12345)
  hparams <- hparams %>% arrange(desc(acc))
  params <- list(eta=hparams[1,'eta'],
                 gamma=hparams[1,'gamma'],
                 max_depth=hparams[1,'max_depth'],
                 min_child_weight=hparams[1,'min_child_weight'],
                 subsample=hparams[1,'subsample'],
                 colsample_bytree=hparams[1,'colsample_bytree'])
  final <- xgb_xval(d,10,params)
  label <- 'xgboost'
  newrow_xgb(final,label,scaled)
}

# TODO: when I run this for real, it will be picking up where RF left off, not 
# loading from disk
results <- read.csv('results.csv') %>%
  rbind(best_xgb(h_scaled_xgb,d_scaled,TRUE)) %>%
  rbind(best_xgb(h_unscaled_xgb,d_unscaled,FALSE)) 

# Does xgBoost give me variable improtance?
