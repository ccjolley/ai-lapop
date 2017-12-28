library(dplyr)
library(ggplot2)
library(rpart)
library(randomForest)
library(xgboost)
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

# Random forests have a few different hyperparameters we might want to tweak:
# `ntree`: Number of trees grown (default=500)
# `mtry`: Number of variables randomly sampled as candidates at each split. 
# (default=`sqrt(ncol(x))`=`r floor(sqrt(ncol(x)))`)
# sampsize = size of sample to draw (default = `nrow(x)`)

ntree_vals <- round(100*1.3^(0:9))
mtry_vals <- 2*1:8
sampsize_vals <- round(0.1*nrow(x)*1:10)

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

rsearch(d_scaled,2,1)
