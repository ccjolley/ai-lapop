library(dplyr)
library(ggplot2)
library(FNN)
set.seed(12345)

load('w_imp.rda')

# Explain what KNN is all about

x <- w_imp %>%
  select(-prov,-municipio,-w_unscaled,-w_scaled,-poor_unscaled,-poor_scaled)

d_scaled <- w_imp %>% select(poor_scaled) %>% 
  rename(y=poor_scaled) %>% cbind(x)
d_unscaled <- w_imp %>% select(poor_unscaled) %>% 
  rename(y=poor_unscaled) %>% cbind(x)

# Helper functions from last time; explain why I added epsilon

logloss <- function(p,y,epsilon=0.01) {
  p[p==0] <- epsilon
  p[p==1] <- 1-epsilon
  -mean(y*log2(p)) - mean((1-y)*log2(1-p))
}

# Explain why we're putting PCA inside the CV loop

# Go straight to CV, as before
knn_cv <- function(d,k,i,niter=10) {
  plyr::ldply(1:niter,function(j) {
    s <- runif(nrow(d)) > 0.1
    # prepare training data
    train <- d[s,] %>% select(-y)
    train_pca <- model.matrix(~ ., data=train) %>%
      as.data.frame() %>%
      select(-`(Intercept)`) %>%
      prcomp(center=TRUE,scale=TRUE)
    train_cl <- d$y[s]
    # prepare test data
    test <- d[!s,]  %>% select(-y)
    test_pca <- model.matrix(~ ., data=test) %>%
      as.data.frame() %>%
      select(-`(Intercept)`) %>%
      predict(train_pca,newdata=.)
    test_cl <- d$y[!s]
    # train model
    knn_i <- knn.reg(train=train_pca$x[,1:i],
                     test=test_pca[,1:i],
                     y=train_cl,k=k)
    acc <- sum(test_cl == (knn_i$pred >= 0.5)) / nrow(test)
    ll <- logloss(knn_i$pred,test_cl)
    data.frame(acc=acc,ll=ll)
  })
}

# Each knn evaluation takes ~63s, which means that a single 10x x-val will take ~10min

# First hyperparameter: Tune k for a given number of PCs (say, 12)
k <- 12
nsamples <- 50
tune_k <- plyr::ldply(1:nsamples,function(k) {
  paste(k,Sys.time()) %>% print
  knn_cv(d_scaled,k,i=12,niter=1) %>%
    summarize(acc=mean(acc),ll=mean(ll))
})
save(tune_k,'knn_tune_k.rda')
tune_k$k <- 1:niter
ggplot(tune_k,aes(x=k,y=acc)) +
  geom_point(size=2,color='indianred4') +
  geom_line(color='indianred4') +
  xlab('k') + ylab('Accuracy') 

# With 12 PCs, we get fairly poor performance at low k and then stop 
# seeing performance gains after about k=20. This curve would probably
# be a little smoother if we'd used more iterations in cross-validation.

# We can also vary the number of PCs for a given value of k (say, 20)
max_i <- min(ncol(x),70)
tune_i <- plyr::ldply(2:max_i,function(i) {
  paste(i,Sys.time()) %>% print
  knn_cv(d_scaled,k=20,i=i,niter=1) %>%
    summarize(acc=mean(acc),ll=mean(ll))
})
save(tune_i,'knn_tune_i.rda')
tune_i$i <- 2:max_i
ggplot(tune_i,aes(x=i,y=acc)) +
  geom_point(size=2,color='navyblue') +
  geom_line(color='navyblue') +
  xlab('i') + ylab('Accuracy') 

# This shows an increase with the number of principal components; it's possible
# that things will taper off again for larger numbers.

# Potentially large hyperparameter space at play here; use
# random search with 60 points (likely to be within the top 5%, 95% of the time).
# To avoid having to run this for many hours, just doing one evaluation for each
# hyperparameter combination. This will be a rough cut to help us drill in on the
# part of the hyperparameter space we're really interested in.

rsearch <- function(d,nsamp,niter) {
  hyper <- data.frame(k=sample.int(50,size=2*nsamp,replace=TRUE),
                      i=sample.int(ncol(x)-1,size=2*nsamp,replace=TRUE)+1) %>%
    unique %>%
    head(nsamp)
  plyr::ldply(1:nrow(hyper),function(j) {
    i <- hyper[j,'i']
    k <- hyper[j,'k']
    ti <- Sys.time()
    res <- knn_cv(d,k=k,i=i,niter=niter) %>%
      summarize(k=k,i=i,acc=mean(acc),ll=mean(ll))
    dt <- as.double(Sys.time()-ti,units='secs')
    paste(i,k,res$acc,round(dt,2)) %>% print
    cbind(res,data.frame(dt=dt))
  })
}

# Quick run to estimate how much time this will take
quick <- rsearch(d_scaled,5,1)

# A typical best practice for random hyperparameter searches is to try 60 
# parameter combinations. Of our 5 quick test runs, the average computation
# time was `r round(mean(quick$dt),2)` seconds. This means that if we want
# to sample 60 points with 5-fold cross-validation we're looking at roughly
# `r round(mean(quick$dt)*60*5/3600,2)` hours of calculation. We'll want to run 
# this overnight and save the result.

t1 <- Sys.time()
h_scaled <- rsearch(d_scaled,60,5)
Sys.time()-t1
save(h_scaled,file='knn_hscaled.rda')
t2 <- Sys.time()
h_unscaled <- rsearch(d_unscaled,60,5)
Sys.time()-t2
save(h_unscaled,file='knn_hunscaled.rda')
# Run these lines instead
load('knn_hscaled.rda')
load('knn_hunscaled.rda')

# Plot results for scaled
h_scaled %>% 
  mutate(acc=ifelse(acc >= 0.9,acc,NA)) %>%
  ggplot(aes(x=i,y=k)) +
    geom_point(size=5,aes(color=acc)) +
    scale_color_gradientn(colors=rainbow(10)) +
    xlab('Number of PCs (i)') + ylab('Number of neighbors (k)')

# What I've done here is to filter out any cases where the KNN model performed worse than the 
# worst-case naive model (accuracy < 90%). That only appears to be a problem for small values of
# _k_. It looks like our best-performing examples are at fairly moderate values of both k and i.
# I'm normally not a fan of rainbow color scales like this, but in this case it serves to highlight 
# that values close to the maximum are spread out over a large part of our hyperparameter space. This
# is a good thing, because it means that our precise hyperparameter values won't matter very much,
# as long as we pick something in the purple-pink region.

h_scaled %>% arrange(desc(acc)) %>% head

# Now let's do the same for the unscaled wealth variable

h_unscaled %>% 
  mutate(acc=ifelse(acc >= 0.9,acc,NA)) %>%
  ggplot(aes(x=i,y=k)) +
  geom_point(size=5,aes(color=acc)) +
  scale_color_gradientn(colors=rainbow(10)) +
  xlab('Number of PCs (i)') + ylab('Number of neighbors (k)')

# This is kind of a mess, and even the ones that beat 90% accuracy only barely make it. There's 
# no visually-obvious pattern to the colors, which means that the KNN model built on the unscaled
# wealth variable probably doesn't have a well-defined maximum region the way that the scaled
# wealth variable does.

h_unscaled %>% arrange(desc(acc)) %>% head

# Take the best hyperparameters and add those to our results frame
newrow <- function(d,label,scaled) {
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

best_knn <- function(hparams,d,scaled) {
  set.seed(12345)
  hparams <- hparams %>% arrange(desc(acc))
  k_best <- hparams[1,'k']
  i_best <- hparams[1,'i']
  final <- knn_cv(d,k_best,i_best,10)
  label <- paste0('KNN: (k,i)=(',k_best,',',i_best,')')
  newrow(final,label,scaled)
}

results <- read.csv('results.csv') %>%
  #rbind(best_knn(h_scaled,d_scaled,TRUE)) %>%
  rbind(best_knn(h_unscaled,d_unscaled,FALSE)) 

# Best scaled results
results %>% filter(scaled==TRUE) %>% arrange(desc(acc_mean))

# Best unscaled results
results %>% filter(scaled==FALSE) %>% arrange(desc(acc_mean))

write.csv(results,'results.csv',row.names=FALSE)
