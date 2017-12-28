library(dplyr)
library(ggplot2)
library(glmnet)

load('w_imp.rda')

## Simple logistic regression

# separate out all of our predictor variables
x <- w_imp %>%
  select(-prov,-municipio,-w_unscaled,-w_scaled,-poor_unscaled,-poor_scaled)

d_scaled <- w_imp %>% select(poor_scaled) %>% 
  rename(y=poor_scaled) %>% cbind(x)
d_unscaled <- w_imp %>% select(poor_unscaled) %>% 
  rename(y=poor_unscaled) %>% cbind(x)
reg1 <- glm(y ~ .,family=binomial(link='logit'),data=d_scaled)

regplot <- function(d,title) {
  reg1 <- glm(y ~ .,family=binomial(link='logit'),data=d)
  regplot1 <- data.frame(est=summary(reg1)$coefficients[2:119,1],
                         log_p=log10(summary(reg1)$coefficients[2:119,4]),
                         name=row.names(summary(reg1)$coefficients)[2:119]) %>%
    mutate(sig=(log_p < -2),
           label=ifelse(log_p < -25,as.character(name),NA)) 
  
  ggplot(regplot1,aes(x=log_p,y=est,color=sig)) +
    geom_point(size=2) +
    xlab('log(p-value)') + ylab('Model weight') + 
    geom_text(aes(label=label),color='black',vjust=1) +
    ggtitle(title)
}
regplot(d_scaled,'Scaled poverty measure')

# pais15 = Haiti
# ed = education level
# gi0 = frequency of paying attention to news (higher=less attention)
# www1 = internet usage
# pais5 = Nicaragua
# q10d4 = family economic situation = "not enough and having a hard time"
# etid3 = black
# pais17 = Guyana
# q2 = age
# pais4 = Honduras

# Explain about distinction between parametr-space and data-space evaluation

# Unscaled metric
regplot(d_unscaled,'Unscaled poverty measure')

#  One way 
# to do this is to choose a cutoff such that the average fraction of true responses (10%) is the same after
# prediction. We could probably get a higher accuracy by tuning the cutoff slightly

pred1 <- predict(reg1,type='response')
cutoff <- pred1 %>% quantile(1-mean(w_imp$poor_scaled))

# So we'll predict that a given household is poor if its probability is above `r cutoff`.

## Error metrics

# We'll use two different methods to evaluate how well this model works. The first is _accuracy_,
# in which we divide the number of correct predictions by the number of total predictions. This
# definition of accuracy assumes that we care the same amount about false positives (saying a household is
# poor when they are not) and false negatives (failing to identify a poor household). This isn't always true,
# and we'll need to use other metrics if it isn't.

# The predictions returned by logistic regression are not actually predictions of whether the binary variable `poor_scaled`
# takes a true or a false value. They are actually the probability of a true prediction. This means that we'll need
# to choose a cutoff probablility above which we'll be willing to claim a positive prediction. Because 
# we're not weighting false positives or negatives as more important, we'll choose a cutoff of 0.5.

sum(d_scaled$poor_scaled == (pred1 >= 0.5)) / nrow(d_scaled)

# To put this result in context, we 
# need to keep in mind that we defined our binary poverty variable such that about
# 10% of households would qualify. This means that if we just predicted that _no one_ was
# poor, we'd have an accuracy of 90%. Our model is a modest improvement on this.

# The other error metric we'll be interested in is log-loss. Rather that using a 
# cutoff to convert our probabilistic predictions into binary ones, log-loss
# works directly with the probabilities. Log-loss has roots in information theory,
# and estimates the amount of information that is lost by using our predictions instead
# of the true value.

logloss <- function(p,y) {
  -mean(y*log2(p)) - mean((1-y)*log2(1-p))
}
logloss(pred1,d_scaled$poor_scaled)

# This number is a little harder to interpret without having others we can compare
# it with. We could try what we did last time and see what happens if we predict zero poverty, but we end
# up with an infinite result if we try to take the logarithm of a zero probability. Instead,
# we'll just use the minimum value of `pred1` to denote an almost-zero probability.

logloss(min(pred1),d_scaled$poor_scaled)

# This metric shows a larger difference between our model and a naive uniform prediction.

## Cross-validation

# Our model evaluation thus far has a problem. The true performance of a machine learning model can only 
# be judged using out-of-sample data. In other words, we need to test the model using different data than
# were used to train it. With a dataset of fixed size, we can do this by training the model on 90% of the 
# data and testing it on the remaining 10%. If we repeat this train-test split with different random
# samples of the data, then we can get more confident predictions of accuracy and establish confidence intervals.

reg_xval <- function(d) {
  plyr::ldply(1:10,function(i) {
    s <- runif(nrow(d)) > 0.1
    train <- d[s,]
    test <- d[!s,]
    reg_i <- glm(y ~ .,family=binomial(link='logit'),data=train)
    pred_i <- predict(reg_i,newdata=test,type='response')
    acc <- sum(test$y == (pred_i >= 0.5)) / nrow(test)
    ll <- logloss(pred_i,test$y)
    data.frame(acc=acc,ll=ll)
  })
}
logregs <- reg_xval(d_scaled)

# In order to compare this model with the others we'll develop, we can compile the results into
# a data frame

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
results <- newrow(logregs,'Logistic regression',TRUE)
results

# We can compare this to our results without cross-validation, which gave us an accuracy of `r acc` and 
# a log-loss of `r logloss(pred1,d_scaled$poor_scaled)`. According to both metrics, the average performance
# in cross-validation is slightly lower (as we would expect).

results <- results %>%
  rbind(newrow(reg_xval(d_unscaled),'Logistic regression',FALSE))

## LASSO regression

# Another variation of regression comes from regularization. Regularization tries to optimize two
# things at the same time -- fitting the data as well as possible, while also trying to keep model
# weights small. In the case of LASSO [Least Absolute Shrinkage and Selection Operator], regularization tends to push model weights
# to zero, leaving us with a more parsimonious model that eliminates some variables. This leads
# to models that are simpler to understand, but may suffer from a performance tradeoff.

# LASSO involves a parameter called lambda that controls the relative balance between
# optimizing fit and constraining model weights. The non-LASSO example above is 
# equivalent to lambda=0, where weights are not constrained at all. The function
# `cv.glmnet` gives us an automated way to find the appropriate value of lambda
# using cross-validation.

mm <- model.matrix(poor_scaled~.,d_scaled)
y <- as.numeric(d_scaled$poor_scaled)
cv1 <- cv.glmnet(mm,y,alpha=1,family='binomial',type.measure='class')
plot(cv1)

# This plot shows the logarithm of lambda and the misclassification error. As lambda
# increases (moving to the right), the error also increases but the model becomes
# simpler. The vertical dashed lines mark the minimum error 
# (for a very small lambda), and the value of lambda that is one standard
# deviation above this minimum. We'll take this one-SD point as a reasonable
# balance between accuracy and simplicity. The numbers across the top of the plot
# show the number of variables with non-zero coefficients -- we've nearly 
# cut the complexity of our model in half.

weights <- coef(cv.out,s=cv.out$lambda.1se)[3:nrow(weights),]
weights[weights != 0]

# We have `r sum(weights != 0)` non-zero weights.
# *TODO:* I'd like to make a scatter plot like the one above, but can't figure out how to get p-values from a `glmnet` object.

# We can now do the same type of cross-validation as earlier to see 
# how the LASSO model compares with the standard approach:

lasso_xval <- function(d,niter=10) {
  plyr::ldply(1:niter,function(i) {
    paste(i,Sys.time()) %>% print
    s <- runif(nrow(d)) > 0.1
    train <- d[s,]
    test <- d[!s,]
    train_mm <- model.matrix(y~.,train)
    y <- as.numeric(train$y)
    cv_i <- cv.glmnet(train_mm,y,alpha=1,family='binomial',type.measure='class')
    test_mm <- model.matrix(y~.,test)
    pred_i <- predict(cv_i,newx=test_mm,type='response')
    acc <- sum(test$y == (pred_i >= 0.5)) / nrow(test)
    ll <- logloss(pred_i,test$y)
    data.frame(acc=acc,ll=ll)
  })
}
results <- results %>%
  rbind(newrow(lasso_xval(d_scaled,2),'LASSO regression',TRUE)) %>%
  rbind(newrow(lasso_xval(d_unscaled,2),'LASSO regression',FALSE)) 
results

# By both of our model evaluation metrics, the performance of LASSO regression was slightly worse than
# the standard logistic regression. While the LASSO model is simpler, it's still fairly complex, and a model 
# with 66 non-zero weights might still not be very interpretable.

write.csv(results,'results.csv',row.names=FALSE)
