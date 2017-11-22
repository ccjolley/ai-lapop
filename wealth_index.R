library(haven)
library(dplyr)
library(mice)
library(ggplot2)
library(GGally)

# Load the LAPOP "Grand Merge" dataset and filter it to contain only records 
# from 2012 or later.

setwd("C:/Users/Craig/Desktop/Live projects/AI/LAPOP")
lapop <- read_dta('AmericasBarometer Grand Merge 2004-2014 v3.0_FREE.dta') %>%
  filter(year >= 2012)

# Select variables related to wealth

wealth_data <- lapop %>% 
  select(r3,r4,r4a,r5,r6,r7,r8,r12,r14,r15,r18,r1,r16,r26,q10new,q10g)

sapply(wealth_data,function(x) attr(x,'label'))

# Measuring wealth is tricky -- the most relevant measure for development is
# household consumption, but this is hard to assess in a one-time household
# survey. LAPOP's approach is to ask questions about assets (refrigerators,
# cell phones, microwave ovens, etc.) and also about income. 

# Assets can be somewhat problematic because some items (especially cell phones) have become
# dramatically more popular during the survey period -- relatively wealthy
# people had cell phones in 2012, while a much broader popluation uses them today.

# Income can also be diffuclt, because of differences in currencies and
# purchasing power between countries. Each household was asked to rate their 
# income on a scale from 0 to 16, with the meaning of each element on the 
# scale defined differently for each country.

# TODO: once I have internet access, I should look up the categories for each
# country and check how well they match in terms of PPP.

avg_wealth <- lapop %>% 
  group_by(pais) %>%
  summarize(income=mean(q10new,na.rm=TRUE)) %>%
  left_join(data.frame(pais=attr(lapop$pais,'labels'),
                       country=attr(lapop$pais,'labels') %>% names),by='pais')
print(avg_wealth)

# This makes sense intuitively; relatively prosperous countries like Brazil 
# and Uruguay score close to 10, while Guatemala and Haiti are closer to 5.

# We also don't know whether this scale is linear or something else.

# It's also important to think about missing values. Some survey questions (for
# example about TVs or cell phones) were answered by more than 99% of 
# households. About 23% didn't provide any information on income, however, and 
# 25% didn't say whether they have in-home internet service.

(is.na(wealth_data) %>% colSums) / nrow(wealth_data)

# We have a few options for dealing with missing data. One would be to simply
# ignore those records, dropping them completely from the dataset. This is 
# problematic for two reasons. One is that we end up throwing away a lot of data.

(wealth_data %>% na.omit %>% nrow) / nrow(wealth_data)

# In this case, we are left with only 32% of our original data if we disregard
# any households who failed to answer a question. It's also possible that 
# throwing away records with missing data may bias our results.

wealth_data %>% 
  mutate(r18ans=ifelse(is.na(r18),'NA',r18)) %>%
  group_by(r18ans) %>% 
  summarize(income=mean(q10new,na.rm=TRUE))

# What this table is telling us is that those who said they have in-home 
# internet service have an average income score of 10.3, while those without
# it have an income score of 6.8. Those who did not answer the question had
# an average income of 6.6. As a result, eliminating those who failed to 
# answer this question from the dataset would skew our results toward 
# higher-income households.

# What if we just assume that if someone refuses to answer a question about
# owning an asset then the answer must be no? 

wealth_data %>% 
  mutate(r4ans=ifelse(is.na(r4),'NA',r4)) %>%
  group_by(r4ans) %>% 
  summarize(income=mean(q10new,na.rm=TRUE))

# In the case of landline telephones, those who failed to answer the question 
# are intermediate in income between declared owners and non-owners. If we 
# assume that higher-income households are more likely to have a landline 
# phone, then it is likely that some non-answerers have one and others don't.

# Another possibility would be to replace missing values with an
# average value for the entire dataset. This would run into similar problems, 
# where (for the example of internet service) we assume that a lot of very poor
# households have internet when it's likely that they don't.

# It's also possible to use a machine learning method to predict the values of
# missing variables based on the questions that were answered. So, for example,
# households who didn't say whether they have a landline phone will be more
# likely to have one if they have a high income and lots of other assets.

# Here we'll use a method called MICE (Multiple Imputation through Chained ...)
# that attempts to predict missing values as well as possible while avoiding
# changes to the mean and variance of the variable. This assumes that the 
# distribution of missing values is random. We know that this isn't true --
# with the in-home internet example we saw that people who didn't answer the
# question are significantly poorer, on average, than those who did. So while
# approaches like MICE are better than simple averages, they can be unfairly
# skewed by significant patterns in missing variables.

w_imp <- mice(wealth_data,printFlag = T,seed=12345) 

# TODO: How hard would it be to do imputation differently, e.g. using KNN?

complete(w_imp,1) %>% 
  group_by(r18) %>% 
  summarize(income=mean(q10new,na.rm=TRUE))

# TODO: Comment on how this differs from what I have above

# This imputed dataset provides a fairly complex picture of a household's 
# assets and income. To simplify further analysis, we'll want to boil this
# down to a single number that captures their "level of wealth." We'll do
# this using principal components analysis (PCA). PCA is a method that takes
# a dataset in N dimensions (N=16 in this case) and converts it into N new
# variables such that variance is maximized by the lowest-numbered 
# principal components. In other words, the first principal component captures
# as much of the variation in the inital dataset as possible and is a good 
# candidate for our composite wealth index.

# We need to make some choices when we do PCA. First, our imputation using MICE
# is inherently random and produced five different versions of the imputed 
# dataset. We could base our wealth index off of just one of these imputations;
# what we'll do instead is to develop a wealth index for each and see how well
# they agree. Secondly, we can choose whether to center our variables (so that
# the mean is equal to zero) and/or scale them (so that the standard deviation 
# is 1) before beginning the PCA caluclation. PCA works best with a normal 
# distribution, and transforming variables in this way is widely considered a 
# good practice. Ours are decidedly not normal. For example, r1 (television in
# home) takes only two values (0=no, 1=yes), and 92% of households answered 
# yes. It may be safer not to rescale such a skewed distribution. For the sake
# of experimentation, we'll try it both ways.

pr <- lapply(1:5,function(x) prcomp(complete(w_imp,x),scale=FALSE,center=TRUE))
plot(pr[[1]])
summary(pr[[1]]) 
# 82.6% of the variance has been captured in the first component.
cor(pr[[1]]$x[,1],pr[[2]]$x[,1])
# correlation = 0.8196


pr_scale <- lapply(1:5,function(x) prcomp(complete(w_imp,x),scale=TRUE,center=TRUE))
plot(pr_scale[[1]])
summary(pr_scale[[1]]) 
qplot(pr_scale[[1]]$x[,1],pr_scale[[1]]$x[,2]) + theme_classic()
cor(pr_scale[[1]]$x[,1],pr_scale[[2]]$x[,1])
# correlation = 0.992

# This time, only 34.2% of variance was captured in the first component. My
# suspicion is that the unscaled index weights income a lot more heavily
# (because those numbers are larger) and doesn't lean as much on assets. While the
# index with scaling captures more of the overall variance, it seems to be less
# robust, probably because it relies on a smaller set of variables.

# How well do my two indices correlate with each other?
cor(pr_scale[[1]]$x[,1],pr[[1]]$x[,1])
# 0.602 - not very well. 

# We've run the PCA calculation for each of our five imputations; we want to
# compare these to each other to see how consistent the construction of the 
# wealth index was. The easiest way to do this is to make pairwise scatter
# plots of the first principal component for the five different analyses. 
# Because each imputation contains 69,321 rows, we're going to randomly sample
# just 200 points for easier plotting.
all_pc1 <- data.frame(plyr::llply(1:5, function(i) pr[[i]]$x[,1])) 
names(all_pc1) <- c('imp1','imp2','imp3','imp4','imp5')
sample_n(all_pc1,200) %>% ggpairs() + theme_classic()

# The lower-left half of this "matrix" contains the pairwise scatterplots, and the
# upper-right half contains correlation coefficients (which can range from -1 to 1).
# The diagonal displays the distribution for each imputation result. They all look 
# a little different, but are highest at intermediate values and taper off 
# for high and low levels of the index. Each distribution has a double peak, with 
# two different wealth values that are especially common.
# Note that we don't know yet whether a high
# index value corresponds to high or low wealth; we'll check that later.

all_pc1_scale <- data.frame(plyr::llply(1:5, function(i) pr_scale[[i]]$x[,1])) 
names(all_pc1_scale) <- c('imp1','imp2','imp3','imp4','imp5')
sample_n(all_pc1_scale,200) %>% ggpairs() + theme_classic()

# In contrast, the version of the index that used scaling in the PCA 
# calculation shows much more consistency between different imputations.
# The overall shape is similar, although the double peak that appeared in 
# the unscaled index is gone.

all_pc1$avg <- rowMeans(all_pc1)
all_pc1_scale$avg <- rowMeans(all_pc1_scale)

# This is a function to visualize the effect of index on each component variable,
# so that we can see whether high values of the index correspond to high or low 
# wealth. Because we'll be making the same plot twice, abstracting it out into a
# function allows us to re-use code rather than repeating a lot of typing.
quantile_plot <- function(idx) {
  q25 <- quantile(idx)[2]
  q75 <- quantile(idx)[4]
  qchange <- data.frame(low=colMeans(wealth_data[idx <= q25,],na.rm=TRUE),
                        hi=colMeans(wealth_data[idx >= q75,],na.rm=TRUE))
  qchange[c('q10new','q10g'),'low'] <- qchange[c('q10new','q10g'),'low']/16
  qchange[c('q10new','q10g'),'hi'] <- qchange[c('q10new','q10g'),'hi']/16
  qchange$slope <- qchange$hi - qchange$low
  qchange$label <- rownames(qchange)
  qchange$even <- rank(qchange$hi)/nrow(qchange)*max(c(qchange$low,qchange$hi))
  
  ggplot(data=qchange) +
    geom_segment(aes(x=0,y=qchange$low,xend=1,yend=qchange$hi,
                     color=qchange$slope),size=1) +
    scale_color_gradient(low='#FF420E',high='#89DA59') +
    geom_text(aes(x=1,y=qchange$even,label=qchange$label,
                  hjust=0,color=qchange$slope),size=5) +
    annotate("text",x=0,y=0,label='Average in lowest quartile',hjust=0) +
    annotate("text",x=1,y=0,label='Average in highest quartile',hjust=1) +
    scale_x_continuous(limits=c(0,1.5)) +
    theme_classic() +
    theme(legend.position='none',
          axis.ticks=element_blank(),
          axis.line=element_blank(),
          axis.text.x=element_blank(),
          axis.title=element_blank(),
          text=element_text(size=20))
}

# First, we'll apply our new function to the un-scaled index:
quantile_plot(all_pc1$avg)
# In this plot, each of the component variables is represented by a downward-
# sloping line. The color of this line relates to the steepness of the slope;
# steeper lines are red and more shallow ones are green. The labels on the 
# right-hand side correspond to the order of the endpoints on the right-hand
# side; this means that the top line represents r1 (television ownership), 
# followed by r4a (mobile phone ownership), and the bottom green line represents
# r8 (motorcycle ownership). Finally, all of the asset variables took values
# of 0 or 1, while the income variables (q10new and q10g) took values from
# 0 to 16. The averages for these variables were divided by 16 so that they 
# could be displayed on the same vertical scale.

# The first thing that jumps out in this plot is that all lines slope downward.
# This means that households with a high value of this index have lower incomes
# and own fewer assets; we've actually created an index for poverty, not wealth.

# What we see in this plot is that some assets -- like televisions and mobile
# phones -- are popular among people with widely-varying levels of wealth. 
# Television ownership (r1) is nearly 100% among the richest 25% of the population,
# and more than 75% among the poorest 25%. Conversely, motorcycles (r8) are less
# common among both rich and poor. The steepest, reddest lines are for the income
# variables q10new and q10g -- this suggests that these variables do the most work
# in separating rich from poor with this index.

quantile_plot(all_pc1_scale$avg)

# Comparing this plot to the previous one, we see some common features. All 
# lines slope downward, meaning that households with higher index values are poorer.
# Television (r1) and mobile phone (r4a) ownership still have fairly flat lines
# near the top of the plot, and motorcycle ownership (r8) has a fairly flat line 
# near the bottom. The major difference is that many more variables show steep,
# red lines. In fact, our income variables (q10new and q10g) are no longer the
# steepest. Scaling our variables before PCA put them on a more equal footing,
# so that this index incorporates more information about asset ownership.

# This could be one reason that the calculation of this index was more consistent
# between imputations. The unscaled PCA index relied heavily on a single variable 
# that was missing fairly frequently, resulting in a lot more noise from imputation.
# Our asset-ownership data required much less imputation, and a wealth scale based
# on these more complete data is probably more reliable.

# Finally, we'll make our wealth indices look a little nicer by inverting them
# (to make higher values mean more wealth), and re-scaling them to have a mean
# of zero and a standard deviation of 1.
all_pc1$norm <- -scale(all_pc1$avg)
all_pc1_scale$norm <- -scale(all_pc1_scale$avg)

# Now that we have developed our wealth indices, we can put the rest of the 
# dataset in a form that will be more suitable for ML testing. First, we'll
# want to filter it so that we're only looking at dates later than 2012 (the
# same time window used to construct the wealth index).
recent <- lapop %>%
  filter(year >= 2012)

# Next, we'll need to deal with missing variables. Because different versions
# of the survey were used in different countries and different years, there 
# will be a lot of them. In order to keep the imputation tractable (and reasonably
# reliable), we'll restrict our attention to the variables that are present at
# least 90% of the time. This cutoff is somewhat arbitrary; choosing a more generous
# cutoff will give us more variables with less reliability.
na_frac <- (is.na(recent) %>% colSums) / nrow(recent)
sum(na_frac < 0.1)

# Out of 1791 total variables, only 106 of them are present at least 96% of the time.
# In addition, some variables are probably less relevant to our analysis. We're going
# to skip variables that probably don't tell us much about the household in question,
# such as the language of the survey or the sex of the interviewer. I'm dropping 
# variables that only tell us about the survey sampling (such as sampling strata and 
# primary sampling unit) while keeping those with explicit geographic information.
# Also dropping variables that contain the exact same information as others (such as
# nationality and q2y).

drop_vars <- c('wave','year','idnum','idnum_14','strata','upm','cluster',
               'idiomaq','fecha','wt','weight1500','sexi','colori',
               'uruvbi6notr','uruvbi7notr','nationality','q2y','pid3_t')
good_vars <- na_frac[na_frac < 0.1] %>% 
  names %>% setdiff(drop_vars) %>% setdiff(names(wealth_data))

# We're now down to just 76 variables that we will attempt to correlate with 
# our wealth indices.

ml_me <- recent[,good_vars] %>%
  mutate(w_unscaled=all_pc1$norm,
         w_scaled=all_pc1_scale$norm)

write.csv(ml_me,'ml_me.csv',row.names=FALSE)

# Note that this dataset isn't quite ready to go yet. There are still a lot of missing
# variables in there. We also still haven't made decisions about which variables to treat
# as numeric (because they actually measure something on a scale) and which to treat as 
# factors (using numbers to label qualitatively different things).