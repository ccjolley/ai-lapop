library(haven)
library(ggplot2)
library(ggmap)
library(qgraph)
library(choroplethr)
library(dplyr)

setwd("C:/Users/Craig/Desktop/Live projects/AI/LAPOP")
load('w_imp.rda')
#lapop <- read_dta('AmericasBarometer Grand Merge 2004-2014 v3.0_FREE.dta') %>% 
#  filter(year >= 2012)

# Geographic - mapping out countries with high poverty rates. Which are statistically significant?
# Using just the scaled wealth index for now
# pais_labels <- data.frame(country_name=names(attr(lapop[['pais']],'labels')),
#                           pais=attr(lapop[['pais']],'labels') %>% as.factor)
# save(pais_labels,file='pais_labels.rda')
load('pais_labels.rda')

geo_poverty <- w_imp %>%
  group_by(pais) %>%
  summarize(poor_mean=mean(poor_scaled),poor_sum=sum(poor_scaled),
            poor2_mean=mean(poor_unscaled),poor2_sum=sum(poor_unscaled),
            n=n()) %>%
  left_join(pais_labels,by='pais') %>%
  mutate(country_name=as.character(country_name),
         country_name=ifelse(grepl('Hait',country_name),'Haiti',country_name)) 

map_me <- geo_poverty %>%
  select(country_name,poor_mean) %>%
  dplyr::rename(region=country_name,value=poor_mean) %>%
  filter(!(region %in% c('Bahamas','Barbados'))) %>%
  mutate(region=tolower(region))
country_choropleth(map_me,zoom=na.omit(map_me$region),num_colors=4)

# This isn't the most beautiful map, but it's probably the easiest to make.
# Most importantly, it clearly shows how the high-poverty countries (according
# to this metric) are clustered in Central America and Haiti.

geo_poverty %>%
  select(country_name,poor2_mean) %>%
  dplyr::rename(region=country_name,value=poor2_mean) %>%
  filter(!(region %in% c('Bahamas','Barbados'))) %>%
  mutate(region=tolower(region)) %>%
  country_choropleth(zoom=na.omit(map_me$region),num_colors=4)

# checking statistical significance of more poverty
geo_poverty$pval <- sapply(1:nrow(geo_poverty),function(i) {
  x <- c(geo_poverty$poor_sum[i],
         sum(geo_poverty$poor_sum)-geo_poverty$poor_sum[i])
  n <- c(geo_poverty$n[i],sum(geo_poverty$n)-geo_poverty$n[i])
  pt <- prop.test(x=x,n=n,alternative='greater')
  pt$p.value*nrow(geo_poverty) # correct for multiple hypotheses
})
geo_poverty <- arrange(geo_poverty,pval)
geo_poverty

# By this measure, only 7 countries, mostly in Central America, are significantly poorer
# Haiti, Nicaragua, Guatemala, Honduras, Belize, Guyana, El Salvador
# If I use the unscaled measure instead, the list is;
# Guatemala, Haiti, Honduras, Jamaica, Belize, Panama, Barbados

# Numeric variables with lots of values
glimpse(w_imp)
ggplot(w_imp,aes(x=q2)) +
  geom_histogram(binwidth=1,
                 fill='cornflowerblue',color='gray20') +
  xlab('Age') + ylab('Density')

ggplot(w_imp,aes(x=ed)) +
  geom_histogram(binwidth=1,
                 fill='cornflowerblue',color='gray20') +
  xlab('Years of schooling') + ylab('Density')

ggplot(w_imp,aes(x=q12)) +
  geom_histogram(binwidth=1,
                 fill='cornflowerblue',color='gray20') +
  xlab('# of children') + ylab('Density')

ggplot(w_imp,aes(x=q12c)) +
  geom_histogram(binwidth=1,
                 fill='cornflowerblue',color='gray20') +
  xlab('# of people in household') + ylab('Density')
# There were a few households that reported having a very large number of 
# people in them; this could make this variable do strange things in a 
# predictive model. We might be able to get around this by re-labeling
# as quantiles or by log-transforming it.

# TODO: how to calculate skew of scale variables? See which are skewed
# high or low, which are more peaked or more evenly distributed

# correlations and anticorrelations -- need dummy variables

scale_vars <- which(sapply(w_imp,is.numeric))
factor_vars <- c(which(sapply(w_imp,is.factor)),which(sapply(w_imp,is.logical)))
# ignore factors that still have a ton of levels
factor_vars <- factor_vars[!(names(factor_vars) %in% c('prov','municipio'))]
cors <- data.frame()
# First, correlate scale variables with each other
for (i in 1:(length(scale_vars)-1)) {
  ni <- names(w_imp)[scale_vars[i]]
  for (j in (i+1):length(scale_vars)) {
    nj <- names(w_imp)[scale_vars[j]]
    c <- cor(w_imp[scale_vars[i]],w_imp[scale_vars[j]]) %>% as.numeric
    addme <- data.frame(var1=ni,var2=nj,val=c,row.names=NULL)
    cors <- rbind(cors,addme)
  }
}

# scale-factor correlations
for (i in 1:length(scale_vars)) {
  ni <- names(w_imp)[scale_vars[i]]
  for (j in 1:length(factor_vars)) {
    tmp <- w_imp[,factor_vars[j]] %>% as.data.frame
    names(tmp) <- paste0(names(w_imp)[factor_vars[j]],'_')
    mm <- model.matrix(~ .+0, data=tmp) %>%
      as.data.frame()   
    if (ncol(mm)==2) { mm <- mm %>% select(1)}
    for (k in 1:ncol(mm)) {
      nk <- names(mm)[k]
      c <- cor(w_imp[scale_vars[i]],mm[k]) %>% as.numeric
      addme <- data.frame(var1=ni,var2=nk,val=c,row.names=NULL)
      cors <- rbind(cors,addme)
    }
  }
}

# factor-factor correlations
for (i in 1:(length(factor_vars)-1)) {
  tmp <- w_imp[,factor_vars[i]] %>% as.data.frame
  names(tmp) <- paste0(names(w_imp)[factor_vars[i]],'_')
  mm_i <- model.matrix(~ .+0, data=tmp) %>%
    as.data.frame()
  if (ncol(mm_i)==2) { mm_i <- mm_i %>% select(1)}
  for (j in (i+1):length(factor_vars)) {
    tmp <- w_imp[,factor_vars[j]] %>% as.data.frame
    names(tmp) <- paste0(names(w_imp)[factor_vars[j]],'_')
    mm_j <- model.matrix(~ .+0, data=tmp) %>%
      as.data.frame() 
    if (ncol(mm_j)==2) { mm_j <- mm_j %>% select(1)}
    for (mi in 1:ncol(mm_i)) {
      ni <- names(mm_i)[mi]
      for (mj in 1:ncol(mm_j)) {
        nj <- names(mm_j)[mj]
        c <- cor(mm_i[mi],mm_j[mj]) %>% as.numeric
        addme <- data.frame(var1=ni,var2=nj,val=c,row.names=NULL)
        cors <- rbind(cors,addme)
      }
    }
  }
}

fraction <- 0.01
cutoff <- quantile(abs(cors$val),1-fraction)
strong <- cors %>% 
  filter(abs(val) > cutoff) 

qgraph(strong,theme='classic',asize=0)
# This plot shows the strongest correlations between variables in our dataset. Positive correlations
# (i.e. variables that tend to move in the same direction) are indicated by green lines. For example, 
# there is a tightly-connected cluster with edges between governance-related questions such as
# `b1` (Courts guarantee fair trial), `b3` (Basic rights are protected), `n15` (Evaluation of Administration's
# Handling of Economy). Evidently, people's opinion of different aspects of governance tend to correlate.
# Negative correlations (where one variable increases when another decreases) are shown by red arrows. 
# For example, there is a strong negative correlation between www1 (Internet usage) and our scaled wealth
# index. (Note that large values of `www1` indicate _infrequent_ internet use.) Importantly, our target 
# variables are part of a wide-ranging set of correlations that touch on education, ethnicity, geography,
# and an urban-rural split.

# The presence of many strong correlations raises the question of whether we can reduce the dimensionality
# of this dataset using principal components analysis. We'll omit our wealth-related variables so that our 
# PCA results can still be used to predict them. We'll also omit our highly-diverse factor variables `prov` and
# `municipio`. We'll use `model.matrix` to convert multi-level factor variables into a set of binary variables.

x <- w_imp %>% select(-prov,-municipio,-w_unscaled,-w_scaled,-poor_unscaled,-poor_scaled)
pca <- model.matrix(~ ., data=x) %>%
  as.data.frame() %>%
  select(-`(Intercept)`) %>%
  prcomp(center=TRUE,scale=TRUE)
cum_var <- data.frame(pc=1:length(pca$sdev),
                      cv=summary(pca)$importance[3,])
cutoff <- 0.9
top_pc <- cum_var %>% filter(cv > cutoff) %>% select(pc) %>% min
ggplot(cum_var,aes(x=pc,y=cv)) +
  geom_point(size=2,color='tomato') +
  geom_vline(xintercept=top_pc,color='steelblue4')

# This is a rather diverse set of variables; it takes `r top_pc` to capture `r paste0(cutoff*100,'%')` of the
# variance. Even if we can't reduce the overall size of our dataset, PCA is still handy because it allows us
# to represent all of our data by numeric variables with comparable units, which can be useful for some ML methods. 
# Let's save this to a CSV file for future use.

pca$x %>% as.data.frame %>% 
  cbind(w_imp %>% select(prov,municipio,w_unscaled,w_scaled,poor_unscaled,poor_scaled)) %>%
  write.csv('pca.csv',row.names=FALSE)

pca$x %>% as.data.frame %>% 
  select(PC1,PC2) %>%
  ggplot(aes(x=PC1,y=PC2)) +
    geom_point(size=2,alpha=0.1,color='olivedrab')

# Finally, we'll try to see whether our data separate out into meaningful clusters of similar households. We'll use
# a method called k-means clustering, which separates the data into _k_ groups such that members of the same
# group are more similar to each other than they are to households outside the group. One difficulty here is that 
# we have to specify a value of _k_ in advance. One common practice is to look at the fraction of the variance
# explained by between-cluster variation for different values of _k_. Typically, this will increase rapidly at 
# first and then show an "elbow" where adding more clusters doesn't help explain any more variance.

maxc <- 50
clust_var <- sapply(1:maxc,function(k) {
  km <- kmeans(pca$x,k)
  paste(k,km$betweenss/km$totss) %>% print
  km$betweenss / km$totss
})
data.frame(k=1:maxc,v=clust_var) %>%
  ggplot(aes(x=k,y=v)) +
  geom_point(size=2,color='goldenrod3') +
  geom_line(color='goldenrod3') +
  xlab('Number of clusters') + ylab('Between-cluster variance')

# Unfortunately, this is a case where identifying the "elbow" is rather subjective. If we choose k=16, then
# we're able to explain about 20% of the total variance in terms of cluster membership. We can also explain
# 20% of the total variance with just the first 6 principal components, however. It seems like
# separation into qualitatively-different clusters isn't a big part of the story for this particular dataset.

# What happens if we vary number of PCs for a single value of k?

k <- 16
cols_var <- sapply(1:ncol(pca$x),function(i) {
  km <- kmeans(pca$x[,1:i],k)
  paste(i,km$betweenss/km$totss) %>% print
  km$betweenss / km$totss
})
data.frame(k=1:length(cols_var),v=cols_var) %>%
  ggplot(aes(x=k,y=v)) +
  geom_point(size=2,color='darkorchid2') +
  geom_line(color='darkorchid2') +
  xlab('Number of PCs') + ylab('Between-cluster variance')

# What we're seeing here is that 16 clusters are able to capture a rather large 
# fraction of the variance when we're looking at just a few principal components 
# and much less when we're looking at all of them. We would expect the higher-numbered
# principal components to be adding progressively smaller amounts of new information
# about the structure of the dataset; beyond a certain point they might just be adding noise.

# Counterfactual: random normals with the same means and sd as columns of pca$x

fake <- matrix(nrow=nrow(pca$x),ncol=ncol(pca$x))
for (i in 1:ncol(pca$x)) {
  fake[,i] <- rnorm(n=nrow(pca$x))*sd(pca$x[,i])
}

best <- 0
print('i,k,f_real,f_fake')
plotme <- data.frame()
for (k in 1:32) {
  for (i in 1:ncol(pca$x)) {
    km_real <- kmeans(pca$x[,1:i],k)
    f_real <- km_real$betweenss / km_real$totss
    km_fake <- kmeans(fake[,1:i],k)
    f_fake <- km_fake$betweenss / km_fake$totss
    d <- f_real - f_fake
    plotme <- rbind(plotme,data.frame(k=k,i=i,d=d))
    if (d > best) {
      best <- d
      paste(i,k,f_real,f_fake) %>% print
    }
  }
  #write.csv(partial,'partial4.csv',row.names=FALSE)
  #paste('Saving',nrow(partial),'rows; finished k =',k,'at',Sys.time()) %>% print
}

plotme <- plyr::ldply(c('partial.csv','partial2.csv','partial3.csv','partial4.csv'),read.csv)
save(plotme,file='plotme.rda')

load('plotme.rda')

ggplot(plotme,aes(x=i,y=k)) +
  geom_tile(aes(fill=d)) +
  scale_fill_gradient(low='white',high='steelblue') +
  xlab('Number of PCs') + ylab('Number of clusters') +
  theme_classic()

# This isn't quite finished yet, but there are a few things we can draw from it:
# - For a very small number of PCs (less than about 10), we don't really have enough
#   information for k-means to perform all that well. We'll have to see whether this
#   pattern also holds true when we train ML models on the PCA-transformed data.
# - For a small number of clusters (less than about )
# - In general, if you want to account for a larger number of PCs with some degree of
#   significance, you need more clusters. Conversely, you need more clusters (i.e. more data)
#   to outperform the random sampling on a large number of variables.
# - This isn't going to give us the "right" answer in terms of how many clusters we should
#   want. As we could suspect from earlier, there isn't really a natural number of clusters that
#   jumps out as being the best choice. If we want something that we can use as a factor
#   variable for ML, we should keep it fairly small. On the other hand, if our goal is to
#   pick out a bunch of small clusters and focus on the ones that fit some criteria (such as high
#   poverty), then we could get away with a larger number of clusters that capture more of
#   the overall variance.

# Let's look at the best clustering performance, as a function of k:

plotme %>% 
  group_by(k) %>%
  summarize(d=max(d)) %>%
  ggplot(aes(x=k,y=d)) +
    geom_point(size=2,color='steelblue') +
    geom_line(color='steelblue') +
    xlab('Number of clusters') + ylab('Best improvement over random') +
    theme_classic()

# Again, this is a little subjective, but it seems like we start to see diminishing returns
# after about k=10. How many principal components should we use when k=10?

plotme %>% filter(k==10) %>% arrange(desc(d)) %>% head

k10 <- kmeans(pca$x[,1:11],10)
w_imp$clust <- as.factor(k10$cluster)

# Do these clusters have anything to do with poverty?

ggplot(w_imp,aes(x=clust,y=w_scaled)) +
  geom_boxplot()

# Although we explicitly excluded wealth information when building our clusters, at least
# some show a correlation. In particular, clusters 1 and 9 are significantly poorer than
# the average, while clusters 2 and 7 are significantly richer.

# Note that, if we want to use k-means clustering in actual model building, we should *not*
# use the labels we've calculated here as features. When we're training models, we'll want 
# to evaluate them by holding out part of the data and using it to test the accuracy of
# our predictions. This means that when we're adding new attributes to our data (known 
# as feature engineering), we'll want to make sure that those attributes are based only on 
# the _training_ data, and never on the _test_ data.