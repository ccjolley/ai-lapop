library(dplyr)
library(ggplot2)
library(mice)
library(forcats)
library(haven)

###############################################################################
# Prepare data for analysis by simplifying multilevel factors and imputing
# missing values.
###############################################################################

setwd("C:/Users/Craig/Desktop/Live projects/AI/AI_metrics/LAPOP")
my_data <- read.csv('ml_me.csv')
lapop <- read_dta('AmericasBarometer Grand Merge 2004-2014 v3.0_FREE.dta')
# which variables are better-described as factors rather than integers?
# (i.e. they can't be interpreted as a quantity of something)
fac_vars <- c('ur','q1','aoj22','dem2','etid','ocup4a','prot3','q3c','q10a',
              'q14','vb1','vb2','vb10','vic1ext','vic1hogar','vic44','leng1')
for (f in fac_vars) {
  my_data[,f] <- as.factor(my_data[,f])
}

### Imputation seems not to work unless I can aggregate some of these factor 
### variables with lots of levels into something simpler.

# FWIW, this potentially has a *lot* to do with bias and appropriate use in AI
# I need to make judgement calls about which languages, ethnicities, or 
# religions are similar enough that they can be lumped together, or minor 
# enough to just get tossed into "Other." Not to mention interpreting what
# those labels mean in the first place, given that survey respondants in 
# different countries didn't see the exact same list of options.

# Aggregate factors in leng1 into major groups
leng1_labels <- attr(lapop[['leng1']],'labels')
spanish <- leng1_labels[grep('Spanish|astellano',names(leng1_labels))] %>% as.character
foreign <- leng1_labels[grep('oreign|Chinese|Hindi|Javanese|Dutch|German|French|^Other$',names(leng1_labels))] %>% as.character 
english <- leng1_labels[grep('^English',names(leng1_labels))] %>% as.character 
portuguese <- leng1_labels[grep('Portuguese',names(leng1_labels))] %>% as.character
creole <- leng1_labels[grep('Creole|Criollo|Patois',names(leng1_labels))] %>% as.character
bad <- leng1_labels[grep('No|Don\'t',names(leng1_labels))] %>% as.character
indig <- leng1_labels %>% # all others are indigenous languages (as far as I know)
  as.character %>% 
  setdiff(c(spanish,foreign,english,portuguese,creole,bad)) 
my_data$leng1 <- fct_collapse(my_data$leng1,
                              spanish=spanish,
                              foreign=foreign,
                              english=english,
                              portuguese=portuguese,
                              creole=creole,
                              indig=indig)
rm(spanish,foreign,english,portuguese,creole,bad,indig,leng1_labels)
my_data$leng1 %>% table

# Aggregate factors into etid into major groups
etid_labels <- attr(lapop[['etid']],'labels')
mestizo <- etid_labels[grep('Mestizo',names(etid_labels))] %>% as.character
indian <- etid_labels[grep('Indian|Indo',names(etid_labels))] %>% as.character
white <- etid_labels[grep('White|Spanish|Jews|Mennonite|Portuguese',names(etid_labels))] %>% as.character
black <- etid_labels[grep('Black|Afro|Mullatto|Garifuna|Maroon|Creole|Moreno',names(etid_labels))] %>% as.character
native <- etid_labels[grep('Indigenous|Quechua|Aymara|Amazon|Zamba|Maya',names(etid_labels))] %>% as.character
others <- etid_labels[grep('Other|Javanese|Syrian|Chinese|Oriental|Yellow',names(etid_labels))] %>% as.character
bad <- etid_labels[grep('^No|^Don\'t',names(etid_labels))] %>% as.character
etid_labels[!(etid_labels %in% c(mestizo,indian,white,black,native,others,bad))]
my_data$etid <- fct_collapse(my_data$etid,
                             mestizo=mestizo,
                             indian=indian,
                             white=white,
                             black=black,
                             native=native,
                             others=others)
rm(etid_labels,mestizo,indian,white,black,native,others,bad)
my_data$etid %>% table

# Now for q3c (religion)
q3c_labels <- attr(lapop[['q3c']],'labels')
catholic <- q3c_labels[grep('Catholic',names(q3c_labels))] %>% as.character
mainline <- q3c_labels[grep('Mainline',names(q3c_labels))] %>% as.character
evang <- q3c_labels[grep('^Evangelical',names(q3c_labels))] %>% as.character
other_christian <- q3c_labels[grep('Mormon|Jehovah|Orthodox',names(q3c_labels))] %>% as.character
native <- q3c_labels[grep('Native',names(q3c_labels))] %>% as.character
other <- q3c_labels[grep('Eastern|Kardecian|Muslim|Hindu|Other',names(q3c_labels))] %>% as.character
none <- q3c_labels[grep('None|Atheist',names(q3c_labels))] %>% as.character
q3c_labels[!(q3c_labels %in% c(catholic,mainline,evang,other_christian,native,other,none))]
my_data$q3c <- fct_collapse(my_data$q3c,
                            catholic=catholic,
                            mainline=mainline,
                            evang=evang,
                            other_christian=other_christian,
                            native=native,
                            other=other,
                            none=none)
rm(catholic,mainline,evang,other_christian,native,other,none,q3c_labels)
my_data$q3c %>% table

my_data <- my_data %>% select(-sex) %>% select(-pid3_t)
# not sure what pid3_t was doing in there in the first place
rm(lapop)

# Fill in missing values
x <- select(my_data,-wealth)

bench <- function(n) {
  just_n <- x[sample(nrow(x),n),] %>% droplevels
  ti <- Sys.time()
  imp <- mice(just_n,printFlag=F,seed=12345)
  tf <- Sys.time()
  tf - ti
}
set.seed(12345)
sapply(c(100,200,500,1000,2000,5000,10000,20000,50000),function(i) {
  paste0(i,' ',bench(i)) %>% print
})
# Based on benchmarking, the full 69,321 should take ~2h

###############################################################################
# First attempt: Linear regression
# NOTE: Doing initial coding with imputed values removed; update this after
# I've done an overnight imputation run. Also worth checking to see how much
# my RMSE changes after I add imputed data.
###############################################################################

# get rid of this code for production
my_data_all <- my_data
my_data <- na.omit(my_data_all)

my_lm <- lm(wealth ~ .,my_data)
summary(my_lm)
# R^2 = 0.3332
# I'm more interested in the RMSE
lm_pred <- predict(my_lm)
(lm_pred - my_data$wealth)^2 %>% sum %>% sqrt # 146.7336

###############################################################################
# Second attempt: k-nearest neighbors
# NOTE: Doing initial coding with imputed values removed; update this after
# I've done an overnight imputation run.
###############################################################################
library(FNN)

# First, create dummy variables for multi-level factors
for (x in fac_vars) {
  paste0(x,' ',my_data[,x] %>% unique %>% length) %>% print
}
w_dummies <- my_data %>%
  cbind(model.matrix(~leng1 - 1, data=my_data)) %>%
  cbind(model.matrix(~etid - 1, data=my_data)) %>%
  cbind(model.matrix(~q3c - 1, data=my_data)) %>%
  cbind(model.matrix(~aoj22 - 1, data=my_data)) %>%
  cbind(model.matrix(~dem2 - 1, data=my_data)) %>%
  cbind(model.matrix(~ocup4a - 1, data=my_data)) %>%
  cbind(model.matrix(~vb1 - 1, data=my_data)) %>%
  select(-leng1,-etid,-q3c,-dem2,-ocup4a,-aoj22,-vb1) %>%
  mutate(ur=as.numeric(ur),
         q1=as.numeric(q1),
         prot3=as.numeric(prot3),
         q10a=as.numeric(q10a),
         q14=as.numeric(q14),
         vb2=as.numeric(vb2),
         vb10=as.numeric(vb10),
         vic1ext=as.numeric(vic1ext),
         vic1hogar=as.numeric(vic1hogar),
         vic44=as.numeric(vic44))

knn_rmse2 <- sapply(20:50,function(k) {
  my_knn <- knn.reg(train=select(w_dummies,-wealth),
                    y=w_dummies$wealth,
                    k=k,algorithm='kd_tree')
  rmse <- (my_knn$pred - my_data$wealth)^2 %>% sum %>% sqrt
  paste0(k,' ',rmse) %>% print
  rmse
}) 
# optimized at k=41, RMSE=157.982391150706 (not as good as lin reg)

###############################################################################
# Third attempt: Random forest
# NOTE: Doing initial coding with imputed values removed; update this after
# I've done an overnight imputation run.
###############################################################################
library(randomForest)

rf_bench <- function(n) {
  just_n <- my_data[sample(nrow(my_data),n),]
  x <- select(just_n,-wealth)
  y <- just_n$wealth
  ti <- Sys.time()
  rf_n <- randomForest(x=x,y=y)
  tf <- Sys.time()
  rmse <- (rf_n$predicted - y)^2 %>% sum %>% sqrt
  paste0(n,' ',rmse,' ',tf-ti) %>% print
  tf-ti
}

bench_times <- sapply(c(10,20,50,100,200,500,1000,2000,5000,10000,20000),rf_bench)


my_rf <- randomForest(x=select(my_data,-wealth),y=my_data$wealth)
(my_rf$predicted - my_data$wealth)^2 %>% sum %>% sqrt