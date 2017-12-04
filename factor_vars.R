library(dplyr)
library(ggplot2)
library(mice)
library(forcats)
library(haven)

###############################################################################
# Prepare data for analysis by simplifying multilevel factors and imputing
# missing values.
###############################################################################

setwd("C:/Users/Craig/Desktop/Live projects/AI/LAPOP")
my_data <- read.csv('ml_me.csv')
lapop <- read_dta('AmericasBarometer Grand Merge 2004-2014 v3.0_FREE.dta') %>% 
  filter(year >= 2012)


# We're going to determine which variables are well-suited to be factors by 
# collecting some information about variables and their labels into a data frame.
lvars <- intersect(names(lapop),names(my_data))
codes <- data_frame(var=lvars,
                    label=sapply(lapop[,lvars],function(x) attr(x,'label')),
                    first=unlist(sapply(lapop[,lvars],function(x) 
                      names(attr(x,'labels'))[1])),
                    first_val=unlist(sapply(lapop[,lvars],function(x) 
                      attr(x,'labels')[1])),
                    second=unlist(sapply(lapop[,lvars],function(x) 
                      names(attr(x,'labels'))[2])),
                    second_val=unlist(sapply(lapop[,lvars],function(x) 
                      attr(x,'labels')[2])),
                    num_unique=unlist(sapply(lapop[,lvars],function(x)
                      table(x) %>% length)))

head(codes,9)
# So, for example, we can see that some variables contain only a few options; `ur` is 
# only answered as Urban/Rural, and q1 is only Male/Female. Age (`q2`) is clearly a numeric
# scale, with 84 unique responses given to this question. In other cases, such as `municipio`,
# we have a variable with a large number (1,158) of qualitatively different responses.

# This process of deciding which variables are factors and which are a 
# continuous scale involves some judgements. For example, idio2 (Perception of Personal Economic Situation)
# has three responses: Better, Same, and Worse. Are these three distinct categories, or should they
# be assigned values on a scale of 1-3?

# We can categorize many variables by looking at the text used to describe responses. 
# A few don't fall into 
# these neat categories and we'll need to place them in lists by hand.

# Binary variables are those for which only two responses were available
binary <- (codes %>% filter(num_unique==2))$var
codes %>% filter(var %in% binary) %>% select(var,label,first,second)
# For some other factor variables, a small number of qualitatively-different responses were available.
other_factor <- c('estratosec','tamano','aoj22','dem2','idio2','ocup4a','q10d','q10e','soct2','vb1')
codes %>% filter(var %in% other_factor) %>% select(var,num_unique,label)
# A few factor variables involved a large number of possible responses; we'll need to handle those
# differently.
big_factor <- c('pais','estratopri','prov','municipio','etid','leng1','q3c')
codes %>% filter(var %in% big_factor) %>% select(var,num_unique,label,first)
# Scale variables
# Often, we could spot scale variables based on key words that denote one end of a scale of values
scale <- (codes %>% filter(grepl('Strong|Very|(A Lot)|(Not at All)|(Once a Week)|Daily|None',first)))$var
codes %>% filter(var %in% scale) %>% select(var,label,num_unique,first)
# There were a few other scale variables
other_scale <- c('q2','colorr','q12c')
codes %>% filter(var %in% other_scale) %>% select(var,label,num_unique,first,second)

# Note that, in the case of `q2` (age) and `q12c` (number of people in household), all of the
# valid responses are numbers. This means that the first *labeled* response is "Don't know", which
# gets coded as a missing response, rather than as a number.

for (f in c(binary,other_factor,big_factor)) {
  my_data[,f] <- as.factor(my_data[,f])
}

# Before we can use these data as inputs for a machine learning algorithm, we'll
# need to deal with missing data. Here are some of the variables that are missing most frequently.
codes$missing <- sapply(codes$var, function(x) my_data[,x] %>% is.na %>% mean)
codes %>% arrange(desc(missing)) %>% select(var,label,missing,num_unique) %>% head

# One thing that will be tricky here is that imputation algorithms generally don't work well
# with factor variables that have a very large number of categories. This is because it's hard
# for the algorithm to learn enough about the characteristics of a category to assign anything
# to it with much reliability. We'll need to aggregate some of these factor 
# variables with lots of levels into something simpler. Which factor variables are problems?
codes %>% filter(var %in% big_factor) %>% 
  select(var,num_unique,missing) %>% arrange(desc(missing)) %>% head

# We need to make judgement calls about which languages, ethnicities, or 
# religions are similar enough that they can be lumped together, or minor 
# enough to be grouped into an "Other" category.

leng1_labels <- attr(lapop[['leng1']],'labels')

# This is complicated further by the fact that the same language appears under
# several different codes, based on the country survey. For example, Spanish
# appears in 14 different places on the list.
leng1_labels[which(names(leng1_labels)=='Spanish')]

# As a result, we'll lump different languages together by searching through the 
# text labels, rather than relying on the numbers. There's a small catch, though --
# not all the same options were presented in each country. For example, let's look
# at which countries used a label for Chinese in `leng1`:
pais_labels <- data.frame(country_name=names(attr(lapop[['pais']],'labels')),
                          pais=attr(lapop[['pais']],'labels') %>% as.factor)
my_data %>% filter(leng1 %in% leng1_labels[which(names(leng1_labels)=='Chinese')]) %>%
  select(leng1,pais) %>% unique %>%
  left_join(pais_labels)

# If we took the survey results at face value, we would conclude that *all* the Chinese
# speakers in Latin America live in these four countries. This is unlikely to be true;
# it is more likely that any Chinese speakers in other countries got lumped into an
# "Other" category, possibly along with speakers of Hindi, Dutch and German. In other
# countries however, speakers of those languages got their own categories. As a result,
# we'll end up focusing on a small number of widely-spoken languages, where survey design
# would have been more consistent across the region.

# It's also possible to look at this in the other direction, to see (for example), which
# language responses were available for survey-takers in Mexico:
leng1_df <- data.frame(lang_name=names(leng1_labels),leng1=as.factor(leng1_labels))
pais_labels %>% filter(country_name=='Mexico') %>%
  left_join(my_data %>% select(pais,leng1) %>% unique,by='pais') %>%
  left_join(leng1_df,by='leng1') %>%
  arrange(leng1)

# Note that R's character encodings (with their bias toward English) mangled Espanol and Nahuatl
# to Espa<f1>ol and N<e1>huatl.

# We'll take the 154 language labels present in the dataset and condense them into a few
# groups: Spanish, English, Portugese, Creole, Indigenous, and Other. Note that the analyst's
# Anglophone bias is already showing through here -- English may not actually seem so important
# to someone who isn't based in the U.S. More significantly, the speakers of languages referred to 
# as "Creole" in Haiti, Belize, Barbados and the Bahamas may not agree that they all speak the 
# same language, or that Jamaican Patois belongs in the same group. Similarly, "indigenous" is a
# very broad category, including languages such as Garifuna with significant non-indigenous influences.

spanish <- leng1_labels[grep('Spanish|astellano',names(leng1_labels))] %>% as.character
foreign <- leng1_labels[grep('oreign|Chinese|Hindi|Javanese|Dutch|German|French|^Other$',names(leng1_labels))] %>% as.character 
english <- leng1_labels[grep('^English',names(leng1_labels))] %>% as.character 
portuguese <- leng1_labels[grep('Portuguese',names(leng1_labels))] %>% as.character
creole <- leng1_labels[grep('Creole|Criollo|Patois',names(leng1_labels))] %>% as.character
bad <- leng1_labels[grep('No|Don\'t',names(leng1_labels))] %>% as.character
indig <- leng1_labels %>% 
  as.character %>% 
  setdiff(c(spanish,foreign,english,portuguese,creole,bad)) 
# There are 50 unique language labels in our "indigenous" category:
leng1_labels[as.character(leng1_labels) %in% indig] %>% names %>% unique
my_data$leng1 <- fct_collapse(my_data$leng1,
                              spanish=spanish,
                              foreign=foreign,
                              english=english,
                              portuguese=portuguese,
                              creole=creole,
                              indig=indig)
rm(spanish,foreign,english,portuguese,creole,bad,indig,leng1_labels)
table(my_data$leng1) %>% as.data.frame %>%
  ggplot(aes(x=fct_reorder(Var1,Freq),y=Freq)) +
    geom_bar(stat='identity') +
    coord_flip()
# TODO: I know there's a way to make this prettier with forcats, but I need internet to look it up

# It might be a little surprising that English ranked as having almost half as many speakers 
# as Spanish. We can understand this a little better by looking at the number of survey responses
# from each country. 

table(my_data$pais) %>% as.data.frame %>%
  rename(pais=Var1) %>% left_join(pais_labels,by='pais') %>%
  mutate(country_name=as.character(country_name),
         country_name=ifelse(grepl('Hait',country_name),'Haiti',country_name)) %>%
  ggplot(aes(x=country_name,y=Freq)) +
    geom_bar(stat='identity') +
    coord_flip()

# Most countries have a roughly-similar number of responses, with larger samples coming from 
# a few small Caribbean nations. This means that small English-speaking countries like Jamaica
# or Barbados are dramatically over-represented relative to much more populous Spanish-speaking
# countries like Mexico and Colombia. When we are simply taking averages, this can be corrected
# easily by reweighting the individuals that are representative of larger populations. Similar
# reweighting can be more challenging with some machine-learning models, and may lead to models
# that are more accurate in countries with smaller populations.

# We need to do some similar categorization with the `etid` (ethnicity) variable.
# The categories below seem plausible enough, but imposing a small number of ethnic labels on the 
# Latin American melting pot is always going to be a challenge. The choices we've made here almost
# certainly reflect the point of view of North Americans with very limited knowledge of Latin 
# American or Caribbean cultures, and members of the groups being categorized would likely take 
# issue with some of the choices. There's no way to get around this without actually going out
# and talking to people from the affected groups and getting their input on which classifications
# make sense in context.

# Aggregate factors into etid into major groups
etid_labels <- attr(lapop[['etid']],'labels')
mestizo <- etid_labels[grep('Mestizo',names(etid_labels))] %>% as.character
indian <- etid_labels[grep('Indian|Indo',names(etid_labels))] %>% as.character
white <- etid_labels[grep('White|Spanish|Jews|Mennonite|Portuguese',names(etid_labels))] %>% as.character
black <- etid_labels[grep('Black|Afro|Mullatto|Garifuna|Maroon|Creole|Moreno',names(etid_labels))] %>% as.character
native <- etid_labels[grep('Indigenous|Quechua|Aymara|Amazon|Zamba|Maya',names(etid_labels))] %>% as.character
others <- etid_labels[grep('Other|Javanese|Syrian|Chinese|Oriental|Yellow',names(etid_labels))] %>% as.character
bad <- etid_labels[grep('^No|^Don\'t',names(etid_labels))] %>% as.character
#etid_labels[!(etid_labels %in% c(mestizo,indian,white,black,native,others,bad))]
my_data$etid <- fct_collapse(my_data$etid,
                             mestizo=mestizo,
                             indian=indian,
                             white=white,
                             black=black,
                             native=native,
                             others=others)
rm(etid_labels,mestizo,indian,white,black,native,others,bad)
table(my_data$etid) %>% as.data.frame %>%
  ggplot(aes(x=Var1,y=Freq)) +
  geom_bar(stat='identity') +
  coord_flip()

# The very large population categorized as "Black" seems surprising, until we consider the 
# overrepresentation of small English-speaking Caribbean nations that was noted above.

# Now for q3c (religion). As before, this requires some value judgements to be made by the analysnt.
# Members of the "Other" category might chafe at being lumped together, for example.
q3c_labels <- attr(lapop[['q3c']],'labels')
catholic <- q3c_labels[grep('Catholic',names(q3c_labels))] %>% as.character
mainline <- q3c_labels[grep('Mainline',names(q3c_labels))] %>% as.character
evang <- q3c_labels[grep('^Evangelical',names(q3c_labels))] %>% as.character
native <- q3c_labels[grep('Native',names(q3c_labels))] %>% as.character
other <- q3c_labels[grep('Mormon|Jehovah|Orthodox|Eastern|Kardecian|Muslim|Hindu|Other',names(q3c_labels))] %>% as.character
none <- q3c_labels[grep('None|Atheist',names(q3c_labels))] %>% as.character
q3c_labels[!(q3c_labels %in% c(catholic,mainline,evang,native,other,none))]
my_data$q3c <- fct_collapse(my_data$q3c,
                            catholic=catholic,
                            mainline=mainline,
                            evang=evang,
                            native=native,
                            other=other,
                            none=none)
rm(catholic,mainline,evang,native,other,none,q3c_labels)
table(my_data$q3c) %>% as.data.frame %>%
  ggplot(aes(x=Var1,y=Freq)) +
  geom_bar(stat='identity') +
  coord_flip()


# As you might have noticed, the `q1` and `sex` variables measure the exact same thing, so
# we can drop one of them. Also removing `estratopri` and `estratosec`, since the relevant
# geographic data is already captured by `prov` and `municipio`.
my_data <- my_data %>% select(-q1,-estratopri,-estratosec) 

# When we impute missing values, it would be cheating if we used information from
# our wealth-related variables. (This is an example of what is sometimes referred to as "data snooping"
# and can lead to poorly-performing models.) Also removing `prov` and `municipio` during imputation,
# because those are factor variables with lots of levels. They tend to slow things down dramatically
# and cause imputation to suck up a lot more memory. I'm hoping that I can get away with 
# leaving in `pais` for now.
x <- select(my_data,-w_scaled,-w_unscaled,-prov,-municipio)

# Because we're imputing a lot of variables on a fairly large dataset, we should benchmark 
# on smaller subsets of it to test how long things will take and make sure our computational
# resources are adequate. Many algorithms have performance that is roughly polynomial,
# and we can approximate the benchmarking curve using a power-law function. This should give
# us a rough estimate of whether imputation of the full dataset will take minutes or years; if
# the wait time is unacceptably long, we may need to remove more variables.

bench <- function(n) {
  set.seed(12345)
  just_n <- x[sample(nrow(x),n),] %>% droplevels
  ti <- Sys.time()
  imp <- mice(just_n,m=1,printFlag=T,seed=12345)
  tf <- Sys.time()
  difftime(tf,ti,units='secs')
}

n <- c(200,500,1000,2000)
b <- sapply(n,bench)
d <- data.frame(log_n=log(n),log_b=log(b))
b_lm <- lm(log_b ~ log_n,data=d)
p <- predict(b_lm,newdata=data.frame(log_n=log(nrow(x)),log_b=NA)) 
exp(p)/60

# Based on benchmarking with subsets of 200, 500, 1000, and 2000 rows, the full dataset
# should take about `r exp(p)/60` minutes to impute. How long does it take in reality?

# I commented these rows out because it takes an annoying long time; for now I'm just
# loading the saved result of having done this once. When I'm ready to make the final
# version of this I'll uncomment so it's pretty.
# ti <- Sys.time()
# imp <- mice(x,m=1,printFlag=T,seed=12345)
# Sys.time()-ti
# save(imp,file='imp.rda')
load(file='imp.rda')

# In reality, it took about 84 minutes for 69,321 rows.



# We'll combine the imputed data with the variables that weren't part
# of the imputation (`prov`,`municipio`, and the wealth indices). In
# addition, we'll want categorical variables to describe whether responses
# fall within the poorest 10% of the overall population.

w_imp <- complete(imp,1) %>%
  cbind(my_data[setdiff(names(my_data),names(x))]) %>%
  mutate(poor_unscaled=(w_unscaled <= quantile(w_unscaled,0.1)),
         poor_scaled=(w_scaled <= quantile(w_scaled,0.1)))

# Rather than saving our results as a CSV file, we'll save them in a binary
# format. That way we keep our factor variable assignments for future use.

save(w_imp,file='w_imp.rda')


write.csv(w_imp,'w_imp.csv',row.names=FALSE)