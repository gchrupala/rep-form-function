library(dplyr)
library(ggplot2)

dep <- read.csv("../data/depparse_coco_val.csv", header=TRUE)
omi <- read.csv("../data/omission_coco_val.csv", header=TRUE)

omi_v  <- omi %>% select(-omission_t, -omission_lm) %>% rename(score=omission_v) %>% mutate(pathway="visual")
omi_t  <- omi %>% select(-omission_v, -omission_lm) %>% rename(score=omission_t) %>% mutate(pathway="textual")
omi_lm <- omi %>% select(-omission_t, -omission_v)  %>% rename(score=omission_lm) %>% mutate(pathway="lm")

data_v <- merge(dep, omi_v) 
data_t <- merge(dep, omi_t) 
data_lm <- merge(dep, omi_lm)

dep_lev <- (data_v %>% group_by(dep) %>% summarize(median_score=median(score)) %>% 
              data.frame %>% arrange(median_score))$dep 
postag_lev <- (data_v %>% group_by(postag) %>% summarize(median_score=median(score)) %>%
              data.frame %>% arrange(median_score))$postag

data <- rbind(data_t, data_v) %>% rbind(data_lm) %>% 
  mutate(dep = factor(dep, levels=dep_lev), 
         postag = factor(postag, levels=postag_lev))

filter_by_count <- function(dat, col, count) {
  dat[dat[,col] %in% names(table(dat[,col]))[table(dat[,col]) > count],]
}
MINCOUNT<-500


# FIGURE 2
ggplot(filter_by_count(data, "postag", MINCOUNT)) +
  aes(x=postag, y=score, color=pathway) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  #theme(text=element_text(size=15)) +
  xlab("Part of Speech") +
  ggsave(file="../doc/imaginet-omission-pos-boxplot.png",
         width=5, height=5)
   

ggplot(filter_by_count(data, "dep", MINCOUNT)) +
       aes(x=dep, y=score, color=pathway) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  #theme(text=element_text(size=15)) +
  xlab("Dependency") +
  ggsave(file="../doc/imaginet-omission-dep-boxplot.png",
         width=5, height=5)


# FIGURE 3
ratio <- filter(data, pathway=='visual')$score / filter(data, pathway=='textual')$score
data_r <- filter(data, pathway=="visual") %>% mutate(ratio=ratio, 
                                                    postag=reorder(postag, ratio, median),
                                                    dep=reorder(dep, ratio, median))

ggplot(filter_by_count(data_r,"postag",MINCOUNT) %>% filter,
       aes(x=postag, y=log(ratio))) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15)) +
  xlab("Part of Speech")
ggsave(file="../doc/imaginet-omission-ratio-pos-boxplot.png", width=5, height=5)

ggplot(filter_by_count(data_r,"dep",MINCOUNT) %>% filter,
       aes(x=dep, y=log(ratio))) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15)) +
  xlab("Dependency")
ggsave(file="../doc/imaginet-omission-ratio-dep-boxplot.png", width=5, height=5)

quotient <- filter(data, pathway=='lm')$score / filter(data, pathway=='textual')$score
data_q <- filter(data, pathway=="lm") %>% mutate(ratio=quotient, 
                                                     postag=reorder(postag, ratio, median),
                                                     dep=reorder(dep, ratio, median))

ggplot(filter_by_count(data_q,"postag",MINCOUNT) %>% filter,
       aes(x=postag, y=log(ratio))) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15)) +
  xlab("Part of Speech")
ggsave(file="../doc/imaginet-omission-quotient-pos-boxplot.png", width=5, height=5)

ggplot(filter_by_count(data_q,"dep",MINCOUNT) %>% filter,
       aes(x=dep, y=log(ratio))) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15)) +
  xlab("Dependency")
ggsave(file="../doc/imaginet-omission-quotient-dep-boxplot.png", width=5, height=5)

data_lr <- read.table("../data/ridge_scores.txt", header=TRUE) 
levels(data_lr$predictors)[levels(data_lr$predictors)=="situation"] <- "+pos"
levels(data_lr$predictors)[levels(data_lr$predictors)=="dep"] <- "+dep"

data_lr <- data_lr %>% mutate(predictors = factor(predictors, levels=c("word","+pos","+dep","full")), 
                              model=factor(model, levels=c("sum", "LM", "textual", "visual")))
  
for (model in levels(data_lr$model)) {
  data_lr[data_lr$model==model,"score"] = data_lr[data_lr$model==model,"R2"]/data_lr[data_lr$model==model & data_lr$predictors=="+pos","R2"]
  
}
ggplot(data_lr, 
       aes(x=model, color=predictors, y=score)) + 
  #geom_bar(stat="identity", position="dodge") +  
  geom_boxplot() +
  theme(aspect.ratio=2/3, text=element_text(size=25)) +
  ylab('R Squared relative to +pos')
ggsave("../doc/position-new.pdf")


top_words_er <- read.csv("../data/top7_words_er.csv") %>%
  mutate(dep = reorder(dep, omission_v))
ggplot(top_words_er %>% mutate(wordxdep=paste(word,dep)) %>% filter_by_count("wordxdep",5) %>% filter_by_count("dep", 70),
       aes(x=word, y=omission_v, color=dep)) +
  geom_boxplot(position = "dodge") +
  theme(text=element_text(size=25)) +
  ylab("omission score")
ggsave(file="../doc/top_words.pdf", width=12, height=8)

data_coef<- read.table(file="../data/position_coef.txt", header=TRUE)
values <- c("first","second","third","middle","antepenult","penult","last")
data_coef <- data_coef %>% mutate(coef = factor(coef, levels=values, ordered=TRUE),
                    model=factor(model, levels=c("sum","LM","textual","visual")))
ggplot(data_coef, aes(x=as.numeric(coef), y=value, color=model)) + geom_point() + geom_line() +
  xlab("Position") +
  ylab("Coefficient") +
  scale_x_continuous(breaks=1:length(values), 
                     labels=values) +
  theme(aspect.ratio=2/3, text=element_text(size=25))
ggsave("../doc/position-coef.pdf")


