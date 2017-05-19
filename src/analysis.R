library(dplyr)
library(ggplot2)

dep <- read.csv("../data/depparse_coco_val.csv", header=TRUE)
omi <- read.csv("../data/omission_coco_val.csv", header=TRUE)

omi_v <- omi %>% select(-omission_t) %>% rename(score=omission_v) %>% mutate(pathway="visual")
omi_t <- omi %>% select(-omission_v) %>% rename(score=omission_t) %>% mutate(pathway="textual")

data_v <- merge(dep, omi_v) 
data_t <- merge(dep, omi_t) 
dep_lev <- (data_v %>% group_by(dep) %>% summarize(median_score=median(score)) %>% 
              data.frame %>% arrange(median_score))$dep 


data <- rbind(data_t, data_v) %>% mutate(dep = factor(dep, levels=dep_lev))

filter_by_count <- function(dat, col, count) {
  dat[dat[,col] %in% names(table(dat[,col]))[table(dat[,col]) > count],]
}
MINCOUNT<-500

ggplot(filter_by_count(data, "dep", MINCOUNT)) +
       aes(x=dep, y=score, color=pathway) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15))


data_v %>% filter(dep=="attr")
