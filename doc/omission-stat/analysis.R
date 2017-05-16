# INSTALLATION

#install.packages("Rcpp", dependencies=TRUE, repos="http://cran.r-project.org" )
#install.packages("dplyr", dependencies=TRUE, repos="http://cran.r-project.org" )
#install.packages("ggplot2", dependencies=TRUE, repos="http://cran.r-project.org")

library("dplyr")
library("ggplot2")
# READING DATA

read.data <- function(f) {
  data <- read.csv(f, col.names= c('omit','id','word','pos','deprel', 'score')) %>%
  select(-omit) %>% na.omit()
  return(data[! data[,'pos'] %in%c('<END>','END'),])
}
data_v <- read.data('../data/visual_full_omission.csv')
data_v$pathway <- "visual"
data_v$deprel <- reorder(data_v$deprel, data_v$score, median)
data_v$pos <- reorder(data_v$pos, data_v$score, median)
data_t <- read.data('../data/textual_full_omission.csv')
data_t$pathway <- "textual"
data_t$deprel <- reorder(data_t$deprel, data_v$score, median)
data_t$pos <- reorder(data_t$pos, data_v$score, median)

data <- rbind(data_t , data_v) %>% mutate(pathway=as.factor(pathway))

filter_by_count <- function(dat, col, count) {
  dat[dat[,col] %in% names(table(dat[,col]))[table(dat[,col]) > count],]
}

MINCOUNT<-500
subset <- filter_by_count(data %>% filter(pathway=='visual'),"pos", MINCOUNT)



# OMISSION BOXPLOTS

ggplot(subset, aes(x=pos, y=score)) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15))
ggsave(file="visual-omission-pos-boxplot.png",width=5, height=5)

subset <- filter_by_count(data %>% filter(pathway=='visual'),"deprel",MINCOUNT)
ggplot(subset, aes(x=deprel, y=score)) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15))
ggsave(file="visual-omission-deprel-boxplot.png",width=5, height=5)

subset <- filter_by_count(data %>% filter(pathway=='textual'),"pos",MINCOUNT)
ggplot(subset, aes(x=pos, y=score)) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15))
ggsave(file="textual-omission-pos-boxplot.png",
       width=5, height=5)

subset <- filter_by_count(data %>% filter(pathway=='textual'),"deprel",MINCOUNT)
ggplot(subset, aes(x=deprel, y=score)) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15))
ggsave(file="textual-omission-deprel-boxplot.png",
       width=5, height=5)

subset <- filter_by_count(data,"pos",MINCOUNT*2)
ggplot(subset, aes(x=pos, y=score, color=pathway)) +
  geom_boxplot(outlier.size = 1, outlier.colour='grey', varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15))
ggsave(file="imaginet-omission-pos-boxplot.png",
       width=5, height=5)

subset <- filter_by_count(data,"deprel",MINCOUNT*2)
ggplot(subset, aes(x=deprel, y=score, color=pathway)) +
  geom_boxplot(outlier.size = 1, outlier.colour='grey', varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15))
ggsave(file="imaginet-omission-deprel-boxplot.png",
       width=5, height=5)

### Log scores ratios per dependency
ratio <- filter(data, pathway=='visual')$score / filter(data, pathway=='textual')$score
data_r <- filter(data, pathway=='visual')
data_r$ratio <- ratio
data_r$deprel <- reorder(data_r$deprel, data_r$ratio, median)
data_r$pos <- reorder(data_r$pos, data_r$ratio, median)

ggplot(filter_by_count(data_r,"pos",MINCOUNT) %>% filter,
       aes(x=pos, y=log(ratio))) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15))
ggsave(file="omission-pos-boxplot.png", width=5, height=5)
ggplot(filter_by_count(data_r,"deprel",MINCOUNT), aes(x=deprel, y=log(ratio))) +
  geom_boxplot(outlier.colour = 'grey', outlier.size = 1, varwidth = FALSE) +
  coord_flip() +
  theme(text=element_text(size=15))
ggsave(file="omission-deprel-boxplot.png", width=5, height=5)
m1 <- lm(log(ratio) ~ pos, data=data_r)
m2 <- lm(log(ratio) ~ deprel, data=data_r)
m3 <- lm(log(ratio) ~ deprel+pos, data=data_r)

summary(m1)
summary(m2)
summary(m3)



# POSITION SCATTERPLOTS

data <- read.csv("visual_omission_with_position.csv")
data <- data %>% mutate(pos=reorder(pos, score),
                        deprel=reorder(deprel, score)) #%>% filter(pos %in% c("NN","NNS"))
cor(data %>% select(position, score), method = "pearson")
cor(data  %>% filter(pos %in% c("NN","NNP","NNS")) %>% select(position, score), method = "pearson")

ggplot(data %>% filter_by_count("pos",MINCOUNT*2),
            aes(x=position, color=pos, y=score)) +
  geom_point(position = "jitter", alpha=0.03) +
  geom_smooth() +
  xlim(0,20) +
  ylim(0.0, 0.5) +
  theme(aspect.ratio=2/3, text=element_text(size=25)) +
  ggtitle("Visual")
ggsave(file="visual-omission-by-position.png")

data <- read.csv("textual_omission_with_position.csv")
data <- data %>% mutate(pos=reorder(pos, score),
                        deprel=reorder(deprel, score))
cor(data  %>% select(position, score), method = "pearson")

ggplot(data %>% filter_by_count("pos",MINCOUNT*2),
            aes(x=position, color=pos, y=score)) +
  geom_point(position = "jitter", alpha=0.03) +
  geom_smooth() +
  xlim(0,20) +
  ylim(0.0, 0.5) +
  theme(aspect.ratio=2/3, text=element_text(size=25)) +
  ggtitle("Textual")
ggsave(file="textual-omission-by-position.png")


data <- read.csv("bi_visual_omission_with_position.csv")
data <- data %>% mutate(pos=reorder(pos, score),
                        deprel=reorder(deprel, score))
ggplot(data %>% filter_by_count("pos",MINCOUNT*2),
            aes(x=position, color=pos, y=score)) +
  geom_point(position = "jitter", alpha=0.03) +
  geom_smooth() +
  xlim(0,20) +
  ylim(0.0, 0.5) +
  theme(aspect.ratio=2/3, text=element_text(size=25)) +
  ggtitle("Bidirectional Visual")
ggsave(file="bi-visual-omission-by-position.png")

data <- read.csv("sum_visual_omission_with_position.csv")
data <- data %>% mutate(pos=reorder(pos, score),
                        deprel=reorder(deprel, score))
ggplot(data %>% filter_by_count("pos",MINCOUNT*2),
            aes(x=position, color=pos, y=score)) +
  geom_point(position = "jitter", alpha=0.03) +
  geom_smooth() +
  xlim(0,20) +
  ylim(0.0, 0.5) +
  theme(aspect.ratio=2/3, text=element_text(size=25)) +
  ggtitle("Sum Visual")
ggsave(file="bi-visual-omission-by-position.png")


# TOP WORDS

top_words <- read.csv("top_words.csv") %>%
  mutate(deprel = reorder(deprel, score))
ggplot(top_words %>% filter_by_count("deprel",70),
       aes(x=word, y=score, color=deprel)) +
  geom_boxplot(position = "dodge") +
   theme(text=element_text(size=25))
ggsave(file="top_words.pdf")
