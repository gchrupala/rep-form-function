setwd("/home/gchrupala/repos/naacl2016/mi-stat")
library(dplyr)
library(ggplot2)


data <- data.frame()
for (n in 1:3) {
  for (condition in c("deprel","token")) {
    file <- paste("MI/new_results/",
                  condition,
                  "s",
                  n,
                  ".csv", sep="")
    scores <- read.csv(file)
    cond <- if (condition=="token") {"word"} else {"deprel"}
    data <- rbind(data, data.frame(pathway="Visual",
                           n=n,
                           condition=cond,
                           mi=as.numeric(scores[1,2:1025])))          
    data <- rbind(data, data.frame(pathway="Textual",
                                 n=n,
                                 condition=cond,
                                 mi=as.numeric(scores[2,2:1025])))          
    }
  }

ggplot(data %>% filter(n==1), aes(x=condition, y=mi, color=pathway)) + 
  geom_boxplot()

library(boot)

log_ratio <- function(data, ix) {
  w <- log(median(filter(data[ix,], pathway=="Textual", condition=="word")$mi) / 
        median(filter(data[ix,], pathway=="Visual", condition=="word")$mi),base=exp(1))
  d <- log(median(filter(data[ix,], pathway=="Textual", condition=="deprel")$mi) / 
             median(filter(data[ix,], pathway=="Visual", condition=="deprel")$mi),base=exp(1))
  return(c(w=w,d=d))
}


library(reshape2)

data_boot <- data.frame()
for (i in 1:3) {  
  res <- boot(data %>% filter(n==i), log_ratio, R=5000)
  data_boot <- rbind(data_boot, data.frame(n=i, condition="word", log.ratio=res$t[,1]))
  data_boot <- rbind(data_boot, data.frame(n=i, condition="deprel", log.ratio=res$t[,2]))
}
ggplot(data=data_boot,
       aes(x=as.factor(n), color=condition, y=log.ratio)) +
  geom_boxplot() +
  coord_flip() +
  xlab("n-gram range") +
  ylab(expression(log(MI[C]^T / MI[C]^V ))) +
  theme(text=element_text(size=20)) +
  ggsave(file="bootstrappedMI.png",
         width=6, height=4)

