library(dplyr)
library(ggplot2)
library(reshape2)
library(boot)

data <- read.csv("../data/mutual.csv")
ggplot(data %>% filter(pathway != "sum"), aes(x=pathway, y=mi, color=paste(condition, order))) + geom_boxplot() +
  xlab(NULL) +
  ylab("Mutual Information") +
  theme(legend.title=element_blank()) +
  theme(text=element_text(size=20)) 
  ggsave("../doc/raw_mutual.pdf",width=6, height=4)

# Textual vs Visual

log_ratio <- function(data, ix) {
  w <- log(median(filter(data[ix,], pathway=="textual", condition=="word")$mi) / 
        median(filter(data[ix,], pathway=="visual", condition=="word")$mi),base=exp(1))
  d <- log(median(filter(data[ix,], pathway=="textual", condition=="dep")$mi) / 
             median(filter(data[ix,], pathway=="visual", condition=="dep")$mi),base=exp(1))
  return(c(w=w,d=d))
}


data_boot <- data.frame()
for (i in 1:3) {  
  res <- boot(data %>% filter(order==i), log_ratio, R=5000)
  data_boot <- rbind(data_boot, data.frame(order=i, condition="word", log.ratio=res$t[,1]))
  data_boot <- rbind(data_boot, data.frame(order=i, condition="dep", log.ratio=res$t[,2]))
}
ggplot(data=data_boot,
       aes(x=as.factor(order), color=condition, y=log.ratio)) +
  geom_boxplot(notch = TRUE) +
  coord_flip() +
  xlab("n-gram range") +
  ylab(expression(log(MI[C]^T / MI[C]^V ))) +
  theme(text=element_text(size=20)) +
  theme(legend.title=element_blank()) +
  ggsave(file="../doc/bootstrappedMI.pdf", width=6, height=4)


# LM / Textual
log_ratio_2 <- function(data, ix) {
  w <- log(median(filter(data[ix,], pathway=="lm", condition=="word")$mi) / 
             median(filter(data[ix,], pathway=="textual", condition=="word")$mi),base=exp(1))
  d <- log(median(filter(data[ix,], pathway=="lm", condition=="dep")$mi) / 
             median(filter(data[ix,], pathway=="textual", condition=="dep")$mi),base=exp(1))
  return(c(w=w,d=d))
}

data_boot_2 <- data.frame()
for (i in 1:3) {  
  res <- boot(data %>% filter(order==i), log_ratio_2, R=5000)
  data_boot_2 <- rbind(data_boot_2, data.frame(order=i, condition="word", log.ratio=res$t[,1]))
  data_boot_2 <- rbind(data_boot_2, data.frame(order=i, condition="dep", log.ratio=res$t[,2]))
}
ggplot(data=data_boot_2,
       aes(x=as.factor(order), color=condition, y=log.ratio)) +
  geom_boxplot(notch = TRUE) +
  coord_flip() +
  xlab("n-gram range") +
  ylab(expression(log(MI[C]^LM / MI[C]^T ))) +
  theme(text=element_text(size=20)) +
  theme(legend.title=element_blank()) +
  ggsave(file="../doc/bootstrappedMI2.pdf", width=6, height=4)

data %>% filter(pathway=="textual", condition=="dep") %>% arrange(desc(mi)) %>% head
