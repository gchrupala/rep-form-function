library(dplyr)
library(ggplot2)
library(lme4)

# Visualization of predicting omission scores using various predictors
data <- read.table(file="predictor_rsq.txt", header=TRUE) 
for (model in levels(data$model)) {
  data[data$model==model,"score"] = data[data$model==model,"rsq"]/data[data$model==model & data$predictors=="word","rsq"]
}
data <- data %>% mutate(predictors = reorder(predictors, rsq))
ggplot(data, aes(x=model, color=predictors, y=score)) + geom_boxplot() +  theme(aspect.ratio=2/3, text=element_text(size=25)) +
   ylab("R Squared relative to word")
ggsave("position-new.png")

data_coef<- read.table(file="position_coef.txt", header=TRUE)
values <- c("first","second","third","middle","antepenult","penult","last")
data_coef$predictor <- factor(data_coef$predictor, levels=values, ordered=TRUE)
ggplot(data_coef, aes(x=as.numeric(predictor), y=coef, color=model)) + geom_point() + geom_line() +
  xlab("Position") +
  ylab("Coefficient") +
  scale_x_continuous(breaks=1:length(values), 
                     labels=values) +
  theme(aspect.ratio=2/3, text=element_text(size=25))
ggsave("position-coef.png")

