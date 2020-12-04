#Group Bravo
#ERS Project
#Version: 1.2.4
#Date: 12/10/2019
#Email: Mitch213@mail.chapman.edu
#       Vitta101@mail.chapman.edu
#       Brigg122@mail.chapman.edu

#Setup
rm(list = ls())

dir_input = "\\Users\\Chase\\Desktop\\Arrr\\Project"
setwd(dir_input)

library(leaps)
library(tidyverse)
library(fastDummies)
library(data.table)
library(plyr)
library(xts)
library(tseries)
library(zoo)
library(caret)
library(prodlim) 
library(randomForest)

#Initial Variable Creation
data_raw <- read.csv("452_Bravo_Data.csv")
data_gold <- data.frame(cbind(
  data_raw$Gold..look.ahead.return.continuously.compounded,data_raw[, 10:21]))
data_gold <- data.frame(data_gold[1:237,])#Isolate variable for gold model
data_silver <- data.frame(cbind(
  data_raw$Silver..look.ahead.return.continuously.compounded.,data_raw[, 10:21]))
data_silver <- data.frame(data_silver[1:237,])#Isolate variable for silver model
data_platinum <- data.frame(cbind(
  data_raw$Platinum..look.ahead.return.continuously.compounded.,data_raw[, 10:21]))
data_platinum <- data.frame(data_platinum[1:237,])#Isolate variable for platinum model
data_palladium <- data.frame(cbind(
  data_raw$Palladium..look.ahead.return.continuously.compounded.,data_raw[, 10:21]))
data_palladium <- data.frame(data_palladium[1:237,])#Isolate variable for palladium model

#Time Series Exploration
##building time series
data_gold_matrix <- data.matrix(
  data_gold$data_raw.Gold..look.ahead.return.continuously.compounded)
data_silver_matrix <- data.matrix(
  data_silver$data_raw.Silver..look.ahead.return.continuously.compounded.)
data_platinum_matrix <- data.matrix(
  data_platinum$data_raw.Platinum..look.ahead.return.continuously.compounded.)
data_palladium_matrix <- data.matrix(
  data_palladium$data_raw.Palladium..look.ahead.return.continuously.compounded.)

data_gold_ts <- ts(data_gold_matrix,
                   start = c(2000, 1), end = c(2019, 9), frequency = 12)
data_silver_ts <- ts(data_silver_matrix,
                     start = c(2000, 1), end = c(2019, 9), frequency = 12)
data_platinum_ts <- ts(data_platinum_matrix,
                       start = c(2000, 1), end = c(2019, 9), frequency = 12)
data_palladium_ts <- ts(data_palladium_matrix,
                        start = c(2000, 1), end = c(2019, 9), frequency = 12)

##visualize inital time series
plot(data_gold_ts, main = "Gold")
plot(data_silver_ts, main = "Silver")  
plot(data_platinum_ts, main = "Platinum")  
plot(data_palladium_ts, main = "Palladium")  

##check for stationarity in y variables with ADF tests
adf.test(data_gold_ts, k = 12)
adf.test(data_silver_ts, k = 12)
adf.test(data_platinum_ts, k = 12)
adf.test(data_palladium_ts, k = 12)

##build time series and plot to check for stationarity in x variables 
for (x in 2:13){
  loop_ts_x <- ts(data_gold[, x], start = c(2000, 1), end = c(2019, 9), frequency = 12)
  plot(loop_ts_x, main = colnames(data_gold)[x])
}

##check for stationarity in x variables with ADF tests
for (x in 2:13){
  loop_ts_x_adf <- ts(data_gold[, x])
  print(colnames(data_gold)[x])
  print(adf.test(loop_ts_x_adf, k = 12))
}

#Time Series Differencing
##difference all x variables to fix non-stationarity
x_vars_diff_list <- list()
for (x in 2:13){
  loop_ts <- ts(data_gold[, x])
  loop_ts_diff_x <- diff(loop_ts)
  x_vars_diff_list[[x - 1]] <- loop_ts_diff_x
}

##difference all y variables to maintain foward looking model
data_gold_ts <- diff(as.numeric(data_gold_ts))
data_silver_ts <- diff(as.numeric(data_silver_ts))
data_platinum_ts <- diff(as.numeric(data_platinum_ts))
data_palladium_ts <- diff(as.numeric(data_palladium_ts))

##check for stationarity in differenced x variables with plots
for (x in 1:12){
  loop_frame_x <- data.frame(x_vars_diff_list[x])
  loop_ts_x <- ts(loop_frame_x, start = c(2000, 1),
                  end = c(2019, 9), frequency = 12)
  plot(loop_ts_x, main = paste("Differenced", colnames(data_gold)[x + 1]))
}

##check for stationarity in differenced x variables with ADF tests
for (x in 1:12){
  loop_frame_x <- data.frame(x_vars_diff_list[x])
  loop_ts_x <- ts(loop_frame_x)
  print(colnames(data_gold)[x + 1])
  print(adf.test(loop_ts_x), k = 12)
}

#Regression Variable Differencing
##replace precious metal values with differenced y's and diffrenced x's and remove a row
data_gold <- data_gold[-237,]
data_gold$data_raw.Gold..look.ahead.return.continuously.compounded <- data_gold_ts
data_silver <- data_silver[-237,]
data_silver$data_raw.Silver..look.ahead.return.continuously.compounded. <- data_silver_ts
data_platinum <- data_platinum[-237,]
data_platinum$data_raw.Platinum..look.ahead.return.continuously.compounded. <- data_platinum_ts
data_palladium <- data_palladium[-237,]
data_palladium$data_raw.Palladium..look.ahead.return.continuously.compounded. <- data_palladium_ts

for (x in 2:13){
  data_gold[, x] <- x_vars_diff_list[x - 1]
  data_silver[, x] <- x_vars_diff_list[x - 1]
  data_platinum[, x] <- x_vars_diff_list[x - 1]
  data_palladium[, x] <- x_vars_diff_list[x - 1]
}

#Multivariate Linear Regressions Construction
par(mfrow = c(2, 2))
##find optimal number of variables to use in multivariable linear regression for each metal
reg_all_gold = regsubsets(
  data_gold$data_raw.Gold..look.ahead.return.continuously.compounded ~ .,
  data = data_gold, method = "forward", nvmax = 20)
reg_all_gold.summary <- summary(reg_all_gold, statistic = "adjr2")
plot(reg_all_gold.summary$adjr2,
     xlab = "Number of variables",
     ylab = "RSquare",
     main = "Gold Adj.RSQ",type = "l")

reg_all_silver = regsubsets(
  data_silver$data_raw.Silver..look.ahead.return.continuously.compounded. ~ .,
  data = data_silver, method = "forward", nvmax = 20)
reg_all_silver.summary <- summary(reg_all_silver, statistic = "adjr2")
plot(reg_all_silver.summary$adjr2,
     xlab = "Number of variables",
     ylab = "RSquare",
     main = "Silver Adj.RSQ",type = "l")

reg_all_platinum = regsubsets(
  data_platinum$data_raw.Platinum..look.ahead.return.continuously.compounded. ~ .,
  data = data_platinum, method = "forward", nvmax = 20)
reg_all_platinum.summary <- summary(reg_all_platinum, statistic = "adjr2")
plot(reg_all_platinum.summary$adjr2,
     xlab = "Number of variables",
     ylab = "RSquare",
     main = "Platinum Adj.RSQ", type = "l")

reg_all_palladium = regsubsets(
  data_palladium$data_raw.Palladium..look.ahead.return.continuously.compounded. ~ .,
  data = data_palladium, method = "forward", nvmax = 20)
reg_all_palladium.summary <- summary(reg_all_palladium, statistic = "adjr2")
plot(reg_all_palladium.summary$adjr2,
     xlab = "Number of variables",
     ylab = "RSquare",
     main = "Palladium Adj.RSQ",type = "l")

##build multivariable linear regressions using optimal number of variables 
##gold
selected_variables <- 8
coef(reg_all_gold, scale = "adjr2", selected_variables)
summary(reg_all_gold)$adjr2[selected_variables]

gold_model <- lm(data_gold$data_raw.Gold..look.ahead.return.continuously.compounded ~
                   Oil.Prices.Real.dollars. +
                   S.P.500.index.dollars. +
                   Nasdaq.dollars. + VIX +
                   X10.Yr.Spread.Over.Fed.Funds +
                   CPI +US.dollar.Index +
                   Real.GDP.Dollars., data = data_gold)
summary(gold_model)
##silver
selected_variables <- 7
coef(reg_all_silver, scale="adjr2",selected_variables)
summary(reg_all_silver)$adjr2[selected_variables]

silver_model <- lm(data_silver$data_raw.Silver..look.ahead.return.continuously.compounded. ~
                     Oil.Prices.Real.dollars. +
                     S.P.500.index.dollars. +
                     Nasdaq.dollars. + VIX +
                     X10.Yr.Spread.Over.Fed.Funds +
                     CPI + US.dollar.Index, data = data_silver)
summary(silver_model)
##platinum
selected_variables <- 9
coef(reg_all_platinum, scale="adjr2",selected_variables)
summary(reg_all_platinum)$adjr2[selected_variables]

platinum_model <- lm(data_platinum$data_raw.Platinum..look.ahead.return.continuously.compounded. ~
                       Unemployment.Rate.US +
                       Oil.Prices.Real.dollars. +
                       Michigan.Consumer.Sentiment.Index +
                       S.P.500.index.dollars. +
                       Nasdaq.dollars. + VIX +
                       Euro.Stoxx.50..US.Dollars. +
                       CPI + US.dollar.Index, data = data_platinum)
summary(platinum_model)
##palladium
selected_variables <- 6
coef(reg_all_palladium, scale="adjr2",selected_variables)
summary(reg_all_palladium)$adjr2[selected_variables]

palladium_model <- lm(data_palladium$data_raw.Palladium..look.ahead.return.continuously.compounded. ~
                        S.P.500.index.dollars. +
                        Nasdaq.dollars. + VIX +
                        Euro.Stoxx.50..US.Dollars. +
                        CPI + US.dollar.Index, data = data_palladium)
summary(palladium_model)

#K-Fold Cross Validation On Linear Models
##define train control for k fold cross validation (4)
train_control <- trainControl(method = "cv", number = 4)
##gold
k_fold_gold <- train(data_raw.Gold..look.ahead.return.continuously.compounded ~
                       Oil.Prices.Real.dollars. +
                       S.P.500.index.dollars. +
                       Nasdaq.dollars. + VIX +
                       X10.Yr.Spread.Over.Fed.Funds +
                       CPI +US.dollar.Index + Real.GDP.Dollars.,
                     data=data_gold, trControl = train_control, method = "lm")
print(k_fold_gold)
summary(k_fold_gold)
##silver
k_fold_silver <- train(data_raw.Silver..look.ahead.return.continuously.compounded. ~
                         Oil.Prices.Real.dollars. +
                         S.P.500.index.dollars. +
                         Nasdaq.dollars. + VIX +
                         X10.Yr.Spread.Over.Fed.Funds +
                         CPI + US.dollar.Index,
                       data=data_silver, trControl = train_control, method = "lm")
print(k_fold_silver)
summary(k_fold_silver)
#platinum
k_fold_platinum <- train(data_raw.Platinum..look.ahead.return.continuously.compounded. ~
                           Unemployment.Rate.US +
                           Oil.Prices.Real.dollars. +
                           Michigan.Consumer.Sentiment.Index +
                           S.P.500.index.dollars. +
                           Nasdaq.dollars. + VIX +
                           Euro.Stoxx.50..US.Dollars. +
                           CPI + US.dollar.Index,
                         data=data_platinum, trControl = train_control, method = "lm")
print(k_fold_platinum)
summary(k_fold_platinum)
#palladium
k_fold_palladium <- train(data_raw.Palladium..look.ahead.return.continuously.compounded. ~
                            S.P.500.index.dollars. +
                            Nasdaq.dollars. + VIX +
                            Euro.Stoxx.50..US.Dollars. +
                            CPI + US.dollar.Index,
                          data=data_palladium, trControl = train_control, method = "lm")
print(k_fold_palladium)
summary(k_fold_palladium)

#Random Forest Construction/Analysis A
##building models with RandomForest to compare to our linear models
##using all x-variables for each y

##gold
gold_fit <- randomForest(data_raw.Gold..look.ahead.return.continuously.compounded ~ .,
                         data = data_gold)
gold_fit_pred <- predict(gold_fit, data_gold)
##confirm RMSE from OOB
sqrt(abs(sum(gold_fit_pred - 
               data_gold$data_raw.Gold..look.ahead.return.continuously.compounded ^ 2)))
print(gold_fit)
importance(gold_fit)#importance of each predictor

##silver
silver_fit <- randomForest(data_raw.Silver..look.ahead.return.continuously.compounded. ~ .,
                           data = data_silver)
silver_fit_pred <- predict(silver_fit, data_silver)
##confirm RMSE from OOB
sqrt(abs(sum(silver_fit_pred - 
               data_silver$data_raw.Silver..look.ahead.return.continuously.compounded. ^ 2)))
print(silver_fit)
importance(silver_fit)#importance of each predictor

##platinum
platinum_fit <- randomForest(data_raw.Platinum..look.ahead.return.continuously.compounded. ~ .,
                             data = data_platinum)
platinum_fit_pred <- predict(platinum_fit, data_platinum)
##confirm RMSE from OOB
sqrt(abs(sum(platinum_fit_pred - 
               data_platinum$data_raw.Platinum..look.ahead.return.continuously.compounded. ^ 2)))
print(platinum_fit)
importance(platinum_fit)#importance of each predictor

##palladium
palladium_fit <- randomForest(data_raw.Palladium..look.ahead.return.continuously.compounded. ~ .,
                              data = data_palladium)
palladium_fit_pred <- predict(palladium_fit, data_palladium)
##confirm RMSE from OOB
sqrt(abs(sum(palladium_fit_pred - 
               data_palladium$data_raw.Palladium..look.ahead.return.continuously.compounded. ^ 2)))
print(palladium_fit)
importance(palladium_fit)#importance of each predictor

#Random Forest Construction/Analysis B
##building models with RandomForest to compare to our linear models
##using only our previously selected regression x-variables
##for each y-variable

##gold
gold_fit <- randomForest(data_raw.Gold..look.ahead.return.continuously.compounded ~
                           Oil.Prices.Real.dollars. +
                           S.P.500.index.dollars. +
                           Nasdaq.dollars. + VIX +
                           X10.Yr.Spread.Over.Fed.Funds +
                           CPI +US.dollar.Index +
                           Real.GDP.Dollars., data = data_gold)
gold_fit_pred <- predict(gold_fit, data_gold)
##confirm RMSE from OOB
sqrt(abs(sum(gold_fit_pred - 
               data_gold$data_raw.Gold..look.ahead.return.continuously.compounded ^ 2)))
print(gold_fit)
importance(gold_fit)#importance of each predictor

##silver
silver_fit <- randomForest(data_raw.Silver..look.ahead.return.continuously.compounded. ~
                             Oil.Prices.Real.dollars. +
                             S.P.500.index.dollars. +
                             Nasdaq.dollars. + VIX +
                             X10.Yr.Spread.Over.Fed.Funds +
                             CPI +US.dollar.Index, data = data_silver)
silver_fit_pred <- predict(silver_fit, data_silver)
##confirm RMSE from OOB
sqrt(abs(sum(silver_fit_pred - 
               data_silver$data_raw.Silver..look.ahead.return.continuously.compounded. ^ 2)))
print(silver_fit)
importance(silver_fit)#importance of each predictor

##platinum
platinum_fit <- randomForest(data_raw.Platinum..look.ahead.return.continuously.compounded. ~
                               Unemployment.Rate.US +
                               Oil.Prices.Real.dollars. +
                               Michigan.Consumer.Sentiment.Index +
                               S.P.500.index.dollars. +
                               Nasdaq.dollars. + VIX +
                               Euro.Stoxx.50..US.Dollars. +
                               CPI + US.dollar.Index, data = data_platinum)
platinum_fit_pred <- predict(platinum_fit, data_platinum)
##confirm RMSE from OOB
sqrt(abs(sum(platinum_fit_pred - 
               data_platinum$data_raw.Platinum..look.ahead.return.continuously.compounded. ^ 2)))
print(platinum_fit)
importance(platinum_fit)#importance of each predictor

#palladium
palladium_fit <- randomForest(data_raw.Palladium..look.ahead.return.continuously.compounded. ~
                                S.P.500.index.dollars. +
                                Nasdaq.dollars. + VIX +
                                Euro.Stoxx.50..US.Dollars. +
                                CPI + US.dollar.Index, data = data_palladium)
palladium_fit_pred <- predict(palladium_fit, data_palladium)
##confirm RMSE from OOB
sqrt(abs(sum(palladium_fit_pred - 
               data_palladium$data_raw.Palladium..look.ahead.return.continuously.compounded ^ 2)))
print(palladium_fit)
importance(palladium_fit)#importance of each predictor

