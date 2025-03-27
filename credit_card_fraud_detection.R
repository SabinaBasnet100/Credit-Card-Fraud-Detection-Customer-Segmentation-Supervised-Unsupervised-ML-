############# Assignment 2: High Risk Location Analysis #############
library(ggplot2)
library(dplyr)
library(caret)

# Load the data
data <- read.csv("cards_data.csv")

# Simulate 'current_balance' for demonstration
set.seed(123)
data$current_balance <- runif(nrow(data), min = 0, max = as.numeric(gsub("[\$,]", "", data$credit_limit)))

# Calculate utilization rate
data$utilization_rate <- data$current_balance / as.numeric(gsub("[\$,]", "", data$credit_limit))
data$credit_limit <- as.numeric(gsub("[\$,]", "", data$credit_limit))
data$high_utilization <- as.factor(ifelse(data$utilization_rate > 0.75, "Yes", "No"))
data <- na.omit(data)

# Create model matrix and response
x <- model.matrix(~ has_chip + credit_limit + high_utilization - 1, data = data)
y <- as.numeric(data$high_utilization == "Yes")

# Fit logistic regression model
fit <- glm(high_utilization ~ has_chip + credit_limit, data = data, family = binomial())
print(summary(fit))

# Predictions and accuracy
predicted_probabilities <- predict(fit, type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, "Yes", "No")
accuracy <- mean(predicted_classes == data$high_utilization)
print(paste("Model Accuracy for Predicting High Credit Utilization: ", accuracy))

# Cross-validation with caret
control <- trainControl(method = "cv", number = 10)
model <- train(high_utilization ~ has_chip + credit_limit, data = data, method = "glm",
               family = binomial(), trControl = control)
print(model)

############# Business Problem 3: Credit Card Delinquency #############
library(randomForest)
library(pROC)

data <- read.csv("cards_data.csv")
data$credit_limit <- as.numeric(gsub("[\$,]", "", data$credit_limit))
set.seed(123)
data$utilization <- runif(nrow(data), 0, 1) * data$credit_limit
data$delinquency_status <- as.factor(ifelse(data$utilization > 0.8 * data$credit_limit, "Yes", "No"))

training_indices <- createDataPartition(data$delinquency_status, p = 0.8, list = TRUE)
train_data <- data[training_indices[[1]], ]
test_data <- data[-training_indices[[1]], ]

logistic_model <- glm(delinquency_status ~ credit_limit + utilization, family = binomial(), data = train_data)
summary(logistic_model)

predictions <- predict(logistic_model, test_data, type = "response")
predicted_classes <- ifelse(predictions > 0.5, "Yes", "No")

conf_matrix <- table(Predicted = predicted_classes, Actual = test_data$delinquency_status)
print(conf_matrix)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy of the model:", accuracy))

roc_obj <- roc(test_data$delinquency_status, predictions, levels = c("No", "Yes"))
plot(roc_obj, main = "ROC Curve", col = "#1c61b6")
abline(0, 1, lty = 2, col = "red")

############# Assignment 3: Fraud Detection & Clustering #############
library(solitude)
library(cluster)
library(ROSE)

cards_data <- read.csv("cards_data.csv")
cards_data$card_on_dark_web <- as.factor(cards_data$card_on_dark_web == "Yes")
cards_data$has_chip <- as.factor(cards_data$has_chip)
cards_data$credit_limit <- as.numeric(gsub("[\$,]", "", cards_data$credit_limit))
cards_data <- na.omit(cards_data)

clustering_data <- cards_data %>% 
  select(credit_limit, num_cards_issued) %>% 
  mutate(across(everything(), ~ scale(.)))

# Supervised: Logistic Regression
set.seed(123)
trainIndex <- createDataPartition(cards_data$card_on_dark_web, p = 0.7, list = FALSE)
train_data <- cards_data[trainIndex, ]
test_data <- cards_data[-trainIndex, ]

logit_model <- glm(card_on_dark_web ~ has_chip + credit_limit + num_cards_issued, 
                   data = train_data, family = binomial())
summary(logit_model)

logit_pred <- predict(logit_model, newdata = test_data, type = "response")
logit_pred_class <- ifelse(logit_pred > 0.5, 1, 0)
confusionMatrix(as.factor(logit_pred_class), as.factor(test_data$card_on_dark_web))

# Unsupervised: K-Means Clustering
wss <- sapply(1:10, function(k) {
  kmeans(clustering_data, centers = k, nstart = 10)$tot.withinss
})
plot(1:10, wss, type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of Clusters (k)", ylab = "Total Within-Cluster Sum of Squares")

set.seed(123)
kmeans_model <- kmeans(clustering_data, centers = 3)
cards_data$cluster <- kmeans_model$cluster

ggplot(cards_data, aes(x = credit_limit, y = num_cards_issued, color = as.factor(cluster))) +
  geom_point() +
  ggtitle("Customer Segments") +
  xlab("Credit Limit") +
  ylab("Number of Cards Issued")

silhouette_score <- silhouette(kmeans_model$cluster, dist(clustering_data))
mean(silhouette_score[, 3])

# Mixed: Random Forest + Isolation Forest
rf_model <- randomForest(card_on_dark_web ~ has_chip + credit_limit + num_cards_issued, 
                         data = train_data, ntree = 500)
rf_pred <- predict(rf_model, newdata = test_data)

iso_model <- isolationForest(data = train_data %>% select(credit_limit, num_cards_issued))
iso_scores <- predict(iso_model, test_data %>% select(credit_limit, num_cards_issued))

test_data$fraud_risk <- ifelse(rf_pred == 1 | iso_scores$outlier == 1, "High Risk", "Low Risk")
table(test_data$fraud_risk)

balanced_data <- ovun.sample(card_on_dark_web ~ has_chip + credit_limit + num_cards_issued, 
                             data = train_data, method = "over", N = 2000)$data

fit_balanced <- glm(card_on_dark_web ~ has_chip + credit_limit + num_cards_issued, 
                    data = balanced_data, family = binomial())
summary(fit_balanced)

predicted_balanced <- predict(fit_balanced, newdata = test_data, type = "response")
predicted_balanced_class <- ifelse(predicted_balanced > 0.5, 1, 0)
confusionMatrix(as.factor(predicted_balanced_class), as.factor(test_data$card_on_dark_web))
