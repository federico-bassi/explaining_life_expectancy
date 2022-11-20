####### EXPLAINING LIFE EXPECTANCY #######
# This script contains the code used to produce the analysis and plots for project
# "Explaining life expectancy" made for the Statistical Learning curse.

source("R/utils.R")
source("R/packages.R")

# Data pre-processing #----------------------------------------------------------
dataset <- read.csv("data/who_life_exp.csv", header = TRUE)

# Filter the data: we want to retain only data related to 2014.
dataset <- dataset %>% filter(year == 2016) %>% select(-c("region", "year", "life_exp60"))
dataset %>% glimpse()
dataset %>% skimr::skim()
dataset <- dataset %>% select(-c("doctors", "hospitals", "gni_capita", "une_hiv", "une_poverty", "une_life","une_edu_spend","une_literacy", "une_school", "infant_mort", "une_infant"))
dataset %>% skimr::skim()
dataset <-  dataset[complete.cases(dataset), ]
dataset %>% glimpse()

# Create a dataset to perform PCA (see below)
df_diseases <- dataset%>%select(c("country", "measles", "hepatitis", "polio", "diphtheria"))
#rownames(df_diseases) <- df_diseases$country
df_diseases$country <- NULL

# Create a dataset to perform K-means clustering (see below)
dataset_kmeans <- dataset %>% select(-c("une_pop"))
rownames(dataset_kmeans) <- dataset_kmeans$country_code
dataset_kmeans$country <- NULL
dataset_kmeans$country_code <- NULL
dataset_kmeans$life_expect <- NULL

dataset <- dataset %>% select(-c("country", "country_code"))
dataset %>% skimr::skim()

# Train/test split #------------------------------------------------------------
set.seed(123)
splits <- initial_split(dataset, prop = .8)
folds<-vfold_cv(training(splits))

# Recipe specification #--------------------------------------------------------
recipe_specification <- recipe(life_expect ~ ., data = training(splits))
recipe_specification %>% prep() %>% juice() %>% glimpse()


# Linear regression Diagnosic #-----------------------------------------------------------

# Build a linear regression model in order to perform the diagnostics
lin_reg_model <- lm(life_expect~., data = dataset)
summary(lin_reg_model)
lin_reg_model
plot(lin_reg_model)

##* Distributions --------------------------------------------------------------
# Plot the distribution of each variable in the dataset
meltData <- melt(dataset)
theme_update(plot.title = element_text(hjust = 0.5))
p <- ggplot(meltData, aes(factor(variable), value)) + geom_boxplot() + 
  facet_wrap(~variable, scale="free")+xlab("Variables")+ylab("Distribution")+
  labs(title="Distribution of each variable")
p

##* Breusch-Pagan Test to test homoskedasticity---------------------
# If the p-value of the test is lower than 0.05, reject the null of homoskedasticity
bptest(lin_reg_model)

##* Q-Q plot, Shapiro-Wilk test to test normality of the residuals--------------------------
# If the p-value is lower than 0.05, reject the null that the data were sampled from
# a normal distribution.
qqnorm(resid(lin_reg_model), main = "Normal Q-Q Plot Linear Regression Model", col = "darkgrey")
qqline(resid(lin_reg_model), col = "dodgerblue", lwd = 2)

shapiro.test(resid(lin_reg_model))

##* High leverage points -------------------------------------------------------
hats <- as.data.frame(hatvalues(lin_reg_model))
plot(hatvalues(lin_reg_model), type = 'h', ylab="Hat values of the linear regression", main="Leverage")

nrow(dataset)
which(hats > (3*16/nrow(dataset)))


##* Collinearity ---------------------------------------------------------------
cor_matrix <- cor(dataset%>%select(-life_expect))
round(cor_matrix, 2)

# Correlation plot
par(mar=c(10,10,10,10))
corrplot(cor_matrix, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45, main= "Correlation plot", line =-1)

#VIF
vif_values <- vif(lin_reg_model)
par(mar=c(4,8,4,4))
barplot(vif_values,main = "VIF Values", horiz = TRUE, col = "steelblue", las=2)
abline(v = 5, lwd = 3, lty = 2)


# Linear Regression #-----------------------------------------------------------
##* Engine ---------------------------------------------------------------------
model_specification_linear_regression <- linear_reg() %>%
  set_engine("lm")

##* Workflow -------------------------------------------------------------------
workflow_fit_linear_regression <- workflow() %>%
  add_recipe(recipe_specification) %>%
  add_model(model_specification_linear_regression) %>%
  fit(training(splits))

##* Calibration, Evaluation & Plotting -----------------------------------------
workflow_fit_linear_regression %>% extract_fit_parsnip() %>% tidy()

workflow_fit_linear_regression %>% 
  calibrate_evaluate_plot(y = "life_expect", mode = "regression", type = "testing")


# Principal Component Analysis #------------------------------------------------
apply(df_diseases, 2, mean)
apply(df_diseases, 2 , var)

# Perform the PCA
pr.out <- prcomp(df_diseases, scale = FALSE)

# Produce the biplot
biplot(pr.out, scale = 0)

# Compute the variance explained by each principal component
pr.var <- pr.out$sdev^2

# Compute PVE by each principal component
pve <- pr.var/sum(pr.var)
pve

# Scree plot
plot(pve , xlab = "Principal Component", ylab = "PVE", ylim = c(0, 1),type = "b")
plot(cumsum(pve), xlab = "Principal Component", ylab = "Cumulative PVE", ylim = c(0, 1), type = "b")

# Re-initialize the dataset with the principal component obtained
PC1 <- as.data.frame(pr.out$x[,1])
names(PC1) <- "disease_PC"
dataset_pca <- cbind(dataset, PC1) %>% select(-c("polio", "diphtheria", "measles", "hepatitis"))


# Linear regression using the Principal Component#------------------------------
# Train/test split #------------------------------------------------------------
set.seed(123)
splits_pca <- initial_split(dataset_pca, prop = .8)
folds_pca <-vfold_cv(training(splits_pca))

##* Recipe ---------------------------------------------------------------------
recipe_specification_pca <- recipe(life_expect ~ ., data = training(splits_pca))
recipe_specification_pca %>% prep() %>% juice() %>% glimpse()

##* Workflow -------------------------------------------------------------------
workflow_fit_linear_regression_pca <- workflow() %>%
  add_recipe(recipe_specification_pca) %>%
  add_model(model_specification_linear_regression) 

##* Collect the metrics -----------------------------------------
workflow_fit_linear_regression_pca %>% last_fit(split = splits_pca) %>% collect_metrics()
summary(lm(life_expect~., data = dataset_pca))

# Subset selection #------------------------------------------------------------
# Select the best subset selection using a cross validation approach
predict.regsubsets <- function(object , newdata , id, ...) {
  form <- as.formula(object$call [[2]])
  mat <- model.matrix(form , newdata)
  coefi <- coef(object , id = id)
  xvars <- names(coefi)
  mat[, xvars] %*% coefi
}

n <- nrow(training(splits))
set.seed (123)
k = 10
folds_index <- sample(rep (1:k, length = n))
cv.errors <- matrix(NA, k, 15, dimnames = list(NULL , paste (1:15)))
for (j in 1:k) {
  best.fit <- regsubsets(life_expect ~.,data = training(splits)[folds_index != j, ], nvmax = 15)
  for (i in 1:15) {
    pred <- predict(best.fit , dataset[folds_index == j, ], id = i)
    cv.errors[j,i] <- mean((dataset$life_expect[folds_index == j] - pred)^2)
  }
}
cv.errors
mse.cv <- apply(cv.errors , 2, mean)
mse.cv
mean(mse.cv) #cv estimate of the mean squared error

# Cross validation selects a model w/ 8 variables
par(mfrow = c(1, 1))
which.min(mse.cv)
plot(mse.cv , type = "b", ylab = "CV estimate of the MSE", xlab = "Number of regressors", main = "Number of regressor vs MSE error (CV estimate)")
points (10, mse.cv[10] , col = "blue", cex = 2,pch = 20)

# Estimate the RMSE on the test set
regression.bestsubset <- regsubsets(life_expect~., data = training(splits) , nvmax = 10)
test_matrix <- model.matrix(life_expect~., data = testing(splits))
coef10 <- coef(regression.bestsubset , id = 10)
predicted_values <- test_matrix[,names(coef10)]%*%coef10
rmse_test_best_subset <- sqrt(mean((testing(splits)$life_expect-predicted_values)^2)) 
rmse_test_best_subset

colnames(dataset)
# Given that we want a model with 10 variables, perform best subset selection on 
# the full dataset to discover which variables are selected and their coefficients.
reg.bestfinal <- regsubsets(life_expect~., data = dataset , nvmax = 10)
coef10 <- coef(reg.bestfinal , id = 10)
coef10

# Lasso, Ridge, Elastic Net #---------------------------------------------------
##* Engines --------------------------------------------------------------------
# Ridge
ridge_model <-  linear_reg( #RIDGE
  mode = "regression",
  penalty = tune(), #to be tuned!
  mixture = 0
) %>%
  set_engine("glmnet")

# Lasso
lasso_model <-  linear_reg( #LASSO
  mode = "regression",
  penalty = tune(), #to be tuned!
  mixture = 1
) %>%
  set_engine("glmnet")

# Elastic net
elastic_net_model <-  linear_reg( #ELASTIC NET (w/ mixture of norms to be tuned)
  mode = "regression",
  penalty = tune(), #to be tuned!
  mixture = tune() #to be tuned!
) %>%
  set_engine("glmnet")

##* Workflows ----------------------------------------------------------------
workflow_ridge <- workflow() %>% #RIDGE
  add_model(ridge_model) %>%
  add_recipe(recipe_specification)

workflow_lasso <- workflow() %>% #LASSO
  add_model(lasso_model) %>%
  add_recipe(recipe_specification)

workflow_elastic_net <- workflow() %>% #ELASTIC NET
  add_model(elastic_net_model) %>%
  add_recipe(recipe_specification)


##* Grid Search --------------------------------------------------------------
# Grid for only the penalty (to be used for lasso and ridge).
set.seed(42)
model_grid_penalty <- grid_regular( 
  penalty(),
  levels = 50
)
model_grid_penalty
model_grid_penalty %>% map(unique)

# Grid for both the penalty and the mixture (to be used for elastic net).
model_grid_penalty_mixture <- grid_regular(
  penalty(),
  mixture(),
  levels = 50
)

model_grid_penalty_mixture
model_grid_penalty_mixture %>% map(unique)

##* Tuning -------------------------------------------------------------------
set.seed(42)

# Tuning of the hyperparameter (penalty) for the Ridge
parsnip_ctrl <- control_grid(extract = get_glmnet_coefs)
model_result_ridge <- workflow_ridge %>% 
  tune_grid(
    resamples = folds,
    grid = model_grid_penalty
  )

# Tuning of the hyperparameter (penalty) for the Lasso
model_result_lasso <- workflow_lasso %>% 
  tune_grid(
    resamples = folds,
    grid = model_grid_penalty
    
  )

# Tuning of the hyperparameters (penalty and mixture) for the Elastic Net
model_result_elastic_net <- workflow_elastic_net %>% 
  tune_grid(
    resamples = folds,
    grid = model_grid_penalty_mixture
  )


##* Evaluation  --------------------------------------------------------------
# Select the best values of the hyperparameter choosing the value that minimizes
# the accuracy
model_result_ridge %>% collect_metrics()
model_ridge_best <- model_result_ridge %>% select_best("rmse")
model_ridge_best

model_result_lasso %>% collect_metrics()
model_lasso_best <- model_result_lasso %>% select_best("rmse")
model_lasso_best

model_result_elastic_net %>% collect_metrics()
model_elastic_net_best <- model_result_elastic_net %>% select_best("rmse")
model_elastic_net_best

##* Re-fitting  --------------------------------------------------------------
# Refit the models on the entire training set using the best value of the hyperparameters.
# Test the performance of this model on the test set.
workflow_ridge_final <-	workflow_ridge %>%	
  finalize_workflow(model_ridge_best) %>% 
  last_fit(splits) 

workflow_lasso_final <-	workflow_lasso %>%	
  finalize_workflow(model_lasso_best) %>% 
  last_fit(splits)

workflow_elastic_net_final <-	workflow_elastic_net %>%	
  finalize_workflow(model_elastic_net_best) %>% 
  last_fit(splits)

workflow_ridge_final %>%	collect_metrics()
workflow_ridge_final %>% collect_predictions()

workflow_lasso_final %>%	collect_metrics()
workflow_lasso_final %>% collect_predictions()

workflow_elastic_net_final %>%	collect_metrics()
workflow_elastic_net_final %>% collect_predictions()

# Tree-based models #-----------------------------------------------------------
##* Recipes --------------------------------------------------------------------
recipe_trees <- recipe(life_expect ~ ., data = training(splits_pca))
recipe_trees%>%prep() %>%juice() %>% glimpse()

## Simple Tree ##----------------------------------------------------------------
###* Engine --------------------------------------------------------------------
model_spec_tree <- decision_tree(
  mode = "regression",
  cost_complexity = tune(), #cost-complexity parameter
  tree_depth = tune(), # maximum depth of a tree
  min_n = tune() # minimum number of data points in a node that are required for the node to be split further
) %>% set_engine('rpart')

###* Workflow ------------------------------------------------------------------
wrkfl_tree <- workflow() %>%
  add_model(model_spec_tree) %>%
  add_recipe(recipe_trees)

###* Grid Search ---------------------------------------------------------------
set.seed(123)               
tree_grid <- grid_regular(
  min_n(),
  cost_complexity(),
  tree_depth(range = c(1,29)),
  levels = 5
)

###* Tuning  -------------------------------------------------------------------
set.seed(123)               
tree_res <- wrkfl_tree %>% 
  tune_grid(
    resamples = folds_pca,
    grid = tree_grid
  )
###* Select the best model --------------------------------------------------------------------------------------------------
tree_res %>%  collect_metrics()
tree_best <- tree_res %>% select_best("rmse")
tree_best


###* Refit----------------------------------------------------------------------
wrkfl_tree_final <-	wrkfl_tree %>%	
  finalize_workflow(tree_best) %>% 
  last_fit(splits_pca)

wrkfl_tree_final %>%	collect_metrics()
wrkfl_tree_final %>% collect_predictions()

###* Plotting ------------------------------------------------------------------
wrkfl_tree_final %>%
  extract_fit_engine() %>%
  rpart.plot::rpart.plot(roundint = FALSE,5) 

wrkfl_tree_final %>% 
  extract_fit_parsnip() %>% 
  vip::vip()

## Random Forest ##-------------------------------------------------------------
###* Engine ---------------------------------------------------------------------
model_spec_rf <- rand_forest(
  mode = "regression",
  mtry = 3,
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger", importance="permutation") 

###* Workflow ------------------------------------------------------------------
wrkfl_rf <- workflow() %>%      
  add_model(model_spec_rf) %>%
  add_recipe(recipe_trees)

###* Grid Search ---------------------------------------------------------------
set.seed(123)        
rf_grid <- grid_regular(
  min_n(),
  trees(),
  levels = 5
)  

###* Tuning  -------------------------------------------------------------------
set.seed(123)
rf_res <- wrkfl_rf %>% 
  tune_grid(
    resamples = folds_pca,
    grid = rf_grid
  )

###* Select the best model --------------------------------------------------------------------------------------------------
rf_res %>%  collect_metrics()
rf_best <- rf_res %>% select_best("rmse")
rf_best

###* Refit  --------------------------------------------------------------------
wrkfl_rf_final <-	wrkfl_rf %>%
  finalize_workflow(rf_best) %>% 
  last_fit(splits_pca)

wrkfl_rf_final %>% collect_metrics()
wrkfl_rf_final %>% collect_predictions()

###* Plotting ------------------------------------------------------------------
wrkfl_rf_final %>% 
  extract_fit_parsnip() %>% 
  vip::vip()

# K-means clustering #----------------------------------------------------------
set.seed(123)
km.out<- kmeans(dataset_kmeans%>%select(-c("country")), 2, nstart=50)
km.out$cluster
km.out$withinss

##* Map Plot --------------------------------------------------------------------
clusters_df <- as.data.frame(km.out$cluster)
df <- data.frame(rownames(clusters_df), clusters_df[,1])
colnames(df) <- c("country", "cluster")
malMap <- joinCountryData2Map(df, joinCode = "ISO3",nameJoinColumn = "country")
mapParams <- mapCountryData(malMap, nameColumnToPlot="cluster", 
               catMethod = "categorical",missingCountryCol = gray(.8), addLegend = FALSE,
               colourPalette = c("blue", "lightblue"), lwd =1, mapTitle="World Map of the Clusters")
do.call( addMapLegendBoxes, c( mapParams, x="topright", horiz=TRUE, title="cluster"))

##* Distribution of each variable in each cluster-------------------------------
dataset_kmeans["country"] <- rownames(dataset_kmeans)
joined_df <- merge(x=dataset_kmeans, y=df, by="country")

meltData <- melt(joined_df)
meltData <- merge(x=meltData, y=df, by="country")
meltData <- subset(meltData, variable != "cluster")

theme_update(plot.title = element_text(hjust = 0.5))
p <- ggplot(meltData, aes(factor(cluster), value))+
    geom_boxplot() +
    facet_wrap(~variable, scale="free")+
    xlab("Clusters")+
    ylab("Distribution of each variable")+
    labs(title="Distribution of each variable in the clusters")
p 



