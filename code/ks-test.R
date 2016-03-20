
''
#
# ks test cutoff to select features
# see around line 109
#
library(data.table) #Faster reading
library(xgboost)

# Start the clock!
start_time <- t1 <- Sys.time()

na.roughfix2 <- function (object, ...) {
  res <- lapply(object, roughfix)
  structure(res, class = "data.frame", row.names = seq_len(nrow(object)))
}

roughfix <- function(x) {
  missing <- is.na(x)
  if (!any(missing)) return(x)
  
  if (is.numeric(x)) {
    x[missing] <- median.default(x[!missing])
  } else if (is.factor(x)) {
    freq <- table(x)
    x[missing] <- names(freq)[which.max(freq)]
  } else {
    stop("na.roughfix only works for numeric or factor")
  }
  x
}

# Set a seed for reproducibility
set.seed(2016)

cat("reading the train and test data\n")
# Read train and test
train_raw <- fread("./input/train.csv", stringsAsFactors=TRUE) 
print(dim(train_raw))
print(sapply(train_raw, class))

y <- train_raw$target
train_raw$target <- NULL
train_raw$ID <- NULL
n <- nrow(train_raw)

test_raw <- fread("./input/test.csv", stringsAsFactors=TRUE) 
test_id <- test_raw$ID
test_raw$ID <- NULL
print(dim(test_raw))
print(sapply(test_raw, class))
cat("Data read ")
print(difftime( Sys.time(), start_time, units = 'sec'))

# Preprocess data
# Find factor variables and translate to numeric
cat("Preprocess data\n")
all_data <- rbind(train_raw,test_raw)
all_data <- as.data.frame(all_data) # Convert data table to data frame

# Convert v22 to hexavigesimal base
az_to_int <- function(az) {
  xx <- strsplit(tolower(az), "")[[1]]
  pos <- match(xx, letters[(1:26)]) 
  result <- sum( pos* 26^rev(seq_along(xx)-1))
  return(result)
}

all_data$v22<-sapply(all_data$v22, az_to_int)

feature.names <- names(all_data)

cat("assuming text variables are categorical & replacing them with numeric ids\n")
cat("re-factor categorical vars & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(all_data[[f]])=="character" || class(all_data[[f]])=="factor") {
    #levels <- unique(c(all_data[[f]]))
    #all_data[[f]] <- as.integer(factor(all_data[[f]], levels=levels))
    all_data[[f]] <- as.integer(factor(all_data[[f]]))
  }
}

# Small feature addition - Count NA percentage
N <- ncol(all_data)
all_data$NACount_N <- rowSums(is.na(all_data)) / N 

# make feature of counts of zeros factor
all_data$ZeroCount <- rowSums(all_data[,feature.names]== 0) / N
all_data$Below0Count <- rowSums(all_data[,feature.names] < 0) / N

feature.names <- names(all_data)

train <- all_data[1:n,]
test <- all_data[(n+1):nrow(all_data),] 

print(dim(train))
summary(train)
print(dim(test))
summary(test)

rm(all_data)
gc()

#Feature selection using KS test with 0.007 as cutoff.
tmpJ = 1:ncol(test)
ksMat = NULL
for (j in tmpJ) {
  cat(j," ")
  ksMat = rbind(ksMat, cbind(j, ks.test(train[,j],test[,j])$statistic))
}

ksMat2 = ksMat[ksMat[,2]<0.004,]
feats = as.numeric(ksMat2[,1]) 
cat(length(feats),"\n")
cat(names(train)[feats],"\n")

# Input missing data & convert to xgb-data structure
#train[is.na(train)] <- -1
#test[is.na(test)] <- -1

#xgtrain = xgb.DMatrix(as.matrix(train[,feats]), label = y, missing = -1)
#xgtest = xgb.DMatrix(as.matrix(test[,feats]), missing=-1)

all_data <- rbind(train[,feats],test[,feats])


all_data <- na.roughfix2(all_data)
train <- all_data[1:n,]
test <- all_data[(n+1):nrow(all_data),] 

xgtrain = xgb.DMatrix(as.matrix(train), label = y)
xgtest = xgb.DMatrix(as.matrix(test))
#xgtrain = xgb.DMatrix(as.matrix(train[,feats]), label = y)
#xgtest = xgb.DMatrix(as.matrix(test[,feats]))

# Do cross-validation with xgboost - xgb.cv
docv <- function(param0, iter) {
  model_cv = xgb.cv(
    params = param0
    , nrounds = iter
    , nfold = 5
    , data = xgtrain
    , early.stop.round = 200
    , maximize = FALSE
    , nthread = 8
  )
  gc()
  best <- min(model_cv$test.logloss.mean)
  bestIter <- which(model_cv$test.logloss.mean==best)
  
  cat("\n",best, bestIter,"\n")
  print(model_cv[bestIter])
  
  bestIter-1
}

doTest <- function(param0, iter) {
  watchlist <- list('train' = xgtrain)
  model = xgb.train(
    nrounds = iter
    , params = param0
    , data = xgtrain
    , watchlist = watchlist
    , print.every.n = 20
    , nthread = 8
  )
  p <- predict(model, xgtest)
  rm(model)
  gc()
  p
}

param0 <- list(
  # some generic, non specific params
  "objective"  = "binary:logistic"
  , "eval_metric" = "logloss"
  , "eta" = 0.01  # 0.05
  , "subsample" = 0.96
  , "colsample_bytree" = 0.45 # 0.9
  , "min_child_weight" = 1.0  # 1
  , "max_depth" = 11.0 ## 10
  
  #n estimator = 520.0 eta = 0.025
  #max_depth = 12.0 min_child_weigth = 6.0
  #subsample = 0.95 gamma = 0.95
  #colsample_bytree = 0.5
  #scale_pos_weight = 0.85
  # Score 0.460934919895
  
  #, "eta" = 0.025  # 0.05
  #, "subsample" = 0.95
  #, "colsample_bytree" = 0.5 # 0.9
  #, "min_child_weight" = 6.0  # 1
  #, "max_depth" = 12.0 ## 10
  
  
 # md <- 11
 # ss <- 0.96
#  cs <- 0.45
 # mc <- 1
#  np <- 1
  
)

#print(proc.time() - start_time)
print( difftime( Sys.time(), start_time, units = 'sec'))
cat("Training a XGBoost classifier with cross-validation\n")
set.seed(2016)
cv <- docv(param0, 1500) 
# Show the clock
print( difftime( Sys.time(), start_time, units = 'sec'))

# sample submission total analysis
submission <- read.csv("./input/sample_submission.csv")
ensemble <- rep(0, nrow(test))

cv <- round(cv * 1.5)
cat("Calculated rounds:", cv, " Starting ensemble\n")

# Bagging of single xgboost for ensembling
# change to e.g. 1:10 to get quite good results
for (i in 1:10) {
  print(i)
  set.seed(i + 2015)
  p <- doTest(param0, cv) 
  # use 40% to 50% more than the best iter rounds from your cross-fold number.
  # as you have another 50% training data now, which gives longer optimal training time
  ensemble <- ensemble + p
}

# Finalise prediction of the ensemble
cat("Making predictions\n")
submission$PredictedProb <- ensemble/i

# Prepare submission
write.csv(submission, "bnp-xgb-ks3-new.csv", row.names=F, quote=F)
summary(submission$PredictedProb)

# Stop the clock
#print(proc.time() - start_time)
print( difftime( Sys.time(), start_time, units = 'min'))
''
