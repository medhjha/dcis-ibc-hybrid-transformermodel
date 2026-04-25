# Elastic net classifier - IBC-like signature
# input: 2996 significant DEGs from GSE59246
# output: 75-gene model, AUC 0.9972

library(glmnet)
library(pROC)
library(ggplot2)
library(dplyr)
library(caret)

set.seed(42)
dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)


# load data

expr  <- read.csv("data/expr_GSE59246_normalised.csv", row.names = 1, check.names = FALSE)
pheno <- read.csv("data/pheno_GSE59246_clean.csv", stringsAsFactors = FALSE)
degs  <- read.csv("results/DEG_IBC_vs_DCIS_GSE59246.csv", stringsAsFactors = FALSE)

sig_genes <- degs %>% filter(adj.P.Val < 0.05) %>% pull(gene)
cat("DEG input features:", length(sig_genes), "\n")

# align samples
common <- intersect(colnames(expr), pheno$sample_id)
expr  <- expr[, common]
pheno <- pheno[match(common, pheno$sample_id), ]

# IBC and DCIS only
keep <- grepl("IBC|Invasive|DCIS", pheno$disease_state, ignore.case = TRUE)
expr  <- expr[, keep]
pheno <- pheno[keep, ]

# feature matrix - samples x genes
avail <- intersect(sig_genes, rownames(expr))
X <- t(as.matrix(expr[avail, ]))
cat("feature matrix:", nrow(X), "x", ncol(X), "\n")

# labels: IBC=1, DCIS=0
y <- as.integer(grepl("IBC|Invasive", pheno$disease_state, ignore.case = TRUE))
cat("IBC:", sum(y), "DCIS:", sum(y == 0), "\n")


# grid search over alpha (0=ridge, 1=lasso)

alpha_grid <- seq(0, 1, by = 0.1)
cv_res <- lapply(alpha_grid, function(a) {
  cv <- cv.glmnet(X, y, family = "binomial", alpha = a,
                   nfolds = 10, type.measure = "auc", standardize = TRUE)
  list(alpha = a, cv = cv, auc = max(cv$cvm), lambda = cv$lambda.min)
})

best_idx    <- which.max(sapply(cv_res, function(x) x$auc))
best_alpha  <- cv_res[[best_idx]]$alpha
best_lambda <- cv_res[[best_idx]]$lambda
best_cv     <- cv_res[[best_idx]]$cv

cat(sprintf("best alpha: %.1f | lambda: %.6f | CV AUC: %.4f\n",
            best_alpha, best_lambda, cv_res[[best_idx]]$auc))


# final model

final <- glmnet(X, y, family = "binomial",
                 alpha = best_alpha, lambda = best_lambda, standardize = TRUE)

coefs <- as.data.frame(as.matrix(coef(final)))
colnames(coefs) <- "coef"
coefs$gene <- rownames(coefs)
coefs <- coefs %>% filter(gene != "(Intercept)", coef != 0) %>%
  arrange(desc(abs(coef)))

cat("genes selected:", nrow(coefs), "\n")
cat("top 10:\n")
print(head(coefs, 10))

write.csv(coefs, "results/elastic_net_75genes.csv", row.names = FALSE)


# held-out test performance - 80/20 stratified split

set.seed(42)
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
Xtrain <- X[train_idx, ];  Xtest <- X[-train_idx, ]
ytrain <- y[train_idx];    ytest <- y[-train_idx]

m_test <- glmnet(Xtrain, ytrain, family = "binomial",
                  alpha = best_alpha, lambda = best_lambda, standardize = TRUE)

preds <- predict(m_test, newx = Xtest, type = "response")[, 1]
pred_class <- as.integer(preds >= 0.5)

roc_obj <- roc(ytest, preds, quiet = TRUE)
auc_val <- as.numeric(auc(roc_obj))

cm <- table(pred = pred_class, actual = ytest)
sens <- cm["1", "1"] / (cm["1", "1"] + cm["0", "1"])
spec <- cm["0", "0"] / (cm["0", "0"] + cm["1", "0"])

cat(sprintf("AUC: %.4f | sensitivity: %.4f | specificity: %.4f\n", auc_val, sens, spec))

write.csv(
  data.frame(metric = c("AUC", "sensitivity", "specificity", "alpha", "lambda", "n_genes"),
             value  = c(auc_val, sens, spec, best_alpha, best_lambda, nrow(coefs))),
  "results/elastic_net_cv_performance.csv", row.names = FALSE
)


# ROC curve

roc_df <- data.frame(fpr = 1 - roc_obj$specificities, tpr = roc_obj$sensitivities)

p_roc <- ggplot(roc_df, aes(fpr, tpr)) +
  geom_line(colour = "#E8554E", linewidth = 1.2) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  annotate("text", x = 0.6, y = 0.1,
           label = sprintf("AUC = %.4f", auc_val),
           size = 4.5, colour = "#E8554E", fontface = "bold") +
  labs(x = "1 - specificity", y = "sensitivity",
       title = "ROC curve - elastic net classifier (75 genes)") +
  theme_classic(base_size = 11) +
  coord_equal()

ggsave("figures/ROC_elastic_net.pdf", p_roc, width = 6, height = 6)
ggsave("figures/ROC_elastic_net.png", p_roc, width = 6, height = 6, dpi = 300)


# lambda CV curve

cv_df <- data.frame(
  loglambda = log(best_cv$lambda),
  auc       = best_cv$cvm,
  upper     = best_cv$cvup,
  lower     = best_cv$cvlo
)

p_cv <- ggplot(cv_df, aes(loglambda, auc)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#BDDCF0", alpha = 0.4) +
  geom_line(colour = "#2E75B6", linewidth = 1) +
  geom_vline(xintercept = log(best_lambda), linetype = "dashed",
             colour = "#E8554E", linewidth = 0.8) +
  labs(x = expression(log(lambda)), y = "mean CV AUC",
       title = sprintf("Elastic net CV - alpha=%.1f, lambda.min=%.5f",
                       best_alpha, best_lambda)) +
  theme_classic(base_size = 11)

ggsave("figures/elastic_net_CV.png", p_cv, width = 8, height = 5, dpi = 300)

writeLines(capture.output(sessionInfo()), "sessionInfo_04_elasticnet.txt")
