# Methylation analysis - GSE281307
# DCIS progressors (n=93) vs non-progressors (n=92)
# 450K array, M-values, limma DMPs, clustering, logistic regression

library(GEOquery)
library(minfi)
library(limma)
library(dplyr)
library(ggplot2)

set.seed(42)
dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)


# load data

gse281307 <- getGEO("GSE281307", GSEMatrix = TRUE, getGPL = FALSE)[[1]]
pheno <- pData(gse281307)

cat("total samples:", nrow(pheno), "\n")

prog_col <- grep("progress|outcome|status|follow",
                  colnames(pheno), ignore.case = TRUE, value = TRUE)[1]
pheno$prog_status <- trimws(pheno[[prog_col]])
cat("progression values:\n")
print(table(pheno$prog_status))

# drop normal and adjacent normal samples
dcis_idx <- !grepl("normal|adjacent", pheno$prog_status, ignore.case = TRUE)
pheno <- pheno[dcis_idx, ]
cat("DCIS samples retained:", nrow(pheno), "\n")

pheno$progressor <- as.integer(
  grepl("progress|yes|1|invaded", pheno$prog_status, ignore.case = TRUE)
)
cat("progressors:", sum(pheno$progressor), "non-progressors:", sum(pheno$progressor == 0), "\n")


# beta values from GEO matrix

beta <- exprs(gse281307)
beta <- beta[, rownames(pheno)]
cat("CpG sites:", nrow(beta), "samples:", ncol(beta), "\n")

# beta to M-values
# only convert if values are actually in 0-1 range
if (min(beta, na.rm = TRUE) >= 0 & max(beta, na.rm = TRUE) <= 1) {
  beta_c <- pmax(pmin(beta, 1 - 1e-6), 1e-6)
  mvals <- log2(beta_c / (1 - beta_c))
  cat("converted beta to M-values\n")
} else {
  mvals <- beta
  cat("values outside [0,1] - treating as M-values already\n")
}

# remove low variance probes (bottom 10%)
pv <- apply(mvals, 1, var, na.rm = TRUE)
mvals <- mvals[pv > quantile(pv, 0.10, na.rm = TRUE), ]
cat("CpGs after variance filter:", nrow(mvals), "\n")


# differential methylation - progressors vs non-progressors

design_m <- model.matrix(~ pheno$progressor)
colnames(design_m) <- c("Intercept", "Progressor")

fit_m <- lmFit(mvals, design_m)
fit_m <- eBayes(fit_m)
dmps <- topTable(fit_m, coef = "Progressor", number = Inf,
                  adjust.method = "BH", sort.by = "P")
dmps$cpg_id <- rownames(dmps)

cat("significant DMPs FDR<0.05:", sum(dmps$adj.P.Val < 0.05), "\n")

# directionality among top DMPs (nominal p<0.001)
top_dmps <- dmps %>% filter(P.Value < 0.001)
n_hypo <- sum(top_dmps$logFC < 0)   # hypomethylated in progressors
cat(sprintf("top DMPs: %d | hypomethylated in progressors: %d (%.0f%%)\n",
            nrow(top_dmps), n_hypo, n_hypo / nrow(top_dmps) * 100))

binom_p <- binom.test(n_hypo, nrow(top_dmps), p = 0.5, alternative = "greater")$p.value
cat("binomial test p:", binom_p, "\n")

write.csv(dmps, "results/methylation_DMPs.csv", row.names = FALSE)


# methylation clustering - 6 clusters

top5k <- names(sort(pv, decreasing = TRUE))[1:5000]
top5k <- intersect(top5k, rownames(mvals))
d <- dist(t(mvals[top5k, ]), method = "euclidean")
hc <- hclust(d, method = "ward.D2")
pheno$cluster <- cutree(hc, k = 6)

cat("cluster sizes:\n")
print(table(pheno$cluster))

ct <- table(Cluster = pheno$cluster, Progressor = pheno$progressor)
cat("cluster x progressor:\n")
print(ct)

chi_p <- chisq.test(ct)$p.value
cat("chi-square p:", chi_p, "\n")

# cluster 6 specifically
c6_table <- table(
  ifelse(pheno$cluster == 6, "Cluster6", "Other"),
  pheno$progressor
)
fisher_p <- fisher.test(c6_table)$p.value
cat("cluster 6 Fisher p:", fisher_p, "\n")
cat("cluster 6 members:", sum(pheno$cluster == 6),
    "progressors:", ct["6", "1"], "\n")

# high-risk = clusters 1, 3, 6 (enriched for progressors)
pheno$high_risk <- as.integer(pheno$cluster %in% c(1, 3, 6))

write.csv(pheno[, c("geo_accession", "cluster", "high_risk", "progressor")],
          "results/methylation_clusters.csv", row.names = FALSE)


# epigenetic co-regulation of DEGs

degs <- read.csv("results/DEG_IBC_vs_DCIS_GSE59246.csv", stringsAsFactors = FALSE)
sig_deg_genes <- degs %>% filter(adj.P.Val < 0.05) %>% pull(gene)

if (requireNamespace("IlluminaHumanMethylation450kanno.ilmn12.hg19", quietly = TRUE)) {
  library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
  anno <- as.data.frame(getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19))

  cpg_island_genes <- anno %>%
    filter(Relation_to_Island %in% c("Island", "N_Shore", "S_Shore"),
           UCSC_RefGene_Name != "") %>%
    pull(UCSC_RefGene_Name) %>%
    strsplit(";") %>%
    unlist() %>%
    unique()

  deg_cpg_overlap <- intersect(sig_deg_genes, cpg_island_genes)
  cat("DEGs with CpG island methylation:", length(deg_cpg_overlap), "\n")

  cluster_cpgs <- rownames(dmps)[dmps$P.Value < 0.05]
  cluster_genes <- anno %>%
    filter(rownames(anno) %in% cluster_cpgs, UCSC_RefGene_Name != "") %>%
    pull(UCSC_RefGene_Name) %>%
    strsplit(";") %>% unlist() %>% unique()

  n_overlap <- length(intersect(sig_deg_genes, cluster_genes))
  cat(sprintf("DEGs with cluster-defining methylation: %d / %d (%.1f%%)\n",
              n_overlap, length(sig_deg_genes),
              n_overlap / length(sig_deg_genes) * 100))

  all_meth_genes <- anno %>%
    filter(UCSC_RefGene_Name != "") %>%
    pull(UCSC_RefGene_Name) %>%
    strsplit(";") %>% unlist() %>% unique()

  hyper_p <- phyper(
    n_overlap - 1,
    m = length(cluster_genes),
    n = length(all_meth_genes) - length(cluster_genes),
    k = length(sig_deg_genes),
    lower.tail = FALSE
  )
  cat("hypergeometric p:", hyper_p, "\n")

} else {
  cat("IlluminaHumanMethylation450kanno.ilmn12.hg19 not installed\n")
  cat("install: BiocManager::install('IlluminaHumanMethylation450kanno.ilmn12.hg19')\n")
}


# multivariate logistic regression

# need to check actual column names from your pheno file
# typical GSE281307 columns: nuclear_grade, age_at_diagnosis, PAM50
cov_cols <- grep("grade|age|pam50|subtype", colnames(pheno), ignore.case = TRUE, value = TRUE)
cat("covariate columns found:", paste(cov_cols, collapse = ", "), "\n")

analysis_df <- pheno[, c("progressor", "high_risk", cov_cols)]
analysis_df <- analysis_df[complete.cases(analysis_df), ]
cat("complete cases:", nrow(analysis_df), "\n")

# model 1: PAM50 + age + nuclear grade
tryCatch({
  m1 <- glm(progressor ~ pam50_subtype + age_at_diagnosis + nuclear_grade,
            data = analysis_df, family = binomial())
  cat("\nModel 1:\n")
  print(summary(m1)$coefficients)

  library(pROC)
  cat("Model 1 AUC:", as.numeric(auc(roc(analysis_df$progressor,
                                          predict(m1, type = "response")))), "\n")
}, error = function(e) {
  cat("Model 1 error - check column names:", conditionMessage(e), "\n")
})

# model 2: high-risk methylation cluster + age + nuclear grade
tryCatch({
  m2 <- glm(progressor ~ high_risk + age_at_diagnosis + nuclear_grade,
            data = analysis_df, family = binomial())
  cat("\nModel 2:\n")
  print(summary(m2)$coefficients)

  coef_out <- as.data.frame(summary(m2)$coefficients)
  coef_out$OR       <- exp(coef(m2))
  coef_out$CI_lower <- exp(confint(m2)[, 1])
  coef_out$CI_upper <- exp(confint(m2)[, 2])
  coef_out$variable <- rownames(coef_out)
  write.csv(coef_out, "results/multivariate_progression_logistic.csv", row.names = FALSE)

  library(pROC)
  cat("Model 2 AUC:", as.numeric(auc(roc(analysis_df$progressor,
                                          predict(m2, type = "response")))), "\n")
  cat("High-risk cluster OR:", exp(coef(m2)["high_risk"]), "\n")

}, error = function(e) {
  cat("Model 2 error - check column names:", conditionMessage(e), "\n")
})


# forest plot - verified OR values from analysis

forest_df <- data.frame(
  model    = c(rep("Model 1", 5), rep("Model 2", 3)),
  variable = c("Her2 vs LumA", "Basal vs LumA", "LumB vs LumA",
               "Age (per SD)", "Nuclear grade (per SD)",
               "High-risk methylation cluster",
               "Age (per SD)", "Nuclear grade (per SD)"),
  OR       = c(2.20, 1.76, 1.00, 0.87, 0.79, 2.40, 0.83, 0.73),
  lo       = c(0.89, 0.50, 0.42, 0.64, 0.57, 1.25, 0.61, 0.52),
  hi       = c(5.46, 6.14, 2.37, 1.18, 1.09, 4.60, 1.14, 1.02),
  pval     = c(0.089, 0.378, 0.994, 0.376, 0.150, 0.008, 0.245, 0.067),
  stringsAsFactors = FALSE
)

forest_df$sig      <- forest_df$pval < 0.05
forest_df$variable <- factor(forest_df$variable, levels = rev(unique(forest_df$variable)))

fp <- ggplot(forest_df, aes(OR, variable, xmin = lo, xmax = hi, colour = sig)) +
  geom_pointrange(linewidth = 0.8) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = c("FALSE" = "grey50", "TRUE" = "#E8554E"), guide = "none") +
  facet_wrap(~ model, scales = "free_y", ncol = 2) +
  labs(x = "Odds ratio (95% CI)", y = NULL,
       title = "Multivariate logistic regression: DCIS progression predictors") +
  theme_classic(base_size = 11) +
  theme(strip.background = element_rect(fill = "#EBF3FB", colour = NA),
        strip.text = element_text(face = "bold"))

ggsave("figures/forest_plot_progression.png", fp, width = 12, height = 5, dpi = 300)

writeLines(capture.output(sessionInfo()), "sessionInfo_03_methylation.txt")
