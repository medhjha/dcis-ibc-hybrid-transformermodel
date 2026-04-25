# Immune microenvironment analysis
# GSE87517: sorted CD45+CD3+ T cells, Normal/DCIS/IDC
# GSE59246: invasive-like vs DCIS-like subgroup validation

library(GEOquery)
library(DESeq2)
library(ggplot2)
library(ggpubr)
library(pheatmap)
library(dplyr)
library(tidyr)
library(rstatix)

set.seed(42)
dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

MARKERS <- c("CTLA4", "FOXP3", "PDCD1", "LAG3", "TIGIT", "GZMB", "NKG7", "IFNG")


# GSE87517 - T-cell RNA-seq data
# Normal=19, DCIS=11, IDC=11 (sorted CD45+CD3+)

gse87517 <- getGEO("GSE87517", GSEMatrix = TRUE, getGPL = FALSE)[[1]]
pheno87 <- pData(gse87517)

type_col <- grep("tissue|type|disease|source",
                  colnames(pheno87), ignore.case = TRUE, value = TRUE)[1]
pheno87$group <- trimws(pheno87[[type_col]])

pheno87$disease <- case_when(
  grepl("normal|reduction", pheno87$group, ignore.case = TRUE) ~ "Normal",
  grepl("DCIS", pheno87$group, ignore.case = TRUE) ~ "DCIS",
  grepl("IDC|invasive", pheno87$group, ignore.case = TRUE) ~ "IDC",
  TRUE ~ NA_character_
)

cat("disease groups:\n")
print(table(pheno87$disease))

expr87 <- exprs(gse87517)

# map to gene symbols
feat87 <- fData(gse87517)
gsym_col <- grep("gene.symbol|Symbol|gene_name",
                  colnames(feat87), ignore.case = TRUE, value = TRUE)[1]

if (!is.na(gsym_col)) {
  feat87$gene_symbol <- feat87[[gsym_col]]
  valid <- !is.na(feat87$gene_symbol) & feat87$gene_symbol != ""
  expr87 <- expr87[valid, ]
  feat87 <- feat87[valid, ]

  feat87$mean_expr <- rowMeans(expr87)
  feat87$probe_id  <- rownames(feat87)
  best <- feat87 %>%
    group_by(gene_symbol) %>%
    slice_max(mean_expr, n = 1, with_ties = FALSE) %>%
    pull(probe_id)

  expr87 <- expr87[best, ]
  rownames(expr87) <- feat87$gene_symbol[match(best, feat87$probe_id)]
}

pheno87 <- pheno87[colnames(expr87), ]

avail_markers <- intersect(MARKERS, rownames(expr87))
cat("markers available:", paste(avail_markers, collapse = ", "), "\n")


# wilcoxon tests across disease stages

marker_long <- expr87[avail_markers, ] %>%
  as.data.frame() %>%
  tibble::rownames_to_column("gene") %>%
  pivot_longer(-gene, names_to = "sample", values_to = "expr") %>%
  left_join(pheno87 %>% select(geo_accession, disease) %>%
              rename(sample = geo_accession), by = "sample") %>%
  filter(!is.na(disease))

marker_long$disease <- factor(marker_long$disease, levels = c("Normal", "DCIS", "IDC"))

comps <- list(c("Normal", "DCIS"), c("Normal", "IDC"), c("DCIS", "IDC"))

stage_stats <- list()
for (m in avail_markers) {
  mdat <- marker_long %>% filter(gene == m)
  for (cp in comps) {
    g1 <- mdat %>% filter(disease == cp[1]) %>% pull(expr)
    g2 <- mdat %>% filter(disease == cp[2]) %>% pull(expr)
    if (length(g1) < 3 | length(g2) < 3) next
    wt <- wilcox.test(g1, g2, exact = FALSE)
    stage_stats[[paste(m, cp[1], cp[2])]] <- data.frame(
      marker = m, group1 = cp[1], group2 = cp[2],
      n1 = length(g1), n2 = length(g2),
      median1 = round(median(g1), 3), median2 = round(median(g2), 3),
      p = round(wt$p.value, 4),
      stringsAsFactors = FALSE
    )
  }
}

stage_df <- bind_rows(stage_stats)
cat("\nsignificant comparisons:\n")
print(stage_df %>% filter(p < 0.05))

write.csv(stage_df, "results/immune_mannwhitney_disease_stages.csv", row.names = FALSE)


# Figure 3 - boxplots across disease stages

cols <- c("Normal" = "#4DAF4A", "DCIS" = "#2E75B6", "IDC" = "#E8554E")

p_box <- ggplot(marker_long, aes(disease, expr, fill = disease)) +
  geom_boxplot(alpha = 0.8, outlier.size = 0.8, width = 0.6) +
  geom_jitter(width = 0.15, size = 0.5, alpha = 0.5, colour = "grey30") +
  scale_fill_manual(values = cols, guide = "none") +
  facet_wrap(~ gene, scales = "free_y", nrow = 2) +
  stat_compare_means(comparisons = comps, method = "wilcox.test",
                      label = "p.signif", size = 3, tip.length = 0.01) +
  labs(x = NULL, y = "VST-normalised expression",
       title = "Immune checkpoint markers in CD45+CD3+ T cells (GSE87517)",
       subtitle = "Normal (n=19), DCIS (n=11), IDC (n=11)") +
  theme_classic(base_size = 10) +
  theme(strip.text = element_text(face = "bold.italic"),
        strip.background = element_rect(fill = "#F0F0F0", colour = NA))

ggsave("figures/boxplot_immune_markers.png", p_box, width = 12, height = 7, dpi = 300)


# Supplementary S1 - heatmap

mat87 <- t(scale(t(expr87[avail_markers, ])))
anno87 <- data.frame(Disease = pheno87$disease, row.names = colnames(mat87))

png("figures/heatmap_immune_markers.png", width = 2400, height = 1600, res = 300)
pheatmap(mat87,
         annotation_col = anno87,
         annotation_colors = list(Disease = cols),
         color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
         cluster_rows = TRUE, cluster_cols = TRUE,
         show_colnames = FALSE, fontsize_row = 10, border_color = NA,
         main = "Immune markers: Normal / DCIS / IDC (z-scored)")
dev.off()


# invasive-like vs DCIS-like subgroup validation
# uses transformer-derived scores from Python pipeline

if (file.exists("results/dcis_invasive_scores.csv")) {
  scores <- read.csv("results/dcis_invasive_scores.csv", stringsAsFactors = FALSE)
  scores$subgroup <- ifelse(scores$invasive_score >= 0.755, "Invasive-like", "DCIS-like")
  cat("invasive-like:", sum(scores$subgroup == "Invasive-like"),
      "DCIS-like:", sum(scores$subgroup == "DCIS-like"), "\n")
} else {
  cat("dcis_invasive_scores.csv not found - run Python transformer pipeline first\n")
  stop()
}

# load GSE59246 expression for DCIS samples only
expr59 <- read.csv("data/expr_GSE59246_normalised.csv", row.names = 1, check.names = FALSE)
pheno59 <- read.csv("data/pheno_GSE59246_clean.csv", stringsAsFactors = FALSE)

dcis_samp <- pheno59$sample_id[grepl("DCIS", pheno59$disease_state, ignore.case = TRUE)]
expr_dcis <- expr59[, intersect(colnames(expr59), dcis_samp)]

subgroup_map <- setNames(scores$subgroup, scores$sample_id)

avail59 <- intersect(MARKERS, rownames(expr_dcis))
sub_stats <- list()

for (m in avail59) {
  inv  <- as.numeric(expr_dcis[m, names(subgroup_map)[subgroup_map == "Invasive-like"]])
  dcis <- as.numeric(expr_dcis[m, names(subgroup_map)[subgroup_map == "DCIS-like"]])
  inv  <- inv[!is.na(inv)]
  dcis <- dcis[!is.na(dcis)]
  if (length(inv) < 3 | length(dcis) < 3) next

  wt <- wilcox.test(inv, dcis, exact = FALSE)
  r_rb <- (2 * wt$statistic) / (length(inv) * length(dcis)) - 1

  spear_r <- cor(scores$invasive_score,
                  as.numeric(expr_dcis[m, scores$sample_id]),
                  method = "spearman", use = "complete.obs")

  sub_stats[[m]] <- data.frame(
    marker = m,
    n_invasive = length(inv), n_dcis = length(dcis),
    med_invasive = round(median(inv), 3),
    med_dcis     = round(median(dcis), 3),
    p            = round(wt$p.value, 4),
    r_biserial   = round(as.numeric(r_rb), 4),
    spearman_r   = round(spear_r, 4),
    stringsAsFactors = FALSE
  )
}

sub_df <- bind_rows(sub_stats)
cat("\ninvasive-like vs DCIS-like:\n")
print(sub_df[, c("marker", "med_invasive", "med_dcis", "p")])
cat("all p > 0.05:", all(sub_df$p > 0.05), "\n")
cat("p range:", min(sub_df$p), "-", max(sub_df$p), "\n")

write.csv(sub_df, "results/immune_mannwhitney_dcis_subgroups.csv", row.names = FALSE)


# Supplementary S4 - invasive-like validation boxplots

sub_long <- expr_dcis[avail59, ] %>%
  as.data.frame() %>%
  tibble::rownames_to_column("gene") %>%
  pivot_longer(-gene, names_to = "sample", values_to = "expr") %>%
  mutate(subgroup = subgroup_map[sample],
         subgroup = factor(subgroup, levels = c("DCIS-like", "Invasive-like"))) %>%
  filter(!is.na(subgroup))

sub_cols <- c("DCIS-like" = "#2E75B6", "Invasive-like" = "#E8554E")

p_sub <- ggplot(sub_long, aes(subgroup, expr, fill = subgroup)) +
  geom_boxplot(alpha = 0.8, outlier.size = 0.8, width = 0.55) +
  geom_jitter(width = 0.12, size = 0.7, alpha = 0.6, colour = "grey30") +
  scale_fill_manual(values = sub_cols, guide = "none") +
  facet_wrap(~ gene, scales = "free_y", nrow = 2) +
  stat_compare_means(method = "wilcox.test", label = "p.format",
                      size = 3, label.y.npc = 0.95) +
  labs(x = NULL, y = "normalised expression",
       title = "Immune markers: invasive-like (n=16) vs DCIS-like (n=30)",
       subtitle = "GSE59246 | Mann-Whitney U | all p > 0.05") +
  theme_classic(base_size = 10) +
  theme(strip.text = element_text(face = "bold.italic"),
        strip.background = element_rect(fill = "#F0F0F0", colour = NA))

ggsave("figures/immune_invasive_validation.png", p_sub, width = 12, height = 7, dpi = 300)

writeLines(capture.output(sessionInfo()), "sessionInfo_05_immune.txt")
