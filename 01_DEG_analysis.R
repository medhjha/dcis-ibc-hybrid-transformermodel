# DEG analysis - DCIS vs IBC
# GSE59246 discovery, GSE26304 + GSE33692 validation
# limma with BH correction

library(GEOquery)
library(affy)
library(limma)
library(ggplot2)
library(ggrepel)
library(pheatmap)
library(dplyr)
library(tidyr)

set.seed(42)
dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)
dir.create("data", showWarnings = FALSE)


# GSE59246 - primary cohort
# 105 samples: IBC=56, DCIS=46, Normal=3

gse59246 <- getGEO("GSE59246", GSEMatrix = TRUE, getGPL = TRUE)[[1]]
pheno59246 <- pData(gse59246)

# find disease state column
disease_col <- grep("disease.state|disease state|tissue",
                    colnames(pheno59246), ignore.case = TRUE, value = TRUE)[1]

pheno59246$disease_state <- trimws(
  gsub("disease state: ", "", pheno59246[[disease_col]], ignore.case = TRUE)
)

# drop Normal samples (n=3), keep IBC and DCIS only
keep <- pheno59246$disease_state %in% c(
  "Invasive breast cancer (IBC)",
  "Ductal carcinoma in situ (DCIS)"
)
gse59246 <- gse59246[, keep]
pheno59246 <- pheno59246[keep, ]

cat("IBC:", sum(grepl("IBC|Invasive", pheno59246$disease_state)), "\n")
cat("DCIS:", sum(grepl("DCIS", pheno59246$disease_state)), "\n")

# expression matrix - Agilent, already normalised
expr <- exprs(gse59246)
feat <- fData(gse59246)

gene_col <- grep("gene.symbol|GENE_SYMBOL|Symbol|gene_assignment",
                  colnames(feat), ignore.case = TRUE, value = TRUE)[1]
feat$gene_symbol <- feat[[gene_col]]

# remove unannotated probes
keep_probes <- !is.na(feat$gene_symbol) & 
               feat$gene_symbol != "" & 
               feat$gene_symbol != "---"
expr <- expr[keep_probes, ]
feat <- feat[keep_probes, ]

# one probe per gene - highest mean expression
feat$row_mean <- rowMeans(expr, na.rm = TRUE)
feat$probe_id <- rownames(feat)

best_probes <- feat %>%
  group_by(gene_symbol) %>%
  slice_max(row_mean, n = 1, with_ties = FALSE) %>%
  pull(probe_id)

expr <- expr[best_probes, ]
rownames(expr) <- feat$gene_symbol[match(best_probes, feat$probe_id)]

cat("genes x samples:", nrow(expr), "x", ncol(expr), "\n")

write.csv(expr, "data/expr_GSE59246_normalised.csv")

pheno_out <- data.frame(
  sample_id = colnames(expr),
  disease_state = pheno59246$disease_state[
    pheno59246$geo_accession %in% colnames(expr)
  ],
  stringsAsFactors = FALSE
)
write.csv(pheno_out, "data/pheno_GSE59246_clean.csv", row.names = FALSE)


# limma - IBC vs DCIS
# DCIS is reference group

group <- factor(
  ifelse(grepl("IBC|Invasive", pheno59246$disease_state, ignore.case = TRUE),
         "IBC", "DCIS"),
  levels = c("DCIS", "IBC")
)

design <- model.matrix(~ group)
colnames(design) <- c("Intercept", "IBC_vs_DCIS")

fit <- lmFit(expr, design)
fit <- eBayes(fit)

res <- topTable(fit, coef = "IBC_vs_DCIS",
                number = Inf, adjust.method = "BH", sort.by = "P")
res$gene <- rownames(res)

# significant DEGs
sig <- res %>% filter(adj.P.Val < 0.05)
cat("DEGs FDR<0.05:", nrow(sig), "\n")
cat("up:", sum(sig$logFC > 0), "down:", sum(sig$logFC < 0), "\n")

top <- sig %>% filter(abs(logFC) > 1)
cat("top DEGs |logFC|>1:", nrow(top), "\n")

write.csv(res, "results/DEG_IBC_vs_DCIS_GSE59246.csv", row.names = FALSE)
write.csv(top, "results/top_DEGs_IBC_vs_DCIS.csv", row.names = FALSE)


# validation cohort 1 - GSE26304
# 115 in GEO, 67 used: IDC=36, DCIS=31

gse26304 <- getGEO("GSE26304", GSEMatrix = TRUE, getGPL = TRUE)[[1]]
pheno26304 <- pData(gse26304)

type_col <- grep("type|tissue|status|histol",
                  colnames(pheno26304), ignore.case = TRUE, value = TRUE)[1]
pheno26304$sample_type <- trimws(pheno26304[[type_col]])

keep26304 <- pheno26304$sample_type %in% c(
  "IDC", "DCIS",
  "invasive ductal carcinoma",
  "ductal carcinoma in situ"
)
gse26304 <- gse26304[, keep26304]
pheno26304 <- pheno26304[keep26304, ]

expr26304 <- exprs(gse26304)
feat26304 <- fData(gse26304)

gene_col26304 <- grep("gene.symbol|Symbol|GENE_SYMBOL",
                       colnames(feat26304), ignore.case = TRUE, value = TRUE)[1]
feat26304$gene_symbol <- feat26304[[gene_col26304]]

keep_p26304 <- !is.na(feat26304$gene_symbol) & feat26304$gene_symbol != "" &
               feat26304$gene_symbol != "---"
expr26304 <- expr26304[keep_p26304, ]
feat26304 <- feat26304[keep_p26304, ]

feat26304$row_mean <- rowMeans(expr26304)
feat26304$probe_id <- rownames(feat26304)
best26304 <- feat26304 %>%
  group_by(gene_symbol) %>%
  slice_max(row_mean, n = 1, with_ties = FALSE) %>%
  pull(probe_id)

expr26304 <- expr26304[best26304, ]
rownames(expr26304) <- feat26304$gene_symbol[match(best26304, feat26304$probe_id)]

group26304 <- factor(
  ifelse(grepl("IDC|invasive", pheno26304$sample_type, ignore.case = TRUE),
         "IDC", "DCIS"),
  levels = c("DCIS", "IDC")
)
design26304 <- model.matrix(~ group26304)
fit26304 <- eBayes(lmFit(expr26304, design26304))
res26304 <- topTable(fit26304, coef = 2, number = Inf, sort.by = "P")
res26304$gene <- rownames(res26304)

write.csv(res26304, "results/DEG_IDC_vs_DCIS_GSE26304.csv", row.names = FALSE)


# validation cohort 2 - GSE33692
# 45 in GEO, 20 used: IDC=10, DCIS=10, epithelial only

gse33692 <- getGEO("GSE33692", GSEMatrix = TRUE, getGPL = TRUE)[[1]]
pheno33692 <- pData(gse33692)

type_col33692 <- grep("type|tissue|cell.type|compartment",
                       colnames(pheno33692), ignore.case = TRUE, value = TRUE)[1]
pheno33692$sample_type <- trimws(pheno33692[[type_col33692]])

# keep epithelial compartment only, exclude stromal
keep33692 <- grepl("epithelium|epithelial|IDC|DCIS",
                    pheno33692$sample_type, ignore.case = TRUE) &
             !grepl("stroma|stromal|fibroblast",
                    pheno33692$sample_type, ignore.case = TRUE)

gse33692 <- gse33692[, keep33692]
pheno33692 <- pheno33692[keep33692, ]

expr33692 <- exprs(gse33692)
feat33692 <- fData(gse33692)

gene_col33692 <- grep("gene.symbol|Symbol|GENE_SYMBOL|gene_assignment",
                       colnames(feat33692), ignore.case = TRUE, value = TRUE)[1]
feat33692$gene_symbol <- feat33692[[gene_col33692]]

keep_p33692 <- !is.na(feat33692$gene_symbol) & feat33692$gene_symbol != "" &
               feat33692$gene_symbol != "---"
expr33692 <- expr33692[keep_p33692, ]
feat33692 <- feat33692[keep_p33692, ]

feat33692$row_mean <- rowMeans(expr33692)
feat33692$probe_id <- rownames(feat33692)
best33692 <- feat33692 %>%
  group_by(gene_symbol) %>%
  slice_max(row_mean, n = 1, with_ties = FALSE) %>%
  pull(probe_id)

expr33692 <- expr33692[best33692, ]
rownames(expr33692) <- feat33692$gene_symbol[match(best33692, feat33692$probe_id)]

group33692 <- factor(
  ifelse(grepl("IDC|invasive", pheno33692$sample_type, ignore.case = TRUE),
         "IDC", "DCIS"),
  levels = c("DCIS", "IDC")
)
design33692 <- model.matrix(~ group33692)
fit33692 <- eBayes(lmFit(expr33692, design33692))
res33692 <- topTable(fit33692, coef = 2, number = Inf, sort.by = "P")
res33692$gene <- rownames(res33692)

write.csv(res33692, "results/DEG_IDC_vs_DCIS_GSE33692_epithelium.csv", row.names = FALSE)


# cross-cohort concordance

top_genes <- top$gene

common_v1 <- intersect(top_genes, res26304$gene)
conc_v1 <- mean(
  sign(top$logFC[top$gene %in% common_v1]) ==
  sign(res26304$logFC[res26304$gene %in% common_v1]),
  na.rm = TRUE
)
cat(sprintf("GSE26304 concordance: %.1f%%\n", conc_v1 * 100))

common_v2 <- intersect(top_genes, res33692$gene)
conc_v2 <- mean(
  sign(top$logFC[top$gene %in% common_v2]) ==
  sign(res33692$logFC[res33692$gene %in% common_v2]),
  na.rm = TRUE
)
cat(sprintf("GSE33692 concordance: %.1f%%\n", conc_v2 * 100))

# genes concordant in both validation cohorts
both <- intersect(common_v1, common_v2)
disc_dir <- sign(top$logFC[top$gene %in% both])
v1_dir   <- sign(res26304$logFC[res26304$gene %in% both])
v2_dir   <- sign(res33692$logFC[res33692$gene %in% both])

validated <- both[(disc_dir == v1_dir) & (disc_dir == v2_dir)]
cat("validated genes (both cohorts):", length(validated), "\n")

consensus <- top %>%
  filter(gene %in% validated) %>%
  select(gene, logFC, adj.P.Val) %>%
  rename(logFC_disc = logFC, FDR_disc = adj.P.Val) %>%
  left_join(
    res26304 %>% filter(gene %in% validated) %>%
      select(gene, logFC, adj.P.Val) %>%
      rename(logFC_v1 = logFC, FDR_v1 = adj.P.Val),
    by = "gene"
  ) %>%
  left_join(
    res33692 %>% filter(gene %in% validated) %>%
      select(gene, logFC, adj.P.Val) %>%
      rename(logFC_v2 = logFC, FDR_v2 = adj.P.Val),
    by = "gene"
  ) %>%
  mutate(direction = ifelse(logFC_disc > 0, "Up_in_IBC", "Down_in_IBC"))

write.csv(consensus, "results/consensus_DEGs_validated.csv", row.names = FALSE)

# 41-gene core - |logFC|>1 in all three cohorts
core41 <- consensus %>%
  filter(
    abs(logFC_disc) > 1,
    abs(logFC_v1) > 1,
    abs(logFC_v2) > 1,
    !is.na(logFC_v1),
    !is.na(logFC_v2)
  )

cat("core signature genes:", nrow(core41), "\n")
cat("up:", sum(core41$direction == "Up_in_IBC"),
    "down:", sum(core41$direction == "Down_in_IBC"), "\n")

write.csv(core41, "results/core_41gene_signature.csv", row.names = FALSE)


# figures

# volcano plot
plot_df <- res %>%
  mutate(
    sig_group = case_when(
      adj.P.Val < 0.05 & logFC >  1 ~ "Up",
      adj.P.Val < 0.05 & logFC < -1 ~ "Down",
      TRUE ~ "NS"
    ),
    neg_log10p = -log10(adj.P.Val),
    label = ifelse(gene %in% core41$gene[1:15], gene, NA)
  )

p_volcano <- ggplot(plot_df, aes(logFC, neg_log10p, colour = sig_group, label = label)) +
  geom_point(alpha = 0.6, size = 0.8) +
  scale_colour_manual(values = c("Up" = "#E8554E", "Down" = "#2E75B6", "NS" = "#CCCCCC"),
                       guide = "none") +
  geom_text_repel(colour = "black", fontface = "italic", size = 2.8,
                   max.overlaps = 20, segment.size = 0.3, na.rm = TRUE) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", colour = "grey50", linewidth = 0.4) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", colour = "grey50", linewidth = 0.4) +
  labs(x = expression(log[2]~FC~(IBC~vs~DCIS)),
       y = expression(-log[10]~FDR),
       title = "IBC vs DCIS differential expression (GSE59246)") +
  theme_classic(base_size = 11)

ggsave("figures/volcano_IBC_vs_DCIS_GSE59246.png", p_volcano, width = 8, height = 7, dpi = 300)

# heatmap top 50 DEGs
top50 <- top %>%
  filter(gene %in% rownames(expr)) %>%
  arrange(desc(abs(logFC))) %>%
  head(50) %>%
  pull(gene)

mat <- t(scale(t(expr[top50, ])))

anno <- data.frame(
  Group = ifelse(grepl("IBC|Invasive", pheno59246$disease_state, ignore.case = TRUE),
                 "IBC", "DCIS"),
  row.names = colnames(mat)
)

png("figures/heatmap_top50DEGs_IBC_vs_DCIS.png", width = 3000, height = 4500, res = 300)
pheatmap(mat,
         annotation_col = anno,
         annotation_colors = list(Group = c(IBC = "#E8554E", DCIS = "#2E75B6")),
         color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
         cluster_rows = TRUE, cluster_cols = TRUE,
         show_colnames = FALSE, fontsize_row = 7, border_color = NA,
         main = "Top 50 DEGs: IBC vs DCIS (GSE59246)")
dev.off()

writeLines(capture.output(sessionInfo()), "sessionInfo_01_DEG.txt")
