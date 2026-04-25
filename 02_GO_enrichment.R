# GO and KEGG enrichment - upregulated and downregulated DEGs
# clusterProfiler, BH correction, FDR < 0.05

library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)
library(dplyr)

set.seed(42)
dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

degs <- read.csv("results/DEG_IBC_vs_DCIS_GSE59246.csv", stringsAsFactors = FALSE)

sig <- degs %>% filter(adj.P.Val < 0.05, abs(logFC) > 1)
up_genes   <- sig %>% filter(logFC > 0) %>% pull(gene)
down_genes <- sig %>% filter(logFC < 0) %>% pull(gene)
background <- degs$gene

cat("up:", length(up_genes), "down:", length(down_genes), "\n")

# convert to entrez IDs
up_entrez   <- bitr(up_genes,   fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
down_entrez <- bitr(down_genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
bg_entrez   <- bitr(background, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)


# GO BP enrichment

go_up <- enrichGO(
  gene          = up_entrez$ENTREZID,
  universe      = bg_entrez$ENTREZID,
  OrgDb         = org.Hs.eg.db,
  ont           = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.05,
  minGSSize     = 5,
  maxGSSize     = 500,
  readable      = TRUE
)
cat("GO BP terms (up):", nrow(as.data.frame(go_up)), "\n")
write.csv(as.data.frame(go_up), "results/GO_BP_up_in_IBC.csv", row.names = FALSE)

go_down <- enrichGO(
  gene          = down_entrez$ENTREZID,
  universe      = bg_entrez$ENTREZID,
  OrgDb         = org.Hs.eg.db,
  ont           = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.05,
  minGSSize     = 5,
  maxGSSize     = 500,
  readable      = TRUE
)
cat("GO BP terms (down):", nrow(as.data.frame(go_down)), "\n")
write.csv(as.data.frame(go_down), "results/GO_BP_down_in_IBC.csv", row.names = FALSE)


# KEGG enrichment

kegg_up <- enrichKEGG(
  gene          = up_entrez$ENTREZID,
  organism      = "hsa",
  universe      = bg_entrez$ENTREZID,
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  minGSSize     = 5
)
cat("KEGG pathways (up):", nrow(as.data.frame(kegg_up)), "\n")
write.csv(as.data.frame(kegg_up), "results/KEGG_up_in_IBC.csv", row.names = FALSE)

# check TGF-beta specifically
tgfb <- as.data.frame(kegg_up) %>% filter(grepl("TGF", Description, ignore.case = TRUE))
if (nrow(tgfb) > 0) {
  cat("TGF-beta FDR:", tgfb$p.adjust[1], "\n")
} else {
  # check at relaxed threshold
  kegg_relaxed <- enrichKEGG(
    gene = up_entrez$ENTREZID, organism = "hsa",
    pAdjustMethod = "BH", pvalueCutoff = 1.0, minGSSize = 5
  )
  tgfb_nom <- as.data.frame(kegg_relaxed) %>%
    filter(grepl("TGF", Description, ignore.case = TRUE))
  cat("TGF-beta (nominal):", tgfb_nom$pvalue[1], "FDR:", tgfb_nom$p.adjust[1], "\n")
}


# dot plot function

dotplot_go <- function(go_result, title, top_n = 15, col_low = "#2E75B6", col_high = "#E8554E") {
  df <- as.data.frame(go_result) %>%
    arrange(p.adjust) %>%
    head(top_n) %>%
    mutate(
      Description = factor(Description, levels = rev(Description)),
      gr = sapply(GeneRatio, function(x) {
        v <- as.numeric(strsplit(x, "/")[[1]])
        v[1] / v[2]
      })
    )

  ggplot(df, aes(gr, Description, size = Count, colour = p.adjust)) +
    geom_point() +
    scale_colour_gradient(low = col_low, high = col_high,
                           name = "FDR", trans = "log10",
                           labels = scales::scientific) +
    scale_size_continuous(name = "gene count", range = c(3, 10)) +
    labs(x = "gene ratio", y = NULL, title = title) +
    theme_classic(base_size = 11) +
    theme(axis.text.y = element_text(size = 9))
}

p_up <- dotplot_go(go_up, "GO BP: upregulated in IBC vs DCIS")
ggsave("figures/GO_BP_up_in_IBC.png", p_up, width = 10, height = 7, dpi = 300)

p_down <- dotplot_go(go_down, "GO BP: downregulated in IBC vs DCIS",
                      col_low = "#E8554E", col_high = "#2E75B6")
ggsave("figures/GO_BP_down_in_IBC.png", p_down, width = 10, height = 7, dpi = 300)

writeLines(capture.output(sessionInfo()), "sessionInfo_02_GO.txt")
