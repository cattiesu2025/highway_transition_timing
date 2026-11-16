#!/usr/bin/env Rscript
# Unweighted speed vs front-distance score of BAL (0.70, 0.70) policies over the first 100
# evaluation steps, four arms. Bars: mean over seeds (A/B/C 20, D 5); dots: individual seeds.
# Input: outputs/reward_equal_start/score_share/source_data/score_first100_by_seed.csv
#        (experiments/reward_equal_start/score_share_first100.py). R-only exports.
.libPaths(c("tmp/r-lib", .libPaths()))
suppressPackageStartupMessages({library(ggplot2); library(dplyr)})
stopifnot(requireNamespace("ragg", quietly=TRUE), requireNamespace("svglite", quietly=TRUE))
out <- "outputs/reward_equal_start/score_share"
seeds <- read.csv(file.path(out, "source_data/score_first100_by_seed.csv"))
stopifnot(identical(as.integer(table(seeds$arm)[c("A","B","C","D")]), c(20L,20L,20L,5L)))
long <- rbind(transform(seeds, score="S", value=S), transform(seeds, score="D", value=D))[c("arm","seed","score","value")]
bars <- long %>% group_by(arm, score) %>% summarise(top=max(value), value=mean(value), n=n(), .groups="drop")
share <- seeds %>% group_by(arm) %>% summarise(n=n(), share=mean(S)/(mean(S)+mean(D)), .groups="drop")
write.csv(bars, file.path(out, "source_data/score_first100_summary.csv"), row.names=FALSE)
names_arm <- c(A="Single lane", B="Empty target lane", C="Occupied target lane", D="Yellow car farther ahead")
share$label <- sprintf("Arm %s\n%s\nspeed share %.0f%%\n%d seeds", share$arm, names_arm[share$arm], 100*share$share, share$n)
lev <- c(S="speed_score  S", D="front_distance_score  D")
bars$score <- factor(lev[bars$score], lev); long$score <- factor(lev[long$score], lev)
colors <- setNames(c("#DE8C14", "#1A9E6E"), lev)
ink <- "#243746"; muted <- "#596B79"
dodge <- position_dodge(width=.8)
p <- ggplot(bars, aes(arm, value, fill=score)) +
  geom_col(position=dodge, width=.76, colour="white", linewidth=.6) +
  geom_point(data=long, aes(group=score), position=position_jitterdodge(jitter.width=.18, dodge.width=.8, seed=7),
             size=1.3, colour=ink, alpha=.35, show.legend=FALSE) +
  geom_text(aes(y=top, label=sprintf("%.0f", value)), position=dodge, vjust=-.7, size=5, colour=ink) +
  scale_fill_manual(values=colors, name=NULL) +
  scale_x_discrete(labels=setNames(share$label, share$arm)) +
  scale_y_continuous(limits=c(0,110), breaks=seq(0,100,20), expand=c(0,0)) +
  labs(title="Equal weights (0.70, 0.70): speed still contributes less than half",
       subtitle="Development grid, first 100 steps; bars: mean over seeds, dots: individual seeds",
       x=NULL, y="Unweighted score over 100 steps (max 100)",
       caption=paste("A/B/C: the 20 main BAL runs per arm. D: the only (0.70, 0.70) Arm D policies (equal-start ES-BAL, seeds 4200-4204).",
                     "Check: the original Arm D F4 (0.56, 0.84) policies, 19 seeds, score S 66 and D 100 (speed share 40%).", sep="\n")) +
  theme_minimal(base_size=14, base_family="Arial") +
  theme(text=element_text(colour=ink), axis.text=element_text(colour=ink, size=12.5),
        axis.title.y=element_text(colour=muted, size=12.5), panel.grid.major.x=element_blank(),
        panel.grid.minor=element_blank(), panel.grid.major.y=element_line(colour="#E9EDF0"),
        axis.line.x=element_line(colour="#8896A0"), legend.position="top", legend.justification="right",
        legend.text=element_text(size=13), plot.title=element_text(face="bold", size=18),
        plot.subtitle=element_text(colour=muted, size=12.5, margin=margin(b=6)),
        plot.caption=element_text(colour=muted, size=10, hjust=0, lineheight=1.2),
        plot.title.position="plot", plot.caption.position="plot", plot.margin=margin(14,18,10,14))
for (ext in c("png","svg","pdf")) {
  dev <- switch(ext, png=ragg::agg_png, svg=svglite::svglite, pdf=grDevices::cairo_pdf)
  ggsave(file.path(out, paste0("score_share_first100_4arm.", ext)), p, device=dev, width=12, height=7, units="in", dpi=300, bg="white")
}
print(as.data.frame(share[c("arm","n","share")]))
