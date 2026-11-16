#!/usr/bin/env Rscript
# 20-seed onset-time comparison of SP/BAL/FD in four arms, for one equal-start weight set.
# Usage: Rscript plot_onset_20seed.R <es|fl>
# Contract: dots are seed medians over 36 development scenes; diamond and label are the median across seeds;
# bars are the 95% percentile bootstrap CI of that median (resampling seeds, 5,000 draws, fixed RNG seed).
# Onset: A = ego slowdown; B/C/D = first physical lane change. Only seeds with a selected, eligible
# checkpoint have onsets. Descriptive only: no hypothesis test. Input: collect_20seed.py.
.libPaths(c("tmp/r-lib", .libPaths()))
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(patchwork)})
stopifnot(requireNamespace("ragg", quietly=TRUE), requireNamespace("svglite", quietly=TRUE))
set <- commandArgs(trailingOnly=TRUE)[1]
stopifnot(set %in% c("es","fl"))
cfg <- list(
  es=list(file="onset_20seed_t0", title="Onset times under t = 0 equal-start weights, 20 seeds",
          rule="SP/FD weights equalise the expected return of uniform-random driving (2,000 episodes per arm) with BAL."),
  fl=list(file="onset_20seed_first_log", title="Onset times under first-log equal-start weights, 20 seeds",
          rule="SP/FD weights equalise the 20-seed mean of the first logged training reward with BAL."))[[set]]
base <- file.path("outputs/reward_equal_start/analysis_20seed", set)
episodes <- read.csv(file.path(base, "episode_onsets.csv"), stringsAsFactors=FALSE)
status <- read.csv(file.path(base, "run_status.csv"), stringsAsFactors=FALSE)
out <- "outputs/reward_equal_start/figures_20seed"; dir.create(file.path(out, "source_data"), recursive=TRUE, showWarnings=FALSE)
ids <- c("FD","BAL","SP"); arms <- c("A","B","C","D")
stopifnot(nrow(status)==240, !anyDuplicated(episodes[c("arm","variant","seed","episode_id")]))
cover <- episodes %>% count(arm, variant, seed)
stopifnot(all(cover$n==36), nrow(cover)==sum(status$selection_status=="complete"))
failed <- status %>% filter(selection_status!="complete")

seeds <- episodes %>% group_by(arm, variant, seed) %>%
  summarise(n=n(), n_onset=sum(episode_outcome=="valid_onset"), n_collision=sum(tolower(collision)=="true"),
            seconds=median(onset_seconds[episode_outcome=="valid_onset"]), .groups="drop")
stopifnot(all(is.finite(seeds$seconds)))
set.seed(20261114)
boot_ci <- function(x, B=5000) { m <- replicate(B, median(sample(x, replace=TRUE))); unname(quantile(m, c(.025,.975))) }
summary <- seeds %>% group_by(arm, variant) %>%
  summarise(mid=median(seconds), lo=boot_ci(seconds)[1], hi=boot_ci(seconds)[2],
            q1=quantile(seconds,.25), q3=quantile(seconds,.75), min=min(seconds), max=max(seconds), n_seeds=n(),
            onset_episodes=sum(n_onset), episodes=sum(n), collisions=sum(n_collision), .groups="drop")
write.csv(seeds, file.path(out, "source_data", paste0(cfg$file, "_by_seed.csv")), row.names=FALSE)
write.csv(summary, file.path(out, "source_data", paste0(cfg$file, "_summary.csv")), row.names=FALSE)
seeds$variant <- factor(seeds$variant, ids); summary$variant <- factor(summary$variant, ids)

weights <- status %>% distinct(arm, variant, speed_weight, front_distance_weight)
colors <- c(FD="#1B7FB5", BAL="#C13F8E", SP="#DE8C14")  # validated categorical palette
ink <- "#243746"; muted <- "#596B79"
names_arm <- c(A="Arm A, single lane", B="Arm B, empty target lane", C="Arm C, occupied target lane", D="Arm D, yellow car farther")
endpoint <- c(A="Onset = ego slowdown", B="Onset = first lane change", C="Onset = first lane change", D="Onset = first lane change")
ymax <- max(10, ceiling(max(seeds$seconds)/2)*2)
theme_set(theme_classic(base_size=13, base_family="Arial") + theme(
  text=element_text(colour=ink), axis.text=element_text(colour=ink,size=12), axis.title=element_text(size=12),
  axis.line=element_line(colour="#8896A0",linewidth=.4), axis.ticks=element_line(colour="#8896A0",linewidth=.4),
  panel.grid.major.y=element_line(colour="#E9EDF0",linewidth=.4),
  plot.title=element_text(size=13,face="bold",colour=ink,margin=margin(b=4),lineheight=1.15),
  plot.subtitle=element_text(size=11,colour=muted,margin=margin(b=10),lineheight=1.15),
  plot.margin=margin(6,10,6,8), plot.background=element_rect(fill="white",colour=NA), legend.position="none"))
panel <- function(a, letter) {
  d <- filter(seeds, arm==a); s <- filter(summary, arm==a)
  w <- filter(weights, arm==a)
  fmt <- function(v) with(filter(w, variant==v), sprintf("(%.2f, %.2f)", speed_weight, front_distance_weight))
  labels <- setNames(sprintf("%s\nn = %d", ids, sapply(ids, function(v) sum(s$n_seeds[s$variant==v]))), ids)
  ggplot(d, aes(variant, seconds, colour=variant)) +
    geom_point(position=position_jitter(width=.14, height=0, seed=91), size=2, alpha=.45) +
    geom_errorbar(data=s, aes(y=mid, ymin=lo, ymax=hi), width=.16, linewidth=.95) +
    geom_point(data=s, aes(y=mid), shape=23, size=4.4, fill="white", stroke=1.4) +
    geom_text(data=s, aes(y=mid, label=sprintf("%.1f s", mid)), nudge_x=.27, hjust=0, size=4, fontface="bold", colour=ink) +
    scale_colour_manual(values=colors) + scale_x_discrete(labels=labels) +
    scale_y_continuous(limits=c(0,ymax), breaks=seq(0,ymax,2), expand=expansion(mult=c(0,.03))) +
    labs(title=sprintf("%s   %s", letter, names_arm[[a]]),
         subtitle=sprintf("%s\nFD %s  SP %s", endpoint[[a]], fmt("FD"), fmt("SP")),
         x=NULL, y=if (a=="A") "Onset time (s)" else NULL)
}
fail_note <- if (nrow(failed)) paste0("Not shown (trained, but no checkpoint passed model selection): ",
  paste(sprintf("Arm %s %s seed %d", failed$arm, failed$variant, failed$seed), collapse="; "), ".") else "All 240 runs passed model selection."
fig <- wrap_plots(mapply(panel, arms, letters[1:4], SIMPLIFY=FALSE), nrow=1) +
  plot_annotation(title=cfg$title,
    subtitle=paste(cfg$rule, "BAL is (0.70, 0.70) in every arm; weights are (speed, spacing)."),
    caption=paste("Dots: seed medians over 36 development scenes; diamonds and labels: median across seeds; bars: 95% bootstrap CI of that median (seeds resampled).",
      sprintf("All shown seeds: %d of %d episodes with an onset, %d collisions. Descriptive only; no test across preferences.",
              sum(summary$onset_episodes), sum(summary$episodes), sum(summary$collisions)),
      fail_note, sep="\n"),
    theme=theme(plot.title=element_text(size=21,face="bold",colour=ink,margin=margin(b=6),family="Arial"),
      plot.subtitle=element_text(size=12.5,colour=muted,margin=margin(b=12),family="Arial"),
      plot.caption=element_text(size=10.5,hjust=0,colour=muted,lineheight=1.25,family="Arial"),
      plot.margin=margin(16,18,10,18), plot.background=element_rect(fill="white",colour=NA)))
for (ext in c("png","svg","pdf")) {
  dev <- switch(ext, png=ragg::agg_png, svg=svglite::svglite, pdf=grDevices::cairo_pdf)
  ggsave(file.path(out, paste0(cfg$file, ".", ext)), fig, device=dev, width=16, height=6.5, units="in", dpi=300, bg="white")
}
print(as.data.frame(summary %>% select(arm, variant, n_seeds, mid, lo, hi, q1, q3, onset_episodes, episodes, collisions)))
