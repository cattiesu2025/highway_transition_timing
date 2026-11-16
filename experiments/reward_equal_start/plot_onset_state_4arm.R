#!/usr/bin/env Rscript
# Ego speed and lead gap at onset, for one equal-start weight set. Companion to plot_onset_4arm.R:
# same runs, episodes, layout and summary contract (dots = seed medians over 36 development scenes;
# diamond and label = median across seeds; bars = seed range). One figure per quantity.
# Input: outputs/reward_equal_start/<analysis>/episode_onset_state.csv (collect_onset_state.py <set>).
# Usage: Rscript experiments/reward_equal_start/plot_onset_state_4arm.R <t0|first_log>
.libPaths(c("tmp/r-lib", .libPaths()))
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(patchwork)})
stopifnot(requireNamespace("ragg", quietly=TRUE), requireNamespace("svglite", quietly=TRUE))
set <- commandArgs(trailingOnly=TRUE)[1]
stopifnot(set %in% c("t0","first_log"))
S <- list(t0=list(analysis="analysis_t0_onset", out="figures_4arm", tag="equal_start_t0", label="t = 0"),
          first_log=list(analysis="analysis_first_log", out="figures_first_log", tag="equal_start_first_log", label="first-log"))[[set]]
base <- "outputs/reward_equal_start"
state <- read.csv(file.path(base, S$analysis, "episode_onset_state.csv"), stringsAsFactors=FALSE)
status <- read.csv(file.path(base, S$analysis, "run_status.csv"), stringsAsFactors=FALSE)
out <- file.path(base, S$out); dir.create(file.path(out, "source_data"), recursive=TRUE, showWarnings=FALSE)
stopifnot(nrow(status)==60, !anyDuplicated(state[c("arm","variant","seed","episode_id")]),
          nrow(distinct(state, arm, variant, seed))==sum(status$selection_status=="complete"))
failed <- status %>% filter(selection_status!="complete")
weights <- status %>% distinct(arm, variant, speed_weight, front_distance_weight)
ids <- c("FD","BAL","SP"); arms <- c("A","B","C","D")
colors <- c(FD="#1B7FB5", BAL="#C13F8E", SP="#DE8C14")
ink <- "#243746"; muted <- "#596B79"
names_arm <- c(A="Arm A, single lane", B="Arm B, empty target lane", C="Arm C, occupied target lane", D="Arm D, yellow car farther")
endpoint <- c(A="Onset = ego slowdown", B="Onset = first lane change", C="Onset = first lane change", D="Onset = first lane change")
theme_set(theme_classic(base_size=13, base_family="Arial") + theme(
  text=element_text(colour=ink), axis.text=element_text(colour=ink,size=12), axis.title=element_text(size=12),
  axis.line=element_line(colour="#8896A0",linewidth=.4), axis.ticks=element_line(colour="#8896A0",linewidth=.4),
  panel.grid.major.y=element_line(colour="#E9EDF0",linewidth=.4),
  plot.title=element_text(size=13,face="bold",colour=ink,margin=margin(b=4),lineheight=1.15),
  plot.subtitle=element_text(size=11,colour=muted,margin=margin(b=10),lineheight=1.15),
  plot.margin=margin(6,10,6,8), plot.background=element_rect(fill="white",colour=NA), legend.position="none"))

cfg <- list(
  speed=list(col="onset_speed", unit="", digits=1, ylab="Ego speed at onset (m/s)", limits=c(20,36), step=2,
             file=sprintf("onset_speed_%s_4arm", S$tag),
             title=sprintf("Ego speed at onset under %s equal-start weights", S$label),
             def="Speed: ego speed at the onset instant (A: the slowdown decision step; B-D: the step the lane index changes)."),
  gap=list(col="onset_gap", unit="m", digits=0, ylab="Gap to lead vehicle at onset (m)", limits=NULL, step=20,
           file=sprintf("onset_gap_%s_4arm", S$tag),
           title=sprintf("Gap to lead vehicle at onset under %s equal-start weights", S$label),
           def=paste("Gap: bumper-to-bumper longitudinal distance to the lead vehicle in ego's original lane at the onset instant",
                     "(centre distance minus 5 m, highway-env vehicle length).")))

make_fig <- function(c) {
  seeds <- state %>% filter(!is.na(.data[[c$col]])) %>% group_by(arm, variant, seed) %>%
    summarise(n=n(), value=median(.data[[c$col]]), .groups="drop")
  summary <- seeds %>% group_by(arm, variant) %>%
    summarise(mid=median(value), lo=min(value), hi=max(value), n_seeds=n(), episodes=sum(n), .groups="drop")
  write.csv(seeds, file.path(out, "source_data", paste0(c$file, "_by_seed.csv")), row.names=FALSE)
  write.csv(summary, file.path(out, "source_data", paste0(c$file, "_summary.csv")), row.names=FALSE)
  seeds$variant <- factor(seeds$variant, ids); summary$variant <- factor(summary$variant, ids)
  lim <- if (is.null(c$limits)) c(0, ceiling(max(seeds$value)*1.05/c$step)*c$step) else c$limits
  stopifnot(all(seeds$value >= lim[1] & seeds$value <= lim[2]))
  fmtv <- function(v) formatC(v, format="f", digits=c$digits)
  panel <- function(a, letter) {
    d <- filter(seeds, arm==a); s <- filter(summary, arm==a); w <- filter(weights, arm==a)
    fmt <- function(v) with(filter(w, variant==v), sprintf("(%.2f, %.2f)", speed_weight, front_distance_weight))
    labels <- setNames(sprintf("%s\nn = %d", ids, sapply(ids, function(v) sum(s$n_seeds[s$variant==v]))), ids)
    ggplot(d, aes(variant, value, colour=variant)) +
      geom_errorbar(data=s, aes(y=mid, ymin=lo, ymax=hi), width=.14, linewidth=.85) +
      geom_point(position=position_jitter(width=.12, height=0, seed=91), size=2.6, alpha=.6) +
      geom_point(data=s, aes(y=mid), shape=23, size=4.4, fill="white", stroke=1.4) +
      geom_text(data=s, aes(y=mid, label=trimws(paste(fmtv(mid), c$unit))), nudge_x=.25, hjust=0, size=4, fontface="bold", colour=ink) +
      scale_colour_manual(values=colors) + scale_x_discrete(labels=labels, expand=expansion(add=c(.6,.95))) +
      scale_y_continuous(limits=lim, breaks=seq(lim[1], lim[2], c$step), expand=expansion(mult=c(0,.03))) +
      labs(title=sprintf("%s   %s", letter, names_arm[[a]]),
           subtitle=sprintf("%s\nFD %s  SP %s", endpoint[[a]], fmt("FD"), fmt("SP")),
           x=NULL, y=if (a=="A") c$ylab else NULL)
  }
  fail_note <- if (nrow(failed)) paste0("Not shown (trained, but no checkpoint passed model selection): ",
    paste(sprintf("Arm %s %s seed %d", failed$arm, failed$variant, failed$seed), collapse="; "), ".") else "All 60 runs passed model selection."
  fig <- wrap_plots(mapply(panel, arms, letters[1:4], SIMPLIFY=FALSE), nrow=1) +
    plot_annotation(title=c$title,
      subtitle=paste("Same runs, episodes and onsets as the onset-time figure.",
                     "BAL is (0.70, 0.70) in every arm; weights are (speed, spacing)."),
      caption=paste("Dots: seed medians over 36 development scenes; diamonds and labels: median across seeds; bars: range across seeds.",
        c$def,
        sprintf("All shown seeds: %d episodes with an onset. Descriptive only (at most 5 seeds per bar); no test across preferences.",
                sum(summary$episodes)),
        fail_note, sep="\n"),
      theme=theme(plot.title=element_text(size=21,face="bold",colour=ink,margin=margin(b=6),family="Arial"),
        plot.subtitle=element_text(size=12.5,colour=muted,margin=margin(b=12),family="Arial"),
        plot.caption=element_text(size=10.5,hjust=0,colour=muted,lineheight=1.25,family="Arial"),
        plot.margin=margin(16,18,10,18), plot.background=element_rect(fill="white",colour=NA)))
  for (ext in c("png","svg","pdf")) {
    dev <- switch(ext, png=ragg::agg_png, svg=svglite::svglite, pdf=grDevices::cairo_pdf)
    ggsave(file.path(out, paste0(c$file, ".", ext)), fig, device=dev, width=16, height=6.8, units="in", dpi=300, bg="white")
  }
  print(as.data.frame(summary))
}
for (k in names(cfg)) make_fig(cfg[[k]])
