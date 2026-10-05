#!/usr/bin/env Rscript
# Contract: describe training trajectories under alternative reward weights;
# no claim of convergence or improved driving is inferred from reward height.
# Quantitative 3x3 grid, paired training seeds, median and IQR (not a CI).
# R-only rendering; 260x220 mm seminar figure, editable PDF/SVG + PNG/TIFF.
# Source: checked development logs and per-seed physical-onset summaries.
local_lib <- file.path("tmp", "r-lib")
if (dir.exists(local_lib)) .libPaths(c(normalizePath(local_lib), .libPaths()))
dir.create("tmp/r-cache", recursive=TRUE, showWarnings=FALSE)
Sys.setenv(XDG_CACHE_HOME=normalizePath("tmp/r-cache"))
required <- c("ggplot2", "dplyr", "patchwork", "svglite", "ragg")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly=TRUE)]
if (length(missing)) stop("Missing R packages: ", paste(missing, collapse=", "))
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(patchwork)})
args <- commandArgs(trailingOnly=TRUE)
preview <- "--allow-incomplete" %in% args
args <- args[args != "--allow-incomplete"]
source_dir <- if (length(args)) args[1] else "outputs/reward_sensitivity/analysis"
out <- if (length(args)>1) args[2] else "outputs/reward_sensitivity/figures"
dir.create(out, recursive=TRUE, showWarnings=FALSE)
raw <- read.csv(file.path(source_dir, "training_progress.csv"))
status <- read.csv(file.path(source_dir, "run_status.csv"))
complete <- sum(status$status == "complete")
if (complete != 135 && !preview) stop("Incomplete sweep. Use --allow-incomplete for a labelled preview.")
if (!nrow(raw)) stop("No training records available")
# Interpolate only between observed log points: never extend plateaus to 100K.
traces <- split(raw, interaction(raw$arm, raw$variant, raw$seed, drop=TRUE))
aligned <- bind_rows(lapply(traces, function(d) {
  d <- d[order(d$timesteps), ]
  grid <- seq(1000, 99000, by=1000)
  values <- approx(d$timesteps, d$episode_reward, xout=grid, rule=1)$y
  data.frame(arm=d$arm[1], variant=d$variant[1], agent=d$agent[1], seed=d$seed[1],
             speed_weight=d$speed_weight[1], front_distance_weight=d$front_distance_weight[1],
             timesteps=grid, episode_reward=values)
}))
summary <- aligned %>% group_by(arm, agent, variant, speed_weight, front_distance_weight, timesteps) %>%
  summarise(n_seeds=sum(is.finite(episode_reward)), median=median(episode_reward, na.rm=TRUE),
            q1=quantile(episode_reward,.25,na.rm=TRUE), q3=quantile(episode_reward,.75,na.rm=TRUE), .groups="drop") %>%
  filter(n_seeds==5)
if (!nrow(summary)) stop("No time points with five paired seeds")
write.csv(aligned, file.path(out,"training_interpolated.csv"), row.names=FALSE)
write.csv(summary, file.path(out,"training_summary.csv"), row.names=FALSE)
cols <- c("#56616B", "#3D8398", "#C87825")
arms <- c(A="A | Single lane: slowdown", B="B | Empty target lane: lane change", C="C | Occupied target lane: lane change")
groups <- list(FD=c("FD","R1","R2"), BAL=c("BAL","R3","R4"), SP=c("SP","R5","R6"))
titles <- c(FD="Spacing-weighted", BAL="Near equal weights", SP="Speed-weighted")
panels <- list()
for (arm in names(arms)) for (agent in names(groups)) {
  d <- summary[summary$arm==arm & summary$agent==agent, ]
  ids <- groups[[agent]]
  labels <- vapply(ids, function(id) {
    row <- status[status$arm==arm & status$variant==id, ][1, ]
    sprintf("%s (%.2f, %.2f)", id, row$speed_weight, row$front_distance_weight)
  }, character(1))
  panels[[length(panels)+1]] <- ggplot(d, aes(timesteps/1000, median, colour=variant, fill=variant, linetype=variant)) +
    geom_ribbon(aes(ymin=q1,ymax=q3), alpha=.14, colour=NA, linetype=0) +
    geom_line(linewidth=.55) + geom_vline(xintercept=100, linetype="dotted", colour="grey50") +
    scale_colour_manual(values=setNames(cols,ids), breaks=ids, labels=labels, drop=FALSE) +
    scale_fill_manual(values=setNames(cols,ids), breaks=ids, labels=labels, drop=FALSE) +
    scale_linetype_manual(values=setNames(c("solid","longdash","dotted"),ids), breaks=ids, labels=labels, drop=FALSE) +
    scale_x_continuous(limits=c(0,102), breaks=c(0,25,50,75,100)) +
    labs(title=titles[[agent]], subtitle=arms[[arm]], x="Training steps (x 1000)", y="Mean episode reward", colour=NULL, fill=NULL, linetype=NULL) +
    theme_classic(base_size=8, base_family="sans") +
    theme(legend.position="bottom", legend.text=element_text(size=6.5),
          legend.key.width=grid::unit(8,"mm"), panel.grid.major.y=element_line(colour="#E6E9EC",linewidth=.25),
          plot.title=element_text(face="bold"), plot.subtitle=element_text(size=7)) +
    guides(colour=guide_legend(ncol=1),fill=guide_legend(ncol=1),linetype=guide_legend(ncol=1))
}
title <- if (complete==135) "Training trajectories across reward coefficients" else "PREVIEW: baseline only or incomplete coefficient sweep"
p <- wrap_plots(panels,ncol=3) + plot_annotation(title=title,
  subtitle=sprintf("%d/135 run-condition records | 5 seeds per displayed curve | median and IQR | legend: (speed, spacing)",complete),
  caption="Return uses each configuration's own reward; heights are not a driving-performance ranking. No extrapolation to 100K.")
save_plot <- function(p,name,w,h) {
  ggsave(file.path(out,paste0(name,".pdf")),p,device=grDevices::cairo_pdf,width=w,height=h,units="mm")
  ggsave(file.path(out,paste0(name,".svg")),p,device=svglite::svglite,width=w,height=h,units="mm")
  ggsave(file.path(out,paste0(name,".png")),p,device=ragg::agg_png,width=w,height=h,units="mm",dpi=180)
  ggsave(file.path(out,paste0(name,".tiff")),p,device=ragg::agg_tiff,width=w,height=h,units="mm",dpi=600,compression="lzw")
}
save_plot(p,"training_convergence",260,220)
seed <- read.csv(file.path(source_dir,"seed_summary.csv"))
metrics <- c("mean_episode_speed_mps", "collision_rate", "onset_rate", "conditional_median_onset_seconds")
long <- bind_rows(lapply(metrics,function(m) data.frame(arm=seed$arm,variant=seed$variant,seed=seed$seed,metric=m,value=seed[[m]])))
long$variant <- factor(long$variant,levels=c("FD","R1","R2","BAL","R3","R4","SP","R5","R6"))
metric_labels <- c(mean_episode_speed_mps="Mean speed (m/s)", collision_rate="Collision rate",
                   onset_rate="Onset occurrence", conditional_median_onset_seconds="Conditional onset (s)")
long$metric <- factor(long$metric, levels=metrics)
bounds <- expand.grid(arm=c("A","B","C"), metric=c("collision_rate","onset_rate"), value=c(0,1))
bounds$variant <- factor("FD",levels=levels(long$variant))
b <- ggplot(long,aes(variant,value)) + geom_point(position=position_jitter(width=.12,height=0,seed=20261006),size=1.1,alpha=.65,colour="#3D8398") +
  stat_summary(fun=median,geom="point",shape=95,size=5,na.rm=TRUE) + geom_blank(data=bounds) + facet_grid(metric~arm,scales="free_y", labeller=labeller(metric=metric_labels)) +
  labs(title=if(complete==135) "Development behavior across reward coefficients" else "PREVIEW: incomplete development comparison",
       subtitle="Each point is one training seed; bar is median across seeds",x="Reward configuration",y=NULL,
       caption="Onset: persistent slowdown in A; physical lane change in B/C. Latency is conditional on observed onset; inspect occurrence alongside it.") +
  theme_classic(base_size=8) + theme(axis.text.x=element_text(angle=45,hjust=1),strip.text=element_text(size=7))
save_plot(b,"development_behavior",260,220)
writeLines(c("Exploratory development comparison; no tests or confidence intervals.",
             "IQR describes five independent training seeds; episodes are not independent replicates.",
             "Training return is the logged SB3 episode-reward mean, interpolated within observed support only.",
             paste("Complete run-condition records:",complete,"of 135."),
             "No fabricated new-coefficient curves. Missing records remain absent and previews are labelled."),file.path(out,"figure_notes.txt"))
