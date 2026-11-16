#!/usr/bin/env Rscript
# Equal-start SP/BAL/FD training curves for all four arms. R-only, 16:9, SVG/PDF + 300-dpi PNG.
# Contract: x = 0 is the computed random-policy return, identical for SP/BAL/FD within an arm by
# construction; logged curves start at 400 steps (first SB3 log, 282-400 steps). Lines: median of 5 paired seeds; ribbons: IQR.
# Training curves include every trained seed, independent of model selection (Arm D SP 4204 trained
# fully but no checkpoint passed the selection gate). Heights after t = 0 come from different reward
# functions and do not rank driving performance.
.libPaths(c("tmp/r-lib", .libPaths()))
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(patchwork); library(jsonlite)})
stopifnot(requireNamespace("ragg", quietly=TRUE), requireNamespace("svglite", quietly=TRUE))
base <- "outputs/reward_equal_start"
out <- file.path(base, "figures_4arm"); dir.create(file.path(out, "source_data"), recursive=TRUE, showWarnings=FALSE)
ids <- c("FD","BAL","SP"); arms <- c("A","B","C","D")
keep <- c("arm","variant","seed","timesteps","episode_reward")
trace <- rbind(read.csv(file.path(base,"analysis/training_progress.csv"))[keep],
               read.csv(file.path(base,"analysis_bd/training_progress.csv"))[keep])
status_ac <- read.csv(file.path(base,"analysis/run_status.csv")); status_bd <- read.csv(file.path(base,"analysis_bd/run_status.csv"))
stopifnot(nrow(status_ac)==30, all(status_ac$status=="complete"), nrow(status_bd)==30, all(status_bd$final_timesteps>=99000))
runs <- unique(trace[c("arm","variant","seed")])
stopifnot(nrow(runs)==60, all(table(runs$arm, runs$variant)==5))

# t = 0: A/C from the original reference; B/D from their own reference with the frozen 6-decimal weights.
ref_ac <- fromJSON(file.path(base,"equal_start_weights.json"))$arms
ref_bd <- fromJSON(file.path(base,"equal_start_t0_BD.json"))
weights <- function(a) if (a %in% c("A","C")) ref_ac[[a]]$weights else ref_bd[[a]]$weights
t0 <- bind_rows(lapply(arms, function(a) {
  r <- if (a %in% c("A","C")) unlist(ref_ac[[a]]$t0_return) else unlist(ref_bd[[a]]$t0_return)
  stopifnot(diff(range(r)) < 1e-3)  # Equal start holds to weight rounding.
  data.frame(arm=a, t0=unname(r[["BAL"]]))
}))

grid <- c(400, seq(500,9500,500), seq(10000,99000,1000))  # 400: every run has logged by then (first logs 282-400 steps)
aligned <- bind_rows(lapply(split(trace, interaction(trace$arm,trace$variant,trace$seed,drop=TRUE)), function(d) {
  d <- d[order(d$timesteps),]
  data.frame(arm=d$arm[1], variant=d$variant[1], seed=d$seed[1], timesteps=grid,
             reward=approx(d$timesteps,d$episode_reward,xout=grid,rule=1)$y)
}))
curves <- aligned %>% group_by(arm,variant,timesteps) %>%
  summarise(n=sum(is.finite(reward)), mid=median(reward), q1=quantile(reward,.25), q3=quantile(reward,.75), .groups="drop")
stopifnot(all(curves$n==5))
write.csv(curves, file.path(out,"source_data","training_median_iqr.csv"), row.names=FALSE)
write.csv(aligned, file.path(out,"source_data","training_aligned_by_seed.csv"), row.names=FALSE)
write.csv(t0, file.path(out,"source_data","t0_reference.csv"), row.names=FALSE)
curves$variant <- factor(curves$variant, levels=ids)

colors <- c(FD="#1B7FB5", BAL="#C13F8E", SP="#DE8C14")  # validated categorical palette
lines <- c(FD="solid", BAL="longdash", SP="dotdash")
ink <- "#243746"; muted <- "#596B79"
names_arm <- c(A="Arm A, single lane", B="Arm B, empty target lane", C="Arm C, occupied target lane", D="Arm D, yellow car farther")
fmt <- function(w) sprintf("(%.2f, %.2f)", w$speed, w$spacing)
arm_title <- function(a, letter) { w <- weights(a)
  sprintf("%s   %s\nFD %s  SP %s", letter, names_arm[[a]], fmt(w$FD), fmt(w$SP)) }
spread <- function(y, gap) { o <- order(y); v <- y[o]; for (i in 2:length(v)) v[i] <- max(v[i], v[i-1]+gap); v <- v-(mean(v)-mean(y)); r <- y; r[o] <- v; r }
theme_set(theme_classic(base_size=13, base_family="Arial") + theme(
  text=element_text(colour=ink), axis.text=element_text(colour=ink,size=11), axis.title=element_text(size=12),
  axis.line=element_line(colour="#8896A0",linewidth=.4), axis.ticks=element_line(colour="#8896A0",linewidth=.4),
  panel.grid.major.y=element_line(colour="#E9EDF0",linewidth=.4),
  plot.title=element_text(size=12,face="bold",colour=ink,margin=margin(b=6),lineheight=1.15),
  plot.margin=margin(6,10,6,8), plot.background=element_rect(fill="white",colour=NA), legend.position="none"))

layers <- function(d, start, size) list(
  geom_ribbon(aes(ymin=q1,ymax=q3),alpha=.14,colour=NA,linetype=0),
  geom_segment(data=d %>% group_by(variant) %>% filter(timesteps==min(timesteps)) %>% ungroup() %>% mutate(x0=0,y0=start$t0),
               aes(x=x0,y=y0,xend=timesteps/1000,yend=mid),linetype="dotted",linewidth=.5),
  geom_line(linewidth=.85),
  geom_point(data=start,aes(0,t0),inherit.aes=FALSE,size=size,shape=21,fill=ink,colour="white",stroke=.9),
  scale_colour_manual(values=colors), scale_fill_manual(values=colors), scale_linetype_manual(values=lines))
full_panel <- function(a, letter) {
  d <- filter(curves,arm==a); start <- filter(t0,arm==a)
  ends <- d %>% filter(timesteps==max(timesteps)) %>% mutate(label_y=spread(mid,8))
  ggplot(d,aes(timesteps/1000,mid,colour=variant,fill=variant,linetype=variant)) + layers(d,start,3) +
    geom_text(data=ends,aes(x=101,y=label_y,label=variant),hjust=0,size=3.8,colour=ink,fontface="bold",show.legend=FALSE) +
    scale_x_continuous(limits=c(0,115),breaks=seq(0,100,25),expand=c(0,0)) +
    scale_y_continuous(limits=c(0,130),breaks=seq(0,125,25),expand=c(0,0)) +
    labs(title=arm_title(a,letter),x="Training steps (x 1,000)",y=if (a=="A") "Mean episode reward" else NULL)
}
zoom_panel <- function(a, letter) {
  d <- filter(curves,arm==a,timesteps<=10000); start <- filter(t0,arm==a)
  lo <- 5*floor((min(d$q1,start$t0)-3)/5); hi <- 5*ceiling((max(d$q3,start$t0)+3)/5)
  ggplot(d,aes(timesteps/1000,mid,colour=variant,fill=variant,linetype=variant)) + layers(d,start,3.4) +
    annotate("text",x=0.25,y=lo,label=sprintf("t = 0 reference: %.1f",start$t0),hjust=0,vjust=-.6,size=3.6,colour=ink) +
    scale_x_continuous(limits=c(0,10),breaks=seq(0,10,2),expand=expansion(mult=c(.01,.02))) +
    scale_y_continuous(limits=c(lo,hi),breaks=scales::breaks_pretty(4),expand=c(0,0)) +
    labs(title=sprintf("%s   First 10,000 steps, Arm %s",letter,a),x="Training steps (x 1,000)",y=if (a=="A") "Mean episode reward" else NULL)
}
key <- ggplot(data.frame(x=rep(c(0,1),3),y=1,variant=factor(rep(ids,each=2),levels=ids)),aes(x,y,colour=variant,linetype=variant)) +
  geom_line(linewidth=.9) + scale_colour_manual(values=colors,name=NULL) + scale_linetype_manual(values=lines,name=NULL) +
  guides(colour=guide_legend(keywidth=unit(3,"lines")),linetype=guide_legend(keywidth=unit(3,"lines"))) +
  theme_void(base_family="Arial") + theme(legend.position="inside",legend.position.inside=c(.5,.5),legend.direction="horizontal",legend.text=element_text(size=13,colour=ink)) +
  coord_cartesian(xlim=c(5,6),ylim=c(5,6))
top <- wrap_plots(mapply(full_panel, arms, letters[1:4], SIMPLIFY=FALSE), nrow=1)
bottom <- wrap_plots(mapply(zoom_panel, arms, letters[5:8], SIMPLIFY=FALSE), nrow=1)
fig <- top / bottom / key + plot_layout(heights=c(1,.75,.06)) +
  plot_annotation(title="Equal-start training curves in all four arms: SP, BAL and FD share one return at t = 0",
    subtitle="Lines: median of 5 paired seeds; shading: IQR. Weights: (speed, spacing); BAL is (0.70, 0.70) in every arm. Dark point: t = 0 reference; dotted: link to the first logged point.",
    caption=paste("t = 0 is computed from 2,000 uniform-random episodes per arm, scored under each weight pair; it is not a training-log entry. Logged curves are rolling means of the last 100 episodes.",
      "Arm B BAL reuses the main Arm B runs (seeds 4100-4104); Arm D SP includes seed 4204, which trained fully but had no checkpoint pass model selection.",
      "Reward functions differ, so curve heights after t = 0 do not rank driving performance.",sep="\n"),
    theme=theme(plot.title=element_text(size=21,face="bold",colour=ink,margin=margin(b=6),family="Arial"),
      plot.subtitle=element_text(size=12.5,colour=muted,margin=margin(b=10),family="Arial"),
      plot.caption=element_text(size=10.5,hjust=0,colour=muted,lineheight=1.25,family="Arial"),
      plot.margin=margin(16,18,10,18),plot.background=element_rect(fill="white",colour=NA)))
for (ext in c("png","svg","pdf")) {
  dev <- switch(ext, png=ragg::agg_png, svg=svglite::svglite, pdf=grDevices::cairo_pdf)
  ggsave(file.path(out,paste0("training_curves_equal_start_4arm.",ext)),fig,device=dev,width=16,height=9,units="in",dpi=300,bg="white")
}
