#!/usr/bin/env Rscript
# First-log equal-start training curves, 4 arms x SP/BAL/FD x 5 paired seeds. R-only, 16:9, SVG/PDF + 300-dpi PNG.
# Contract: curves start at each run's first SB3 log (mean of its first 4 episodes, 282-400 steps; identical step for
# SP/BAL/FD of a seed). In A, B and D the SP/FD weights make the 5-seed mean of that first point equal to BAL's.
# Thick lines: mean of 5 seeds; ribbons: 95% Student-t CI of the mean (n = 5 is too few for a bootstrap CI);
# thin lines: individual seeds. The first mean point sits at the arm's mean first-log step; later points are
# per-seed linear interpolations on a common step grid. Input: collect_first_log.py.
.libPaths(c("tmp/r-lib", .libPaths()))
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(patchwork)})
stopifnot(requireNamespace("ragg", quietly=TRUE), requireNamespace("svglite", quietly=TRUE))
base <- "outputs/reward_equal_start/analysis_first_log"
out <- "outputs/reward_equal_start/figures_first_log"; dir.create(file.path(out, "source_data"), recursive=TRUE, showWarnings=FALSE)
ids <- c("FD","BAL","SP"); arms <- c("A","B","C","D")

trace <- read.csv(file.path(base, "training_progress.csv"))
status <- read.csv(file.path(base, "run_status.csv"))
stopifnot(nrow(status)==60, all(status$final_timesteps>=99000), all(table(status$arm, status$variant)==5))

first <- status %>% group_by(arm) %>% mutate(x0=mean(first_log_timesteps)) %>% ungroup() %>%
  transmute(arm, variant, seed, timesteps=x0, reward=first_log_reward)
grid <- c(seq(500,9750,250), seq(10000,99000,1000))
aligned <- bind_rows(first, bind_rows(lapply(split(trace, interaction(trace$arm,trace$variant,trace$seed,drop=TRUE)), function(d) {
  d <- d[order(d$timesteps),]
  data.frame(arm=d$arm[1], variant=d$variant[1], seed=d$seed[1], timesteps=grid,
             reward=approx(d$timesteps,d$episode_reward,xout=grid,rule=1)$y)
}))) %>% arrange(arm,variant,seed,timesteps)
curves <- aligned %>% group_by(arm,variant,timesteps) %>%
  summarise(n=sum(is.finite(reward)), mean=mean(reward), sd=sd(reward), .groups="drop") %>%
  mutate(half=qt(.975,n-1)*sd/sqrt(n), lo=mean-half, hi=mean+half)
stopifnot(all(curves$n==5))
starts <- curves %>% group_by(arm) %>% filter(timesteps==min(timesteps)) %>% ungroup()
write.csv(curves, file.path(out,"source_data","training_mean_ci.csv"), row.names=FALSE)
write.csv(aligned, file.path(out,"source_data","training_aligned_by_seed.csv"), row.names=FALSE)
write.csv(starts, file.path(out,"source_data","first_log_start.csv"), row.names=FALSE)
curves$variant <- factor(curves$variant, levels=ids); trace$variant <- factor(trace$variant, levels=ids)

colors <- c(FD="#1B7FB5", BAL="#C13F8E", SP="#DE8C14")  # validated categorical palette (same as the 4-arm figure)
lines <- c(FD="solid", BAL="longdash", SP="dotdash")
ink <- "#243746"; muted <- "#596B79"
names_arm <- c(A="Arm A, single lane", B="Arm B, empty target lane", C="Arm C, occupied target lane", D="Arm D, yellow car farther")
w <- status %>% distinct(arm, variant, speed_weight, front_distance_weight)
fmt <- function(a, v) { r <- filter(w, arm==a, variant==v); sprintf("(%.2f, %.2f)", r$speed_weight, r$front_distance_weight) }
arm_title <- function(a, letter) sprintf("%s   %s\nFD %s  SP %s", letter, names_arm[[a]], fmt(a,"FD"), fmt(a,"SP"))
spread <- function(y, gap) { o <- order(y); v <- y[o]; for (i in 2:length(v)) v[i] <- max(v[i], v[i-1]+gap); v <- v-(mean(v)-mean(y)); r <- y; r[o] <- v; r }
theme_set(theme_classic(base_size=13, base_family="Arial") + theme(
  text=element_text(colour=ink), axis.text=element_text(colour=ink,size=11), axis.title=element_text(size=12),
  axis.line=element_line(colour="#8896A0",linewidth=.4), axis.ticks=element_line(colour="#8896A0",linewidth=.4),
  panel.grid.major.y=element_line(colour="#E9EDF0",linewidth=.4),
  plot.title=element_text(size=12,face="bold",colour=ink,margin=margin(b=6),lineheight=1.15),
  plot.margin=margin(6,10,6,8), plot.background=element_rect(fill="white",colour=NA), legend.position="none"))

layers <- function(a, xmax) list(
  geom_line(data=filter(trace,arm==a,timesteps<=xmax), aes(timesteps/1000, episode_reward, group=interaction(variant,seed), colour=variant),
            inherit.aes=FALSE, linewidth=.25, alpha=.28),
  geom_ribbon(aes(ymin=lo,ymax=hi),alpha=.16,colour=NA,linetype=0),
  geom_line(linewidth=.9),
  scale_colour_manual(values=colors), scale_fill_manual(values=colors), scale_linetype_manual(values=lines))
start_mark <- function(a, size) { s <- filter(starts, arm==a)
  if (diff(range(s$mean)) < .01) list(geom_point(data=s[1,], aes(timesteps/1000, mean), inherit.aes=FALSE, size=size, shape=21, fill=ink, colour="white", stroke=.9))
  else list(geom_point(data=s, aes(timesteps/1000, mean, colour=variant), inherit.aes=FALSE, size=size*.8)) }
full_panel <- function(a, letter) {
  d <- filter(curves,arm==a)
  ends <- d %>% filter(timesteps==max(timesteps)) %>% mutate(label_y=spread(mean,8))
  ggplot(d,aes(timesteps/1000,mean,colour=variant,fill=variant,linetype=variant)) + layers(a, 1e6) + start_mark(a, 2.6) +
    geom_text(data=ends,aes(x=101,y=label_y,label=variant),hjust=0,size=3.8,colour=ink,fontface="bold",show.legend=FALSE) +
    scale_x_continuous(limits=c(0,115),breaks=seq(0,100,25),expand=c(0,0)) +
    coord_cartesian(ylim=c(0,150), clip="off") + scale_y_continuous(breaks=seq(0,150,25),expand=c(0,0)) +
    labs(title=arm_title(a,letter),x="Training steps (x 1,000)",y=if (a=="A") "Mean episode reward" else NULL)
}
zoom_panel <- function(a, letter) {
  d <- filter(curves,arm==a,timesteps<=10000); s <- filter(starts, arm==a)
  lo <- 5*floor((min(d$lo)-3)/5); hi <- 5*ceiling((max(d$hi)+3)/5)
  lab <- if (diff(range(s$mean)) < .01) sprintf("First log: all three %.1f", s$mean[1]) else
    sprintf("First log: SP %.1f, BAL %.1f, FD %.1f", s$mean[s$variant=="SP"], s$mean[s$variant=="BAL"], s$mean[s$variant=="FD"])
  ggplot(d,aes(timesteps/1000,mean,colour=variant,fill=variant,linetype=variant)) + layers(a, 10000) + start_mark(a, 3.2) +
    annotate("text",x=0.3,y=lo,label=lab,hjust=0,vjust=-.6,size=3.6,colour=ink) +
    scale_x_continuous(limits=c(0,10),breaks=seq(0,10,2),expand=expansion(mult=c(.01,.02))) +
    coord_cartesian(ylim=c(lo,hi)) + scale_y_continuous(breaks=scales::breaks_pretty(4),expand=c(0,0)) +
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
  plot_annotation(title="Training curves start from the same first logged reward for SP, BAL and FD",
    subtitle="Thick lines: mean of 5 paired seeds; shading: 95% CI of the mean (Student t); thin lines: individual seeds. Weights: (speed, spacing); BAL is (0.70, 0.70). Dark point: shared first-log mean.",
    caption=paste("Each curve is SB3's rolling mean of the last 100 episode returns; the first log averages the first 4 episodes (282-400 steps), which are uniform-random and identical for SP/BAL/FD of a seed.",
      "A, B, D: SP/FD weights solved so the 5-seed mean of the first log equals BAL's. C: weights from 2,000 random episodes, first-log means within 0.12 (SP 52.87, BAL 52.81, FD 52.76).",
      "Arm D seed 4204 (SP and FD) trained fully but no checkpoint passed model selection; included here. Reward functions differ, so later curve heights do not rank driving performance.",sep="\n"),
    theme=theme(plot.title=element_text(size=21,face="bold",colour=ink,margin=margin(b=6),family="Arial"),
      plot.subtitle=element_text(size=12.5,colour=muted,margin=margin(b=10),family="Arial"),
      plot.caption=element_text(size=10.5,hjust=0,colour=muted,lineheight=1.25,family="Arial"),
      plot.margin=margin(16,18,10,18),plot.background=element_rect(fill="white",colour=NA)))
for (ext in c("png","svg","pdf")) {
  dev <- switch(ext, png=ragg::agg_png, svg=svglite::svglite, pdf=grDevices::cairo_pdf)
  ggsave(file.path(out,paste0("training_curves_first_log_4arm.",ext)),fig,device=dev,width=16,height=9,units="in",dpi=300,bg="white")
}
