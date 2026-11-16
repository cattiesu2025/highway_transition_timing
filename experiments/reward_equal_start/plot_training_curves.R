#!/usr/bin/env Rscript
# Equal-start SP/BAL/FD training curves. R-only rendering, 16:9, editable SVG/PDF + 300-dpi PNG.
# Contract: x=0 is the computed random-policy reference (identical by construction);
# logged curves start near 1K steps. Lines: median of 5 paired seeds; ribbons: IQR.
# Curve heights after t=0 reflect different reward functions, not driving performance.
.libPaths(c("tmp/r-lib", .libPaths()))
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(patchwork); library(jsonlite)})
stopifnot(requireNamespace("ragg", quietly=TRUE), requireNamespace("svglite", quietly=TRUE))
base <- "outputs/reward_equal_start"
out <- file.path(base, "figures"); dir.create(file.path(out, "source_data"), recursive=TRUE, showWarnings=FALSE)
ids <- c("FD","BAL","SP")
trace <- read.csv(file.path(base, "analysis/training_progress.csv"), stringsAsFactors=FALSE)
status <- read.csv(file.path(base, "analysis/run_status.csv"), stringsAsFactors=FALSE)
stopifnot(nrow(status)==30, all(status$status=="complete"), setequal(trace$variant, ids))
ref <- fromJSON(file.path(base, "equal_start_weights.json"))
t0 <- data.frame(arm=c("A","C"), t0=c(ref$arms$A$t0_return$BAL, ref$arms$C$t0_return$BAL))
stopifnot(all(abs(unlist(ref$arms$A$t0_return)-t0$t0[1])<1e-9), all(abs(unlist(ref$arms$C$t0_return)-t0$t0[2])<1e-9))

# Interpolate within observed support only; no smoothing or extrapolation.
grid <- c(seq(1000,9500,500), seq(10000,99000,1000))
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
colors <- c(FD="#2764AC", BAL="#2A9D8F", SP="#D9622B")
lines <- c(FD="solid", BAL="longdash", SP="dotdash")
ink <- "#243746"; muted <- "#596B79"
fmt <- function(w) sprintf("(%.3f, %.3f)", w$speed, w$spacing)
arm_title <- function(a, name) sprintf("%s\nFD %s   BAL (0.700, 0.700)   SP %s", name, fmt(ref$arms[[a]]$weights$FD), fmt(ref$arms[[a]]$weights$SP))
# Keep end labels legible where curves finish close together, without moving the data.
spread <- function(y, gap) { o <- order(y); v <- y[o]; for (i in 2:length(v)) v[i] <- max(v[i], v[i-1]+gap); v <- v-(mean(v)-mean(y)); r <- y; r[o] <- v; r }
titles <- c(A=arm_title("A","Arm A, single-lane slowdown"), C=arm_title("C","Arm C, occupied target lane"))
theme_set(theme_classic(base_size=15, base_family="Arial") + theme(
  text=element_text(colour=ink), axis.text=element_text(colour=ink,size=13), axis.title=element_text(size=15),
  axis.line=element_line(colour="#8896A0",linewidth=.4), axis.ticks=element_line(colour="#8896A0",linewidth=.4),
  panel.grid.major.y=element_line(colour="#E9EDF0",linewidth=.4), plot.title=element_text(size=14,face="bold",colour=ink,margin=margin(b=8),lineheight=1.15),
  plot.margin=margin(6,12,6,10), plot.background=element_rect(fill="white",colour=NA), legend.position="none"))

full_panel <- function(a, letter) {
  d <- filter(curves,arm==a); start <- filter(t0,arm==a)
  ends <- d %>% filter(timesteps==max(timesteps)) %>% mutate(label_y=spread(mid,7))
  connect <- d %>% group_by(variant) %>% filter(timesteps==min(timesteps)) %>% ungroup() %>% mutate(x0=0,y0=start$t0)
  ggplot(d,aes(timesteps/1000,mid,colour=variant,fill=variant,linetype=variant)) +
    geom_ribbon(aes(ymin=q1,ymax=q3),alpha=.14,colour=NA,linetype=0) +
    geom_segment(data=connect,aes(x=x0,y=y0,xend=timesteps/1000,yend=mid),linetype="dotted",linewidth=.5) +
    geom_line(linewidth=.9) +
    geom_point(data=start,aes(0,t0),inherit.aes=FALSE,size=3.4,shape=21,fill=ink,colour="white",stroke=.9) +
    geom_text(data=ends,aes(x=100.8,y=label_y,label=variant),hjust=0,size=4.6,colour=ink,fontface="bold",show.legend=FALSE) +
    scale_colour_manual(values=colors) + scale_fill_manual(values=colors) + scale_linetype_manual(values=lines) +
    scale_x_continuous(limits=c(0,109),breaks=seq(0,100,25),expand=c(0,0)) +
    scale_y_continuous(limits=c(0,125),breaks=seq(0,125,25),expand=c(0,0)) +
    labs(title=paste0(letter,"   ",titles[[a]]),x="Training steps (x 1,000)",y="Mean episode reward")
}
zoom_panel <- function(a, letter) {
  d <- filter(curves,arm==a,timesteps<=10000); start <- filter(t0,arm==a)
  connect <- d %>% group_by(variant) %>% filter(timesteps==min(timesteps)) %>% ungroup() %>% mutate(x0=0,y0=start$t0)
  ggplot(d,aes(timesteps/1000,mid,colour=variant,fill=variant,linetype=variant)) +
    geom_ribbon(aes(ymin=q1,ymax=q3),alpha=.14,colour=NA,linetype=0) +
    geom_segment(data=connect,aes(x=x0,y=y0,xend=timesteps/1000,yend=mid),linetype="dotted",linewidth=.5) +
    geom_line(linewidth=.9) +
    geom_point(data=start,aes(0,t0),inherit.aes=FALSE,size=3.8,shape=21,fill=ink,colour="white",stroke=.9) +
    geom_text(data=start,aes(0.2,t0-2.6,label=sprintf("t = 0: %.2f",t0)),inherit.aes=FALSE,hjust=0,size=4.3,colour=ink) +
    scale_colour_manual(values=colors) + scale_fill_manual(values=colors) + scale_linetype_manual(values=lines) +
    scale_x_continuous(limits=c(0,10),breaks=seq(0,10,2),expand=expansion(mult=c(.01,.02))) +
    scale_y_continuous(limits=c(48,80),breaks=seq(50,80,10),expand=c(0,0)) +
    labs(title=paste0(letter,"   First 10,000 steps, Arm ",a),x="Training steps (x 1,000)",y="Mean episode reward")
}
key <- ggplot(data.frame(x=rep(c(0,1),3),y=1,variant=factor(rep(ids,each=2),levels=ids)),aes(x,y,colour=variant,linetype=variant)) +
  geom_line(linewidth=.9) + scale_colour_manual(values=colors,name=NULL) + scale_linetype_manual(values=lines,name=NULL) +
  guides(colour=guide_legend(keywidth=unit(3,"lines")),linetype=guide_legend(keywidth=unit(3,"lines"))) +
  theme_void(base_family="Arial") + theme(legend.position="inside",legend.position.inside=c(.5,.5),legend.direction="horizontal",legend.text=element_text(size=14,colour=ink)) +
  coord_cartesian(xlim=c(5,6),ylim=c(5,6))
fig <- (full_panel("A","a") | full_panel("C","b")) / (zoom_panel("A","c") | zoom_panel("C","d")) / key +
  plot_layout(heights=c(1,.8,.07)) +
  plot_annotation(title="Equal-start training curves: SP, BAL and FD share one return at t = 0",
    subtitle="Lines: median of 5 paired seeds; shading: IQR. Weights: (speed, spacing). Dark point: t = 0 reference; dotted: link to the first logged point.",
    caption=paste("t = 0 is computed from 2,000 uniform-random episodes per arm, scored under each weight pair; it is not a training-log entry. Logged curves are rolling means of the last 100 episodes.",
      "Reward functions differ, so curve heights after t = 0 do not rank driving performance.",sep="\n"),
    theme=theme(plot.title=element_text(size=24,face="bold",colour=ink,margin=margin(b=6),family="Arial"),
      plot.subtitle=element_text(size=14.5,colour=muted,margin=margin(b=12),family="Arial"),
      plot.caption=element_text(size=12,hjust=0,colour=muted,lineheight=1.25,family="Arial"),
      plot.margin=margin(18,22,12,22),plot.background=element_rect(fill="white",colour=NA)))
for (ext in c("png","svg","pdf")) {
  dev <- switch(ext, png=ragg::agg_png, svg=svglite::svglite, pdf=grDevices::cairo_pdf)
  ggsave(file.path(out,paste0("training_curves_equal_start.",ext)),fig,device=dev,width=16,height=9,units="in",dpi=300,bg="white")
}
