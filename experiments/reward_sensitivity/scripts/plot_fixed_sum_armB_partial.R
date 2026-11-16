#!/usr/bin/env Rscript
# Fixed-sum F1-F9 report: R-only rendering.
# Claim: describe how onset timing and training returns vary with weight allocation.
# Quantitative grids; descriptive median/IQR, five paired seeds, no inference.
# Contract: table describes timing across all weights; training grid describes
# logged returns, not performance rankings; counterfactual grid reports both
# behavior occurrence and collisions. Five training seeds are the replicates.
# 16:9 slide-sized quantitative grids, editable SVG/PDF and 300-dpi PNG.
.libPaths(c("tmp/r-lib", .libPaths()))
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(patchwork)})
stopifnot(requireNamespace("ragg", quietly=TRUE), requireNamespace("svglite", quietly=TRUE))
out <- "outputs/reward_fixed_sum/figures_armB_partial"
dir.create(out, recursive=TRUE, showWarnings=FALSE)
dir.create(file.path(out, "source_data"), showWarnings=FALSE)
dir.create("tmp/reward_fixed_sum_armB_partial_preview", recursive=TRUE, showWarnings=FALSE)
ids <- paste0("F",1:9)
reward_groups <- setNames(rep(c("FD-like","Intermediate","SP-like"),each=3),ids)
ink <- "#243746"; blue <- "#397F96"; orange <- "#BA6D35"
read_table <- function(experiment, name) read.csv(file.path("outputs", experiment, "analysis", paste0(name, ".csv")), stringsAsFactors=FALSE)
seed <- read_table("reward_fixed_sum", "seed_summary")
trace <- read_table("reward_fixed_sum", "training_progress")
# Partial B coverage: only exact coefficient matches may be reused.
b_trace <- read_table("reward_sensitivity", "training_progress") %>%
  filter(arm == "B", variant %in% c("R2", "BAL")) %>%
  mutate(variant = recode(variant, R2="F3", BAL="F5"))
stopifnot(nrow(b_trace)>0,
          all(b_trace$speed_weight[b_trace$variant=="F3"]==0.42),
          all(b_trace$front_distance_weight[b_trace$variant=="F3"]==0.98),
          all(b_trace$speed_weight[b_trace$variant=="F5"]==0.70),
          all(b_trace$front_distance_weight[b_trace$variant=="F5"]==0.70),
          all((b_trace %>% distinct(variant,seed) %>% count(variant))$n==5))
trace <- bind_rows(trace, b_trace)
write.csv(trace,file.path(out,"source_data","training_progress.csv"),row.names=FALSE)
status <- read_table("reward_fixed_sum", "run_status")
stopifnot(nrow(seed)==90, nrow(status)==90, all(status$status=="complete"),
          all(seed$n_onset==36), all(seed$n_collision==0), all(seed$n_total==36),
          all(count(seed, arm, variant)$n==5), !anyDuplicated(seed[c("arm","variant","seed")]),
          setequal(seed$variant,ids),
          all(abs(seed$speed_weight+seed$front_distance_weight-1.4)<1e-9))
write.csv(seed, file.path(out,"source_data","behavior_by_seed.csv"), row.names=FALSE)
theme_set(theme_classic(base_size=16, base_family="Arial") + theme(
  text=element_text(colour=ink), plot.title=element_text(face="bold", colour=ink),
  axis.text=element_text(colour=ink), plot.background=element_rect(fill="white",colour=NA),
  plot.margin=margin(12,16,12,16), plot.caption=element_text(size=12,hjust=0,colour="#596B79")))
save_material <- function(p, name) {
  if ("--training-only" %in% commandArgs(trailingOnly=TRUE) && name != "02_training_curves") return(invisible(NULL))
  if ("--main-only" %in% commandArgs(trailingOnly=TRUE) && name == "03_counterfactual_results") return(invisible(NULL))
  for (ext in c("png","svg","pdf")) {
    dev <- switch(ext, png=ragg::agg_png, svg=svglite::svglite, pdf=grDevices::cairo_pdf)
    ggsave(file.path(out,paste0(name,".",ext)), p, device=dev,
           width=20,height=11.25,units="in",dpi=300,bg="white",limitsize=FALSE)
  }
  ggsave(file.path("tmp/reward_fixed_sum_armB_partial_preview",paste0(name,".png")),p,
         device=ragg::agg_png,width=20,height=11.25,units="in",dpi=85,bg="white")
}

tab <- seed %>% filter(arm=="A") %>% distinct(variant,speed_weight,front_distance_weight)
tab <- tab[match(ids,tab$variant),]
tab$Speed_weight <- tab$speed_weight
tab$Spacing_weight <- tab$front_distance_weight

# Align within observed support only. No extrapolation to the 100K budget.
aligned <- bind_rows(lapply(split(trace,interaction(trace$arm,trace$variant,trace$seed,drop=TRUE)),function(d) {
  d <- d[order(d$timesteps),]; grid <- seq(1000,99000,1000)
  data.frame(arm=d$arm[1],variant=d$variant[1],seed=d$seed[1],timesteps=grid,
             reward=approx(d$timesteps,d$episode_reward,xout=grid,rule=1)$y)
}))
curves <- aligned %>% group_by(arm,variant,timesteps) %>% summarise(n=sum(is.finite(reward)),
  mid=median(reward,na.rm=TRUE),q1=quantile(reward,.25,na.rm=TRUE),q3=quantile(reward,.75,na.rm=TRUE),.groups="drop") %>% filter(n==5)
stopifnot(nrow(curves)>0, all(is.finite(trace$episode_reward)))
write.csv(curves,file.path(out,"source_data","training_median_iqr.csv"),row.names=FALSE)
write.csv(aligned,file.path(out,"source_data","training_aligned_by_seed.csv"),row.names=FALSE)
labels <- setNames(sprintf("%s  (%.2f, %.2f)",ids,tab$Speed_weight,tab$Spacing_weight),ids)
trace$variant <- factor(trace$variant,levels=ids); curves$variant <- factor(curves$variant,levels=ids)
reward_colors <- setNames(c("#123E75","#2764AC","#428BC5",
                            "#124D2C","#3D9555","#91BE96",
                            "#9B2335","#D04D58","#E58B86"),ids)
reward_lines <- setNames(rep(c("solid","longdash","dotdash"),3),ids)
training <- ggplot(curves,aes(timesteps/1000,mid,colour=variant,fill=variant,linetype=variant)) +
  geom_ribbon(aes(ymin=q1,ymax=q3),alpha=.09,colour=NA,linetype=0,show.legend=FALSE) +
  geom_line(linewidth=1.05) +
  geom_text(data=data.frame(arm="B",x=50,y=12,label="Partial coverage: F3 and F5 only\nF1/F2/F4/F6/F7/F8/F9 unavailable"),
            aes(x=x,y=y,label=label),inherit.aes=FALSE,size=4,colour="#596B79") +
  geom_vline(xintercept=100,linetype="dotted",colour="#788793",linewidth=.45) +
  facet_wrap(~arm,nrow=1,labeller=as_labeller(c(A="a   Arm A | Single-lane slowdown",B="b   Arm B | Empty target lane",C="c   Arm C | Occupied target lane"))) +
  scale_colour_manual(values=reward_colors,breaks=ids,labels=labels) +
  scale_fill_manual(values=reward_colors,guide="none") +
  scale_linetype_manual(values=reward_lines,breaks=ids,labels=labels) +
  scale_x_continuous(limits=c(0,102),breaks=c(0,25,50,75,100),expand=c(0,0)) +
  scale_y_continuous(limits=c(0,NA),expand=expansion(mult=c(0,.05))) +
  labs(title="Training curves across all reward configurations",
    subtitle="Median across 5 training seeds; shading shows IQR. Colors identify reward configurations in both arms.",
    x="Training steps (x 1,000)",y="Mean episode reward",colour=NULL,linetype=NULL,
    caption=paste("Legend: reward (speed weight, front-spacing weight). Dashed: FD/BAL/SP baselines. Solid: R1-R8. Dotted vertical line: 100K training budget.",
                  "Return is specific to each reward function; curve heights do not rank driving performance. No extrapolation beyond observed training logs.",sep="\n")) +
  guides(colour=guide_legend(nrow=2,byrow=TRUE),linetype=guide_legend(nrow=2,byrow=TRUE)) +
  theme(strip.background=element_blank(),strip.text=element_text(size=16,face="bold",hjust=0,margin=margin(t=15,b=18)),
    axis.text=element_text(size=16),axis.title=element_text(size=19),
    axis.title.x=element_text(margin=margin(t=14)),axis.title.y=element_text(margin=margin(r=14)),
    panel.grid.major.y=element_line(colour="#E5EBEF",linewidth=.35),panel.spacing.x=grid::unit(15,"mm"),
    legend.position="bottom",legend.text=element_text(size=15),legend.key.width=grid::unit(16,"mm"),
    legend.key.height=grid::unit(9,"mm"),legend.margin=margin(t=14,b=10),
    plot.title=element_text(size=28,face="bold",colour=ink),plot.subtitle=element_text(size=18,margin=margin(t=8,b=10)),
    plot.caption=element_text(size=13,hjust=0,lineheight=1.35),plot.margin=margin(22,30,20,25))
legend_data <- data.frame(variant=ids, x=rep(c(4.3,9.2,14.1),3), y=rep(3:1,each=3))
legend_data$label <- unname(labels[legend_data$variant])
legend_plot <- ggplot(legend_data) +
  geom_segment(aes(x=x,xend=x+.65,y=y,yend=y,colour=variant,linetype=variant),linewidth=1.05) +
  geom_text(aes(x=x+.82,y=y,label=label),hjust=0,size=4.4,colour=ink) +
  geom_text(data=data.frame(x=.1,y=3:1,label=c("FD-like  (r > 2)","Intermediate  (0.5 <= r <= 2)","SP-like  (r < 0.5)")),
            aes(x,y,label=label),hjust=0,size=4.2,fontface="bold",colour=ink) +
  scale_colour_manual(values=reward_colors,guide="none") + scale_linetype_manual(values=reward_lines,guide="none") +
  coord_cartesian(xlim=c(0,20.3),ylim=c(.5,3.5),expand=FALSE,clip="off") + theme_void() +
  theme(plot.margin=margin(8,25,4,40))
training <- ((training + labs(title=NULL,subtitle=NULL,caption=NULL) + theme(legend.position="none")) /
             legend_plot) + plot_layout(heights=c(1,.22)) +
  plot_annotation(title="Fixed-sum training curves by reward-ratio group",
    subtitle="Blue: FD-like | Green: Intermediate | Red: SP-like. Lines: median across 5 seeds; shading: IQR.",
    caption="Weights: (speed, spacing), total = 1.40. Arm B: only F3 and F5 available (5 seeds each); other 7 configurations have not been run.",
    theme=theme(plot.title=element_text(size=28,face="bold",colour=ink),plot.subtitle=element_text(size=17,margin=margin(t=8,b=8)),
                plot.caption=element_text(size=13,hjust=0,lineheight=1.35),plot.margin=margin(22,25,20,25)))
save_material(training,"02_training_curves")

writeLines(c("Partial three-arm preview; A/C unchanged.",
"B uses exact weight matches R2=F3 and BAL=F5 from existing training logs, five seeds each.",
"Seven B configurations are unavailable; no curves are inferred or fabricated.",
"Lines: median; ribbons: IQR; 1K-step interpolation within observed support only.",
"Raw reward heights are not common performance scores across reward functions."),file.path(out,"figure_notes.md"))
