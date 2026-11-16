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
out <- "outputs/reward_fixed_sum/figures"
dir.create(out, recursive=TRUE, showWarnings=FALSE)
dir.create(file.path(out, "source_data"), showWarnings=FALSE)
dir.create("tmp/reward_fixed_sum_preview", recursive=TRUE, showWarnings=FALSE)
ids <- paste0("F",1:9)
reward_groups <- setNames(rep(c("FD-like","Intermediate","SP-like"),each=3),ids)
ink <- "#243746"; blue <- "#397F96"; orange <- "#BA6D35"
read_table <- function(experiment, name) read.csv(file.path("outputs", experiment, "analysis", paste0(name, ".csv")), stringsAsFactors=FALSE)
seed <- read_table("reward_fixed_sum", "seed_summary")
trace <- read_table("reward_fixed_sum", "training_progress")
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
  ggsave(file.path("tmp/reward_fixed_sum_preview",paste0(name,".png")),p,
         device=ragg::agg_png,width=20,height=11.25,units="in",dpi=85,bg="white")
}

# Table: median [Q1, Q3] of five seed-specific conditional onset medians.
behavior <- seed %>% group_by(arm,variant,speed_weight,front_distance_weight) %>%
  summarise(mid=median(conditional_median_onset_seconds),
            q1=quantile(conditional_median_onset_seconds,.25),
            q3=quantile(conditional_median_onset_seconds,.75), .groups="drop")
tab <- data.frame(Configuration=ids, Group=unname(reward_groups[ids]))
for (arm_id in c("A","C")) {
  d <- behavior %>% filter(.data$arm==.env$arm_id)
  d <- d[match(ids,d$variant),]
  tab[[paste0(arm_id," onset (s): median [IQR]")]] <- sprintf("%.1f [%.1f, %.1f]",d$mid,d$q1,d$q3)
  if(arm_id=="A") {tab$Speed_weight <- d$speed_weight; tab$Spacing_weight <- d$front_distance_weight}
}
tab$Spacing_speed_ratio <- tab$Spacing_weight/tab$Speed_weight
tab <- tab[,c(1,2,4,5,7,3,6)]
write.csv(tab,file.path(out,"01_reward_configuration_table.csv"),row.names=FALSE)
write.table(tab,file.path(out,"01_reward_configuration_table.tsv"),sep="\t",row.names=FALSE,quote=FALSE)
write.csv(behavior,file.path(out,"source_data","behavior_summary_numeric.csv"),row.names=FALSE)
xs <- c(.2,1.5,3.5,5.25,7.0,9.6,13.1)
headers <- c("Reward","Group","Speed\nweight","Front-spacing\nweight","Spacing / speed\nratio","A: slowdown onset\n(s)","C: actual lane-change onset\n(s)")
cells <- bind_rows(lapply(seq_len(ncol(tab)),function(j) data.frame(x=xs[j],y=10-seq_len(9),
  label=if(j %in% c(3,4,5)) sprintf("%.2f",tab[[j]]) else tab[[j]],hjust=if(j<=2) 0 else .5)))
tbl <- ggplot() +
  annotate("rect",xmin=0,xmax=15.1,ymin=9.65,ymax=11.15,fill="#E8F0F4") +
  geom_rect(data=data.frame(y=10-seq_len(9)),aes(xmin=0,xmax=15.1,ymin=y-.43,ymax=y+.43),
            fill=c(rep("#EFF5FC",3),rep("#EFF7F1",3),rep("#FCF0EF",3)),colour=NA) +
  annotate("segment",x=0,xend=15.1,y=c(9.6,6.5,3.5,.4),yend=c(9.6,6.5,3.5,.4),colour="#CBD7DE",linewidth=.6) +
  geom_text(data=data.frame(x=xs,y=10.4,label=headers,hjust=c(0,0,.5,.5,.5,.5,.5)),
            aes(x,y,label=label,hjust=hjust),size=5.7,fontface="bold",colour=ink,lineheight=1.1) +
  geom_text(data=cells,aes(x,y,label=label,hjust=hjust),size=5.8,colour=ink) +
  coord_cartesian(xlim=c(0,15.1),ylim=c(.3,11.2),expand=FALSE,clip="off") + theme_void(base_family="Arial") +
  labs(title="Fixed-sum reward configurations and behavioral outcomes",
       subtitle="A: single-lane slowdown   |   C: lane change with an occupied target lane   |   5 paired training seeds per configuration",
       caption=paste("Speed + spacing = 1.40. Cells: median [Q1, Q3] of 5 seed-specific onset medians; 36 episodes per seed.",
 "Groups: FD-like r > 2; Intermediate 0.5 <= r <= 2; SP-like r < 0.5. F3 reuses R2; F5 reuses BAL.", sep="\n")) +
  theme(plot.title=element_text(size=28,face="bold",colour=ink,margin=margin(b=12)),
        plot.subtitle=element_text(size=17,colour="#596B79",margin=margin(b=24)),
        plot.caption=element_text(size=14,hjust=0,lineheight=1.4,margin=margin(t=20),colour="#596B79"),
        plot.margin=margin(25,30,25,30),plot.background=element_rect(fill="white",colour=NA))
save_material(tbl,"01_reward_configuration_table")

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
  geom_vline(xintercept=100,linetype="dotted",colour="#788793",linewidth=.45) +
  facet_wrap(~arm,nrow=1,labeller=as_labeller(c(A="a   Arm A | Single-lane slowdown",C="b   Arm C | Occupied target lane"))) +
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
  theme(strip.background=element_blank(),strip.text=element_text(size=21,face="bold",hjust=0,margin=margin(t=15,b=18)),
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
    caption="Weights: (speed, spacing), total = 1.40.",
    theme=theme(plot.title=element_text(size=28,face="bold",colour=ink),plot.subtitle=element_text(size=17,margin=margin(t=8,b=8)),
                plot.caption=element_text(size=13,hjust=0,lineheight=1.35),plot.margin=margin(22,25,20,25)))
save_material(training,"02_training_curves")


writeLines(c(
  "# Fixed-sum F1-F9 figure notes",
  "Two descriptive quantitative grids matching the supplied presentation layout; 20 x 11.25 inches.",
  "F1-F9 includes 70 new models and 20 reused condition records (F3=R2; F5=BAL).",
  "Five paired training seeds per arm/configuration; 36 development episodes per seed.",
  "Table: median [Q1, Q3] across five seed-specific conditional onset medians.",
  "All 3240 original-scene episodes have observed onset and no recorded collision.",
  "A measures persistent slowdown; C measures actual physical lane change.",
  "Curves: median and IQR over seeds on a 1K-step grid, only where all five seeds have observed support; no extrapolation or smoothing.",
  "Raw returns use different reward functions; curve heights are not comparable performance scores.",
  "Development analysis is descriptive: no hypothesis tests or independent held-out claims.",
  "PNG at 300 dpi; vector SVG/PDF with text; R-generated low-resolution QA previews.",
  "Blue/green/red match the reference; recurring line styles distinguish configurations within each family."
),file.path(out,"figure_notes.md"))
print(tab)
cat("Created two fixed-sum figures in",out,"\n")
