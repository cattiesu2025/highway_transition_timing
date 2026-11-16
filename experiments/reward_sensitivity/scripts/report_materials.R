#!/usr/bin/env Rscript
# English presentation materials: A/C development comparison, R-only rendering.
# Contract: table describes timing across all weights; training grid describes
# logged returns, not performance rankings; counterfactual grid reports both
# behavior occurrence and collisions. Five training seeds are the replicates.
# 16:9 slide-sized quantitative grids, editable SVG/PDF and 300-dpi PNG.
.libPaths(c("tmp/r-lib", .libPaths()))
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(patchwork)})
stopifnot(requireNamespace("ragg", quietly=TRUE), requireNamespace("svglite", quietly=TRUE))
out <- "outputs/reward_report_en"
dir.create(out, recursive=TRUE, showWarnings=FALSE)
dir.create(file.path(out, "source_data"), showWarnings=FALSE)
dir.create("tmp/reward_report_preview", recursive=TRUE, showWarnings=FALSE)
original_ids <- c("FD", "BAL", "SP", paste0("R", 1:8))
ids <- c("R7","R2","FD","R1","R3","R4","BAL","R5","R6","R8","SP")
reward_groups <- setNames(c(rep("FD-like",3),rep("Intermediate",5),rep("SP-like",3)),ids)
cf_ids <- c("FD", "BAL", "SP", "R7", "R8")
conditions <- c("original", "no-front", "matched-speed-front", "far-front")
ink <- "#243746"; blue <- "#397F96"; orange <- "#BA6D35"
read_table <- function(experiment, name) read.csv(file.path("outputs", experiment, "analysis", paste0(name, ".csv")), stringsAsFactors=FALSE)
combined <- function(name) {
  a <- read_table("reward_sensitivity", name) %>% filter(arm %in% c("A", "C"))
  b <- read_table("reward_symmetric", name) %>% filter(variant %in% c("R7", "R8"))
  bind_rows(a,b)
}
seed <- combined("seed_summary")
trace <- combined("training_progress")
status <- combined("run_status")
stopifnot(nrow(seed)==110, nrow(status)==110, all(status$status=="complete"),
          all(seed$n_onset==36), all(seed$n_collision==0),
          all(count(seed, arm, variant)$n==5), !anyDuplicated(seed[c("arm","variant","seed")]))
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
  ggsave(file.path("tmp/reward_report_preview",paste0(name,".png")),p,
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
cells <- bind_rows(lapply(seq_len(ncol(tab)),function(j) data.frame(x=xs[j],y=12-seq_len(11),
  label=if(j %in% c(3,4,5)) sprintf("%.2f",tab[[j]]) else tab[[j]],hjust=if(j<=2) 0 else .5)))
tbl <- ggplot() +
  annotate("rect",xmin=0,xmax=15.1,ymin=11.65,ymax=13.15,fill="#E8F0F4") +
  geom_rect(data=data.frame(y=12-seq_len(11)),aes(xmin=0,xmax=15.1,ymin=y-.43,ymax=y+.43),
            fill=c(rep("#EFF5FC",3),rep("#EFF7F1",5),rep("#FCF0EF",3)),colour=NA) +
  annotate("segment",x=0,xend=15.1,y=c(11.6,8.5,3.5,.4),yend=c(11.6,8.5,3.5,.4),colour="#CBD7DE",linewidth=.6) +
  geom_text(data=data.frame(x=xs,y=12.4,label=headers,hjust=c(0,0,.5,.5,.5,.5,.5)),
            aes(x,y,label=label,hjust=hjust),size=5.7,fontface="bold",colour=ink,lineheight=1.1) +
  geom_text(data=cells,aes(x,y,label=label,hjust=hjust),size=5.8,colour=ink) +
  coord_cartesian(xlim=c(0,15.1),ylim=c(.3,13.2),expand=FALSE,clip="off") + theme_void(base_family="Arial") +
  labs(title="Reward configurations and behavioral outcomes",
       subtitle="A: single-lane slowdown   |   C: lane change with an occupied target lane   |   5 paired training seeds per configuration",
       caption=paste("Cells: median [Q1, Q3] of 5 seed-specific onset medians; 36 episodes per seed.",
                     "Groups use r = spacing / speed: FD-like r > 2; Intermediate 0.5 <= r <= 2; SP-like r < 0.5.", sep="\n")) +
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
reward_colors <- c(R7="#123E75",R2="#2764AC",FD="#428BC5",
                   R1="#124D2C",R3="#23723E",R4="#3D9555",BAL="#68AC73",R5="#91BE96",
                   R6="#9B2335",R8="#D04D58",SP="#E58B86")
reward_lines <- setNames(c(rep("longdash",3),rep("solid",8)),original_ids)
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
legend_data <- data.frame(variant=ids, x=c(seq(3.7,10.3,3.3),seq(3.7,16.9,3.3),seq(3.7,10.3,3.3)),
                          y=c(rep(3,3),rep(2,5),rep(1,3)))
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
  plot_annotation(title="Training curves by reward-ratio group",
    subtitle="Blue: FD-like | Green: Intermediate | Red: SP-like. Lines: median across 5 seeds; shading: IQR.",
    caption="Weights: (speed, spacing).",
    theme=theme(plot.title=element_text(size=28,face="bold",colour=ink),plot.subtitle=element_text(size=17,margin=margin(t=8,b=8)),
                plot.caption=element_text(size=13,hjust=0,lineheight=1.35),plot.margin=margin(22,25,20,25)))
save_material(training,"02_training_curves")

# Counterfactuals: five configurations have all four variants in both arms.
cf <- combined("counterfactual_summary") %>% filter(variant %in% cf_ids)
stopifnot(nrow(cf)==200,all(count(cf,arm,variant,seed)$n==4),
          !anyDuplicated(cf[c("arm","variant","seed","counterfactual_variant")]))
cf$n_eval <- ifelse(cf$arm=="A",cf$n,cf$n_episodes)
cf$n_onset <- ifelse(cf$arm=="A",cf$valid_onset_count,cf$actual_lane_change_episode_count)
cf$n_collision <- cf$collision_episode_count
for(i in which(cf$arm=="A")) {
  rr <- status %>% filter(arm==cf$arm[i],variant==cf$variant[i],seed==cf$seed[i])
  stopifnot(nrow(rr)==1)
  f <- file.path("outputs",basename(rr$detail),"rollout_counterfactual_front_vehicle",cf$counterfactual_variant[i],"analysis","episode_outcomes.csv")
  episodes <- read.csv(f,stringsAsFactors=FALSE) %>% filter(agent_condition==cf$agent[i])
  stopifnot(nrow(episodes)==36,sum(episodes$episode_outcome=="valid_onset")==cf$n_onset[i])
  cf$n_collision[i] <- sum(tolower(episodes$collision_flag)=="true")
}
stopifnot(all(cf$n_eval==36),all(is.finite(cf$n_collision)),all(is.finite(cf$n_onset)))
cf_summary <- cf %>% group_by(arm,variant,counterfactual_variant) %>%
  summarise(n_seeds=n(),episodes=sum(n_eval),onsets=sum(n_onset),collisions=sum(n_collision),
            onset_pct=100*mean(n_onset/n_eval),collision_pct=100*mean(n_collision/n_eval),.groups="drop")
stopifnot(all(cf_summary$n_seeds==5),all(cf_summary$episodes==180))
write.csv(cf,file.path(out,"source_data","counterfactual_by_seed.csv"),row.names=FALSE)
write.csv(cf_summary,file.path(out,"source_data","counterfactual_rates.csv"),row.names=FALSE)
cf_summary$variant <- factor(cf_summary$variant,levels=rev(cf_ids))
cf_summary$counterfactual_variant <- factor(cf_summary$counterfactual_variant,levels=conditions)
heat <- function(arm_id,metric,title,col) {
  d <- cf_summary %>% filter(arm==arm_id);d$value <- d[[metric]]
  d$label <- ifelse(abs(d$value-round(d$value))<.001,sprintf("%.0f%%",d$value),sprintf("%.1f%%",d$value))
  ggplot(d,aes(counterfactual_variant,variant)) +
    geom_tile(aes(fill=value),colour="white",linewidth=2,width=.98,height=.95) +
    geom_text(aes(label=label,colour=value>80),size=6.4,fontface="bold") +
    scale_colour_manual(values=c("FALSE"=ink,"TRUE"="white"),guide="none") +
    scale_fill_gradient(low="#F1F5F7",high=col,limits=c(0,100),guide="none") +
    scale_x_discrete(labels=c("Original","No front\nvehicle","Matched-speed\nfront vehicle","Far front\nvehicle"),position="top",expand=c(.01,0)) +
    scale_y_discrete(expand=c(.01,0)) + labs(title=title,x=NULL,y=NULL) +
    theme_minimal(base_size=16,base_family="Arial") +
    theme(panel.grid=element_blank(),axis.text.x=element_text(size=14,colour=ink),axis.text.y=element_text(size=17,face="bold",colour=ink),
          plot.title=element_text(size=19,face="bold",colour=ink,margin=margin(b=12)),plot.margin=margin(12,20,12,20))
}
counter <- (heat("A","onset_pct","a   Arm A | Slowdown occurrence",blue) |
             heat("A","collision_pct","b   Arm A | Collision rate",orange)) /
            (heat("C","onset_pct","c   Arm C | Actual lane-change occurrence",blue) |
             heat("C","collision_pct","d   Arm C | Collision rate",orange)) +
  plot_annotation(title="Counterfactual behavior and collision outcomes",
    subtitle="No-front controls suppress the target behavior; matched-speed fronts expose collision risk in Arm C.",
    caption=paste("Cells: mean episode rate across 5 seeds (36 episodes per seed; 180 per cell). Both color scales span 0-100%; occurrence and collision can coexist.",
                  "Coverage: FD, BAL, SP, R7 and R8. Counterfactual evaluations for R1-R6 are unavailable. Development evaluation; descriptive results.",sep="\n"),
    theme=theme(plot.title=element_text(size=28,face="bold",colour=ink),plot.subtitle=element_text(size=17,margin=margin(b=8)),
                plot.caption=element_text(size=13,hjust=0,lineheight=1.35),plot.margin=margin(22,22,22,22)))
save_material(counter,"03_counterfactual_results")

md <- c("# English presentation materials", "",
        "## Reward configurations and behavioral outcomes", "",
        paste0("| ",paste(names(tab),collapse=" | ")," |"),
        paste0("| ",paste(rep("---",ncol(tab)),collapse=" | ")," |"),
        apply(tab,1,function(r)paste0("| ",paste(r,collapse=" | ")," |")), "",
        "Values are median [Q1, Q3] across five training-seed medians, each based on 36 development episodes. All original-scene A/C episodes have observed onset and no collision.", "",
        "## Training trajectories", "",
        "Eleven reward configurations, five paired seeds per arm. Arm A is on the left and Arm C on the right. Colors identify configurations consistently; dashed lines mark FD/BAL/SP baselines. Lines and ribbons show median and IQR on a 1K-step grid within observed support. Individual seed trajectories are omitted to reduce overlap. No extrapolation to 100K. Raw return levels are not comparable performance scores across reward functions. No statistical convergence claim is made.", "",
        "## Counterfactual evaluation", "",
        "Five configurations have complete counterfactual coverage in both arms. Rates average over five seeds with 36 episodes each. A measures persistent slowdown; C measures actual physical lane change. Collision is recorded independently of onset. R1-R6 counterfactual evaluations are unavailable.", "",
        "All three materials use the same A/C seed sets (3100-3104 and 4200-4204). IQR is descriptive variability, not a confidence interval. No hypothesis tests are shown. R7/R8 are targeted exploratory follow-ups, not additional random configurations.")
writeLines(md,file.path(out,"report_notes.md"))
cat("Created three English materials in",out,"\n")
print(tab)
print(cf_summary)
