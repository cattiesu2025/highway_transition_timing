#!/usr/bin/env Rscript
# Figure contract: separate learning progress from behaviour for selected weights F1/F4/F7.
# Top: each seed's return rescaled to its own start and plateau, so scoring scale is removed.
# Bottom: weight-independent speed and capped front gap of trained policies on evaluation scenes.
# Arm D FD/4209 and BAL/4205 failed checkpoint selection, so they have no behaviour points.
.libPaths(c('tmp/r-lib',.libPaths()))
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork)})
out<-'outputs/learning_vs_behaviour';src<-file.path(out,'source_data')
curves<-read.csv('outputs/four_arm_training_curves/source_data/aligned_seed_curves.csv')
behaviour<-read.csv(file.path(src,'behaviour_by_seed.csv'))
stopifnot(nrow(curves)==240*99,nrow(behaviour)==238,setequal(behaviour$arm,c('A','B','C','D')),all(behaviour$collision_rate==0))
agents<-c('FD','BAL','SP');arms<-c('A','B','C','D')
arm_labels<-c(A='Arm A · Single lane',B='Arm B · Empty target lane',C='Arm C · Occupied target lane',D='Arm D · Yellow car farther')
cols<-c(FD='#1B7FB5',BAL='#C13F8E',SP='#DE8C14');ltys<-c(FD='solid',BAL='22',SP='12');ink<-'#263543';muted<-'#63717C'
progress<-curves %>% group_by(arm,agent,seed) %>% arrange(timesteps,.by_group=TRUE) %>%
 mutate(start=episode_reward[timesteps==1000],plateau=mean(episode_reward[timesteps>=90000]),
 progress=(episode_reward-start)/(plateau-start)) %>% ungroup()
stopifnot(all(is.finite(progress$progress)),all(progress$plateau>progress$start))
summary_curve<-progress %>% group_by(arm,agent,timesteps) %>%
 summarise(mid=median(progress),lo=quantile(progress,.25),hi=quantile(progress,.75),.groups='drop')
# First step after which a seed stays at or above the threshold, which ignores brief early crossings.
reach<-function(step,value,level){below<-which(value<level);if(!length(below)) step[1] else if(max(below)==length(step)) NA_real_ else step[max(below)+1]}
timing<-progress %>% group_by(arm,agent,seed) %>% summarise(t50=reach(timesteps,progress,.5),t90=reach(timesteps,progress,.9),.groups='drop')
timing_summary<-timing %>% group_by(arm,agent) %>% summarise(t50_median=median(t50,na.rm=TRUE),t50_iqr=IQR(t50,na.rm=TRUE),
 t90_median=median(t90,na.rm=TRUE),t90_iqr=IQR(t90,na.rm=TRUE),t90_missing=sum(is.na(t90)),.groups='drop')
write.csv(summary_curve,file.path(src,'progress_median_iqr.csv'),row.names=FALSE)
write.csv(timing,file.path(src,'progress_timing_by_seed.csv'),row.names=FALSE)
write.csv(timing_summary,file.path(src,'progress_timing_summary.csv'),row.names=FALSE)
print(as.data.frame(timing_summary))
facet_arm<-function(x) factor(x,arms,arm_labels[arms])
base<-theme_classic(base_size=10.5,base_family='Arial')+theme(text=element_text(colour=ink),axis.text=element_text(colour=ink),
 axis.line=element_line(colour='#9AA6AE',linewidth=.3),axis.ticks=element_line(colour='#9AA6AE',linewidth=.3),
 panel.grid.major.y=element_line(colour='#E9EEF1',linewidth=.3),strip.background=element_blank(),
 strip.text=element_text(face='bold',hjust=0,size=10.5),panel.spacing=grid::unit(6,'mm'),
 plot.title=element_text(size=12.5,face='bold',margin=margin(b=3)),plot.subtitle=element_text(size=9.5,colour=muted,margin=margin(b=6)))
summary_curve$arm_f<-facet_arm(summary_curve$arm);summary_curve$agent<-factor(summary_curve$agent,agents)
epsilon<-data.frame(arm_f=factor(unname(arm_labels['A']),unname(arm_labels[arms])),x=25,y=.2,label='ε reaches 0.05\nat 25k')
p1<-ggplot(summary_curve,aes(timesteps/1000,mid,colour=agent,fill=agent,linetype=agent))+
 geom_hline(yintercept=c(0,1),colour='#C9D1D6',linewidth=.3)+
 geom_vline(xintercept=25,colour=muted,linewidth=.35,linetype='dashed')+
 geom_ribbon(aes(ymin=lo,ymax=hi),alpha=.14,colour=NA)+geom_line(linewidth=.8)+
 geom_text(data=epsilon,aes(x+3,y,label=label),inherit.aes=FALSE,hjust=0,size=2.9,colour=muted,lineheight=.95,family='Arial')+
 facet_wrap(~arm_f,nrow=1)+scale_colour_manual(values=cols)+scale_fill_manual(values=cols)+scale_linetype_manual(values=ltys)+
 scale_x_continuous(breaks=c(0,25,50,75,100),limits=c(0,100),expand=expansion(mult=c(0,.02)))+
 scale_y_continuous(breaks=c(0,.5,1),labels=c('0 (start)','0.5','1 (plateau)'))+
 labs(title='Learning progress, each seed rescaled to its own start and plateau',subtitle='Removes the height difference between reward conditions',
 x='Training steps (× 1,000)',y='Learning progress',colour=NULL,fill=NULL,linetype=NULL)+
 base+theme(legend.position='top',legend.justification='left',legend.key.width=grid::unit(11,'mm'),legend.margin=margin(0,0,2,0))
long<-bind_rows(lapply(c('speed_ms','front_gap_m'),function(m) data.frame(behaviour[c('arm','agent','seed')],metric=m,value=behaviour[[m]])))
metric_labels<-c(speed_ms='Mean speed (m/s)',front_gap_m='Mean front gap (m)')
long$metric_f<-factor(metric_labels[long$metric],metric_labels);long$agent<-factor(long$agent,agents);long$arm_f<-facet_arm(long$arm)
medians<-long %>% group_by(arm_f,metric_f,agent) %>% summarise(value=median(value),.groups='drop')
limits<-bind_rows(data.frame(metric_f=factor(unname(metric_labels[1]),unname(metric_labels)),value=c(10,35)),data.frame(metric_f=factor(unname(metric_labels[2]),unname(metric_labels)),value=c(60,125)))
limit_rows<-merge(data.frame(arm_f=factor(arm_labels[arms],arm_labels[arms])),limits)
set.seed(20261105)
p2<-ggplot(long,aes(agent,value))+geom_blank(data=limit_rows,aes(x=factor('BAL',agents),value),inherit.aes=FALSE)+
 geom_point(aes(colour=agent,shape=agent),position=position_jitter(width=.14,height=0),size=1.25,alpha=.5,stroke=0)+
 geom_crossbar(data=medians,aes(ymin=value,ymax=value),width=.56,colour=ink,linewidth=0,middle.linewidth=.45)+
 facet_grid(metric_f~arm_f,scales='free_y',switch='y',labeller=labeller(arm_f=function(x) rep('',length(x))))+
 scale_x_discrete(limits=agents,drop=FALSE)+scale_colour_manual(values=cols,guide='none')+scale_shape_manual(values=c(FD=16,BAL=17,SP=15),guide='none')+
 labs(title='Behaviour of trained policies, independent of reward weights',subtitle='Dots: training seeds; black bar: median. First 20 s of evaluation scenes; no collisions in any run.',x=NULL,y=NULL)+
 base+theme(strip.placement='outside',strip.text.x=element_blank(),strip.text.y.left=element_text(angle=90,face='plain',hjust=.5,size=9.5))
figure<-p1/p2+plot_layout(heights=c(1,1.75))+plot_annotation(
 title='Learning runs on a similar schedule; speed and gap differ in Arms A and C, not B and D',
 subtitle='Selected weights · FD (front 1.12, speed 0.28) · BAL (0.84, 0.56) · SP (0.42, 0.98)',
 caption=paste('Top: thick line is the median across 20 seeds; band is the IQR (25–75%). Start = rolling return at 1k steps; plateau = mean over 90–99k.','Bottom: 20 seeds per condition, except Arm D FD and BAL with 19 (one seed each failed checkpoint selection). Front gap counts no front vehicle as 120 m, the reward scoring horizon.',sep='\n'),
 theme=theme(text=element_text(family='Arial',colour=ink),plot.title=element_text(size=17,face='bold',margin=margin(b=5)),
 plot.subtitle=element_text(size=10.5,colour=muted,margin=margin(b=10)),plot.caption=element_text(size=8.5,colour=muted,hjust=0,lineheight=1.2,margin=margin(t=10)),plot.margin=margin(14,16,10,14)))
for(ext in c('png','pdf','svg')){dev<-switch(ext,png=ragg::agg_png,pdf=grDevices::cairo_pdf,svg=svglite::svglite)
 ggsave(file.path(out,paste0('learning_vs_behaviour.',ext)),figure,device=dev,width=11.2,height=8.4,units='in',dpi=300,bg='white')}
ggsave(file.path(out,'learning_vs_behaviour_preview.png'),figure,device=ragg::agg_png,width=11.2,height=8.4,units='in',dpi=130,bg='white')
