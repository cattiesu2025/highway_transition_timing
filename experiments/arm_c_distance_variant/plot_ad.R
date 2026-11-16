#!/usr/bin/env Rscript
# Contract: quantitative grid; describe selected-policy responses within A and D.
# No claim of equal endpoint meaning, independent controls, or held-out confirmation.
# R-only drawing/export. Source aggregation is non-visual and separately audited.
.libPaths(c('tmp/r-lib',.libPaths()))
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork)})
stopifnot(requireNamespace('ragg',quietly=TRUE),requireNamespace('svglite',quietly=TRUE))
out <- 'outputs/arm_ad_figures'; src <- file.path(out,'source_data')
read <- function(n) read.csv(file.path(src,n),stringsAsFactors=FALSE)
write <- function(x,n) write.csv(x,file.path(src,n),row.names=FALSE)
agents <- c('FD','BAL','SP'); colors <- c(FD='#397F96',BAL='#7B65A8',SP='#CA7B22')
ink <- '#263543'; muted <- '#63717C'; line <- '#E9EEF1'
set.seed(20261026)
boot <- function(x) {
 x <- x[is.finite(x)]
 if(!length(x)) return(data.frame(mid=NA_real_,lo=NA_real_,hi=NA_real_,n=0))
 b <- replicate(5000,median(sample(x,length(x),replace=TRUE)))
 data.frame(mid=median(x),lo=unname(quantile(b,.025)),hi=unname(quantile(b,.975)),n=length(x))
}
theme_set(theme_classic(base_size=11,base_family='Arial')+theme(
 text=element_text(colour=ink),axis.text=element_text(colour=ink),
 axis.line=element_line(colour='#9AA6AE',linewidth=.3),axis.ticks=element_line(colour='#9AA6AE',linewidth=.3),
 panel.grid.major.y=element_line(colour=line,linewidth=.3),
 plot.title=element_text(face='bold',size=13,margin=margin(b=5)),
 plot.subtitle=element_text(size=10,colour=muted,margin=margin(b=12)),
 plot.margin=margin(10,16,9,10),legend.position='none'))
main <- read('main_episodes.csv') %>% mutate(agent=factor(agent,agents),observed=tolower(observed)=='true')
cf <- read('controls_by_seed.csv') %>% mutate(agent=factor(agent,agents))
status <- read('run_status.csv')
stopifnot(nrow(main)==4248,nrow(cf)==472,all(cf$n_total==36))
seed_times <- main %>% filter(observed) %>% group_by(arm,agent,seed) %>%
 summarise(seconds=median(seconds),n_episodes=n(),.groups='drop')
centers <- seed_times %>% group_by(arm,agent) %>% group_modify(~boot(.x$seconds)) %>% ungroup()
write(seed_times,'main_times_by_seed.csv');write(centers,'main_time_summary.csv')
complete <- status %>% filter(status=='complete') %>% count(arm,seed) %>% filter(n==3)
stopifnot(sum(complete$arm=='A')==20,sum(complete$arm=='D')==18)
paired <- inner_join(main,complete %>% select(arm,seed),by=c('arm','seed')) %>% filter(observed)
pairs <- list(c('FD','BAL'),c('FD','SP'),c('BAL','SP'))
deltas <- bind_rows(lapply(pairs,function(pair) {
 keys <- c('arm','seed','exposure_id')
 a <- paired %>% filter(agent==pair[1]) %>% select(all_of(keys),a=seconds)
 b <- paired %>% filter(agent==pair[2]) %>% select(all_of(keys),b=seconds)
 inner_join(a,b,by=keys) %>% mutate(delta=a-b) %>% group_by(arm,seed) %>%
 summarise(delta=median(delta),n_exposures=n(),.groups='drop') %>% mutate(contrast=paste(pair,collapse=' − '))
}))
stopifnot(all(deltas$n_exposures==36))
contrasts <- deltas %>% group_by(arm,contrast) %>% group_modify(~boot(.x$delta)) %>% ungroup() %>%
 mutate(contrast=factor(contrast,rev(c('FD − BAL','FD − SP','BAL − SP'))))
write(deltas,'paired_time_differences_by_seed.csv');write(contrasts,'paired_time_difference_summary.csv')
cf_sum <- cf %>% group_by(arm,agent,control) %>% group_modify(~ {
 d<-boot(.x$seconds);d$n_observed<-sum(.x$n_observed);d$n_total<-sum(.x$n_total)
 d$n_seed_total<-nrow(.x);d
}) %>% ungroup() %>% mutate(percent=100*n_observed/n_total)
write(cf_sum,'control_summary.csv')

savefig <- function(p,name,height) {
 for(ext in c('png','pdf','svg')) {
  dev<-switch(ext,png=ragg::agg_png,pdf=grDevices::cairo_pdf,svg=svglite::svglite)
  ggsave(file.path(out,paste0(name,'.',ext)),p,device=dev,width=11.2,height=height,units='in',dpi=300,bg='white')
 }
 ggsave(file.path(out,paste0(name,'_preview.png')),p,device=ragg::agg_png,width=11.2,height=height,units='in',dpi=150,bg='white')
}
mainpanel <- function(a) {
 d<-filter(seed_times,arm==a);s<-filter(centers,arm==a)
 title<-if(a=='A') 'Arm A · Single lane' else 'Arm D · Yellow car farther ahead'
 subtitle<-if(a=='A') 'ONSET = ego slowdown time' else 'ONSET = ego lane-crossing time'
 ggplot(d,aes(agent,seconds,colour=agent))+
 geom_point(position=position_jitter(width=.15,height=0,seed=91),alpha=.53,size=2.2)+
 geom_errorbar(data=s,aes(y=mid,ymin=lo,ymax=hi),width=.13,linewidth=.85)+
 geom_point(data=s,aes(y=mid),shape=18,size=4.2)+
 geom_text(data=s,aes(y=mid,label=sprintf('%.2f s',mid)),nudge_x=.23,hjust=0,size=3.5,fontface='bold')+
 scale_colour_manual(values=colors)+scale_y_continuous(limits=c(0,4),breaks=0:4,expand=expansion(mult=c(0,.03)))+
 labs(title=title,subtitle=subtitle,x=NULL,y='Onset time (s)')
}
contrastpanel <- function(a) {
 d<-filter(contrasts,arm==a)
 title<-if(a=='A') 'c   Paired time differences' else 'd   Paired time differences'
 subtitle<-if(a=='A') '20 complete seeds · within-scene differences' else '18 complete seeds · within-scene differences'
 ggplot(d,aes(mid,contrast))+
 geom_vline(xintercept=0,colour='#9AA6AE',linetype='22',linewidth=.5)+
 geom_errorbar(aes(xmin=lo,xmax=hi),orientation='y',width=0,linewidth=1.1,colour='#355C70')+
 geom_point(shape=21,fill='#355C70',colour='white',stroke=.6,size=3.2)+
 scale_x_continuous(limits=c(-2,.5),breaks=seq(-2,.5,.5))+
 labs(title=title,subtitle=subtitle,x='First preference − second preference (s)',y=NULL)+
 theme(panel.grid.major.y=element_blank(),panel.grid.major.x=element_line(colour=line,linewidth=.3))
}
annotation_theme<-theme(plot.title=element_text(size=20,face='bold',colour=ink),
 plot.subtitle=element_text(size=11,colour=muted),
 plot.caption=element_text(size=9,hjust=0,colour=muted,lineheight=1.2),plot.margin=margin(16,18,12,18))
fig1 <- (mainpanel('A')|mainpanel('D')) +
 plot_annotation(caption=paste('Dots: seed medians. Diamonds: median across seeds; bars: 95% seed-bootstrap CI. 36 scenes per seed; t = 0 responses retained.',
 'A: 20 seeds per preference. D: FD 19 / BAL 19 / SP 20 eligible seeds; 2 of 60 policies failed selection.',sep='\n'),theme=annotation_theme)
savefig(fig1,'01_AD_response_timing',4.8)

control_names<-c(original='Slow white car',`far-front`='White car farther away',`matched-speed-front`='White car at ego speed',`no-front`='White car removed')
levels_control<-rev(names(control_names))
cf<-cf %>% mutate(y=match(control,levels_control)+(2-as.numeric(agent))*.20)
cf_sum<-cf_sum %>% mutate(y=match(control,levels_control)+(2-as.numeric(agent))*.20)
controlpanel <- function(a) {
 d<-filter(cf,arm==a,control!='no-front');s<-filter(cf_sum,arm==a,control!='no-front')
 ggplot(d,aes(seconds,y,colour=agent,shape=agent))+
 geom_point(position=position_jitter(width=0,height=.025,seed=27),alpha=.25,size=1.7)+
 geom_errorbar(data=s,aes(x=mid,xmin=lo,xmax=hi),orientation='y',width=0,linewidth=.9)+
 geom_point(data=s,aes(x=mid),size=3)+
 scale_colour_manual(values=colors)+scale_shape_manual(values=c(FD=16,BAL=15,SP=18))+
 scale_y_continuous(breaks=2:4,labels=control_names[levels_control[2:4]],limits=c(1.6,4.5),expand=c(0,0))+
 scale_x_continuous(limits=c(0,if(a=='A') 24 else 60),breaks=if(a=='A') seq(0,24,6) else seq(0,60,15),expand=expansion(mult=c(.01,.03)))+
 labs(title=if(a=='A') 'a   A · Single lane' else 'b   D · Yellow car farther ahead',
 subtitle=if(a=='A') 'Sustained deceleration · 20 seeds per preference' else 'Actual lane crossing · FD 19 / BAL 19 / SP 20 seeds',
 x='Response time among observed events (s)',y=NULL,colour=NULL,shape=NULL)+
 theme(panel.grid.major.y=element_blank(),panel.grid.major.x=element_line(colour=line,linewidth=.3))
}
ratepanel<-function(a) {
 d<-filter(cf_sum,arm==a) %>% mutate(control=factor(control,levels_control),
 label=paste0(sprintf('%.0f%%',percent),'\n',n_observed,'/',n_total))
 ggplot(d,aes(agent,control))+
 geom_tile(aes(fill=percent),width=.94,height=.91,colour='white',linewidth=.6)+
 geom_text(aes(label=label),size=3.0,lineheight=.95,colour=ink)+
 scale_fill_gradient(low='#F1F3F5',high='#D7E6ED',limits=c(0,100),guide='none')+
 scale_y_discrete(labels=control_names)+scale_x_discrete(position='top')+
 labs(title=if(a=='A') 'c   Response occurrence' else 'd   Response occurrence',
 subtitle='Observed episodes / all evaluated episodes',x=NULL,y=NULL)+
 theme(axis.line=element_blank(),axis.ticks=element_blank(),panel.grid=element_blank(),
 axis.text.x=element_text(face='bold'),plot.subtitle=element_text(margin=margin(b=5)))
}
fig2<- (controlpanel('A')|controlpanel('D')) / (ratepanel('A')|ratepanel('D')) +
 plot_layout(heights=c(1.2,1),guides='collect')+
 plot_annotation(title='How do responses change when the white car changes?',
 subtitle='FD · front-distance preference     BAL · intermediate preference     SP · speed preference',
 caption=paste('Points: seed medians; large symbols: across-seed median with 95% bootstrap CI. Panel time scales differ; all horizons are 120 s.',
 'Removing the white car produces no events in either arm; no time estimate is assigned. D retains the yellow car in this control.',
 'Selected policies only: A 60/60; D 58/60. These controls also informed checkpoint selection; this is not independent validation.',sep='\n'),theme=annotation_theme) & theme(legend.position='top',legend.text=element_text(size=10))
savefig(fig2,'02_AD_hazard_controls',8.3)
print(centers);print(contrasts)
writeLines(c(
 '# A + D figure notes',
 'Preview design: two quantitative grids. Main claim is descriptive: selected-policy response times and paired differences within each arm.',
 'A uses sustained-deceleration valid onset; D uses first actual lane-index change at the post-step sample, not the command onset used by its gate.',
 'D physical time uses the explicit seconds column, including the +1 post-step correction. These endpoint definitions are not interchangeable.',
 'A: 20 seeds per preference. D: FD 19, BAL 19, SP 20; FD/4209 and BAL/4205 failed all-checkpoint selection and are reported as failures.',
 'Main panels include all eligible policies; labels are medians, not means. Paired contrasts are retained in source tables, not plotted. All D paired contrasts use the same 18 complete seeds; A uses 20. Differences are paired by scene within seed, then median per seed.',
 'Each seed has 36 scenes. Main time includes t=0 observations; event-free episodes are missing, never zero or 120 s. All selected main-scene episodes here have events.',
 'Centers are medians across seed medians. CIs are percentile bootstrap with 5,000 seed resamples, R seed 20261026. Contrasts bootstrap seed-level paired differences.',
 'Intervals are descriptive, unadjusted for multiple contrasts. No p-values, significance colors, causal mechanism claims or cross-arm significance tests are shown.',
 'All control event rates use all episodes of eligible policies as denominators; failed training runs are separately reported, not counted as no-event episodes.',
 'A uses existing selected-weight results (first five seeds informed weight selection); D is the new distance-only extension. Both are development results, not held-out confirmation.',
 'Controls participated in selection; absence of no-front events among selected policies is not independent evidence of specificity.',
 'Exports: R-generated PNG at 300 dpi, vector PDF/SVG, and 150 dpi previews. Shared FD/BAL/SP colors; shape encoding added for controls.',
 'Source provenance, raw episode table, per-seed controls, status, bootstraps and paired summaries are in source_data/.'
),file.path(out,'figure_notes.md'))
