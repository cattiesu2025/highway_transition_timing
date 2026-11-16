#!/usr/bin/env Rscript
# Appendix figure contract: show all four arms, all eligible seeds, and three fixed preferences.
# R-only quantitative 2x2 grid. Distinct endpoint definitions are stated in each panel.
.libPaths(c('tmp/r-lib',.libPaths()))
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork)})
stopifnot(requireNamespace('ragg',quietly=TRUE),requireNamespace('svglite',quietly=TRUE))
out<-'outputs/arm_abcd_onset';src<-file.path(out,'source_data');dir.create(src,recursive=TRUE,showWarnings=FALSE)
abc<-read.csv('outputs/reward_selected_validation/analysis/episode_outcomes.csv',stringsAsFactors=FALSE)
ad<-read.csv('outputs/arm_ad_figures/source_data/main_episodes.csv',stringsAsFactors=FALSE)
abc_status<-read.csv('outputs/reward_selected_validation/analysis/run_status.csv',stringsAsFactors=FALSE)
ad_status<-read.csv('outputs/arm_ad_figures/source_data/run_status.csv',stringsAsFactors=FALSE)
stopifnot(nrow(abc_status)==180,all(abc_status$status=='complete'))
main<-bind_rows(abc %>% transmute(arm,agent,seed,exposure_id=sub('^[^_]+_([^_]+)_.*$','\\1',episode_id),
 observed=episode_outcome=='valid_onset',seconds=response_latency_seconds),
 ad %>% filter(arm=='D') %>% transmute(arm,agent,seed,exposure_id,observed=tolower(observed)=='true',seconds))
stopifnot(nrow(main)==238*36,!anyDuplicated(main[c('arm','agent','seed','exposure_id')]),all(is.finite(main$seconds[main$observed])))
status<-bind_rows(abc_status %>% select(arm,agent,seed,status),ad_status %>% filter(arm=='D'))
write.csv(main,file.path(src,'main_episodes.csv'),row.names=FALSE)
write.csv(status,file.path(src,'run_status.csv'),row.names=FALSE)
coverage<-main %>% group_by(arm,agent,seed) %>% summarise(total=n(),observed=sum(observed),.groups='drop')
stopifnot(all(coverage$total==36),all(coverage$observed==36))
write.csv(coverage,file.path(src,'event_coverage.csv'),row.names=FALSE)
seeds<-main %>% filter(observed) %>% group_by(arm,agent,seed) %>% summarise(seconds=median(seconds),n_episodes=n(),.groups='drop')
counts<-seeds %>% count(arm,agent)
stopifnot(nrow(counts)==12,all(counts$n[counts$arm!='D']==20),all(counts$n[counts$arm=='D']==c(19,19,20)))
# Reuse the already displayed A/D intervals exactly; bootstrap B/C identically.
ad_summary<-read.csv('outputs/arm_ad_figures/source_data/main_time_summary.csv')
for(i in seq_len(nrow(ad_summary))) {
 row<-ad_summary[i,]; x<-seeds %>% filter(arm==row$arm,agent==row$agent)
 stopifnot(abs(median(x$seconds)-row$mid)<1e-10,nrow(x)==row$n)
}
set.seed(20261027)
boot<-function(x) {
 b<-replicate(5000,median(sample(x,length(x),replace=TRUE)))
 data.frame(mid=median(x),lo=unname(quantile(b,.025)),hi=unname(quantile(b,.975)),n=length(x))
}
summary<-bind_rows(ad_summary,seeds %>% filter(arm %in% c('B','C')) %>% group_by(arm,agent) %>% group_modify(~boot(.x$seconds)) %>% ungroup()) %>% arrange(arm,agent)
write.csv(seeds,file.path(src,'onset_times_by_seed.csv'),row.names=FALSE)
write.csv(summary,file.path(src,'onset_time_summary.csv'),row.names=FALSE)
seeds$agent<-factor(seeds$agent,c('FD','BAL','SP'));summary$agent<-factor(summary$agent,c('FD','BAL','SP'))
colors<-c(FD='#397F96',BAL='#7B65A8',SP='#CA7B22');ink<-'#263543';muted<-'#63717C'
titles<-c(A='Arm A · Single lane',B='Arm B · Empty target lane',C='Arm C · Occupied target lane',D='Arm D · Yellow car farther ahead')
theme_set(theme_classic(base_size=11,base_family='Arial')+theme(text=element_text(colour=ink),axis.text=element_text(colour=ink),
 axis.line=element_line(colour='#9AA6AE',linewidth=.3),axis.ticks=element_line(colour='#9AA6AE',linewidth=.3),
 panel.grid.major.y=element_line(colour='#E9EEF1',linewidth=.3),plot.title=element_text(size=13,face='bold',margin=margin(b=5)),
 plot.subtitle=element_text(size=10,colour=muted,margin=margin(b=12)),legend.position='none',plot.margin=margin(12,16,9,10)))
panel<-function(a) {
 d<-filter(seeds,arm==a);s<-filter(summary,arm==a)
 ymax<-10
 ggplot(d,aes(agent,seconds,colour=agent))+
 geom_point(position=position_jitter(width=.15,height=0,seed=91),alpha=.53,size=2.2)+
 geom_errorbar(data=s,aes(y=mid,ymin=lo,ymax=hi),width=.13,linewidth=.85)+
 geom_point(data=s,aes(y=mid),shape=18,size=4.2)+
 geom_text(data=s,aes(y=mid,label=sprintf('%.2f s',mid)),nudge_x=.23,hjust=0,size=3.5,fontface='bold')+
 scale_colour_manual(values=colors)+scale_y_continuous(limits=c(0,ymax),breaks=seq(0,10,2),expand=expansion(mult=c(0,.03)))+
 labs(title=titles[a],subtitle=if(a=='A') 'ONSET = ego slowdown time' else 'ONSET = ego lane-crossing time',x=NULL,y='Onset time (s)')
}
p<-(panel('A')|panel('B'))/(panel('C')|panel('D'))+
 plot_annotation(caption=paste(
 'Dots: seed medians; diamonds and labels: median across seeds; bars: 95% seed-bootstrap CI. 36 scenes per seed.',
 'A/B/C: 20 seeds per preference. D: FD 19 / BAL 19 / SP 20 eligible seeds; 2 of 60 policies failed selection.',
 sep='\n'),
 theme=theme(plot.caption=element_text(size=9,hjust=0,colour=muted,lineheight=1.2),plot.margin=margin(10,18,12,18)))
for(ext in c('png','pdf','svg')) {
 dev<-switch(ext,png=ragg::agg_png,pdf=grDevices::cairo_pdf,svg=svglite::svglite)
 ggsave(file.path(out,paste0('ABCD_onset_time.',ext)),p,device=dev,width=11.2,height=7.6,units='in',dpi=300,bg='white')
}
ggsave(file.path(out,'ABCD_onset_time_preview.png'),p,device=ragg::agg_png,width=11.2,height=7.6,units='in',dpi=150,bg='white')
writeLines(c('# Appendix onset-time figure',
 'Four-arm 2x2 descriptive figure, with the same colors and summary definitions as the A+D main figure.',
 'A uses valid sustained-deceleration onset. B/C use the explicit first_actual_lane_change_seconds field collected into episode_outcomes response_latency_seconds; D uses the same physical-time field.',
 'B/C/D time is the first post-step lane-index change, not action onset or completed centering. No cross-arm timing hypothesis is tested.',
 'A/B/C each have 20 selected seeds per preference; D FD19/BAL19/SP20. D failures remain in the status table. 238 displayed policies, 36 scenes each; all displayed main-scene episodes have observed responses.',
 'First take median across scenes within seed, then median across seeds. t=0 observations retained. Missing events would not be assigned zero or the horizon.',
 'A/D bootstrap intervals reused exactly from the existing figure. B/C percentile intervals use 5000 seed bootstrap draws, R seed 20261027. These are not IQRs.',
 'All four panels share a 0–10 s y-axis with identical 2 s ticks.',
 'These are selected-policy development results; controls informed checkpoint selection and the initial reward-weight selection reused some seeds.',
 'All drawing and exports use R. Editable PDF/SVG, 300-dpi PNG and 150-dpi preview accompany episode/seed/summary data.'
),file.path(out,'figure_notes.md'))
print(summary)
