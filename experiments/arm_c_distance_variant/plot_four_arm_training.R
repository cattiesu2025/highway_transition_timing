#!/usr/bin/env Rscript
# Figure contract: show training dynamics of the three frozen preferences in A/B/C/D.
# Quantitative 2x2 grid; all 20 training seeds included, independent of selection.
# R-only exports; reward-function-specific return is not a common performance score.
.libPaths(c('tmp/r-lib',.libPaths()))
suppressPackageStartupMessages({library(ggplot2);library(dplyr)})
stopifnot(requireNamespace('ragg',quietly=TRUE),requireNamespace('svglite',quietly=TRUE))
out<-'outputs/four_arm_training_curves';src<-file.path(out,'source_data')
d<-bind_rows(read.csv(file.path(src,'ABC_training_progress.csv')),read.csv(file.path(src,'D_training_progress.csv')))
stopifnot(all(is.finite(d$timesteps)),all(is.finite(d$episode_reward)))
variants<-c(FD='F1',BAL='F4',SP='F7');speed<-c(FD=.28,BAL=.56,SP=.98);spacing<-c(FD=1.12,BAL=.84,SP=.42)
stopifnot(all(d$variant==variants[d$agent]),all(abs(d$speed_weight-speed[d$agent])<1e-9),all(abs(d$front_distance_weight-spacing[d$agent])<1e-9))
coverage<-d %>% group_by(arm,agent,seed) %>% summarise(first_step=min(timesteps),last_step=max(timesteps),n_records=n(),.groups='drop')
counts<-coverage %>% count(arm,agent)
stopifnot(nrow(coverage)==240,nrow(counts)==12,all(counts$n==20),!anyDuplicated(d[c('arm','agent','seed','timesteps')]))
# Interpolate only inside recorded support. Do not invent a final point at 100k.
grid<-seq(1000,99000,1000)
stopifnot(all(coverage$first_step<=min(grid)),all(coverage$last_step>=max(grid)))
aligned<-d %>% group_by(arm,agent,seed) %>% group_modify(~data.frame(timesteps=grid,
 episode_reward=approx(.x$timesteps,.x$episode_reward,xout=grid,rule=1)$y)) %>% ungroup()
stopifnot(nrow(aligned)==240*length(grid),all(is.finite(aligned$episode_reward)))
s<-aligned %>% group_by(arm,agent,timesteps) %>% summarise(mid=median(episode_reward),
 lo=quantile(episode_reward,.25),hi=quantile(episode_reward,.75),n_seed=n(),.groups='drop')
stopifnot(all(s$n_seed==20))
write.csv(coverage,file.path(src,'log_coverage.csv'),row.names=FALSE)
write.csv(aligned,file.path(src,'aligned_seed_curves.csv'),row.names=FALSE)
write.csv(s,file.path(src,'median_iqr_curves.csv'),row.names=FALSE)
write.csv(s %>% filter(timesteps==99000),file.path(src,'final_logged_grid_summary.csv'),row.names=FALSE)
s$arm<-factor(s$arm,c('A','B','C','D'));s$agent<-factor(s$agent,c('FD','BAL','SP'))
cols<-c(FD='#397F96',BAL='#7B65A8',SP='#CA7B22');ink<-'#263543'
labels<-c(A='Arm A · Single lane',B='Arm B · Empty target lane',C='Arm C · Occupied target lane',D='Arm D · Yellow car farther ahead')
aligned$arm<-factor(aligned$arm,c('A','B','C','D'));aligned$agent<-factor(aligned$agent,c('FD','BAL','SP'))
limits<-c(min(0,floor(min(aligned$episode_reward)/25)*25),ceiling(max(aligned$episode_reward)/25)*25)
p<-ggplot(s,aes(timesteps/1000,mid,colour=agent,fill=agent))+
 geom_ribbon(aes(ymin=lo,ymax=hi),alpha=.16,colour=NA)+
 geom_line(data=aligned,aes(y=episode_reward,group=interaction(agent,seed)),linewidth=.27,alpha=.24,show.legend=FALSE)+
 geom_line(linewidth=1.0)+
 facet_wrap(~arm,ncol=2,labeller=as_labeller(labels))+
 scale_colour_manual(values=cols,labels=c(FD='FD',BAL='BAL',SP='SP'))+
 scale_fill_manual(values=cols,guide='none')+
 scale_x_continuous(limits=c(0,100),breaks=c(0,25,50,75,100),expand=expansion(mult=c(0,.015)))+
 scale_y_continuous(limits=limits,breaks=seq(limits[1],limits[2],25),expand=expansion(mult=c(0,.02)))+
 labs(title='Training curves across four arms',subtitle='Selected weights · 20 training seeds per preference in every arm',
 x='Training steps (× 1,000)',y='Mean episode reward',colour=NULL,linetype=NULL,
 caption=paste('Thin lines: individual seeds; thick lines: median across 20 seeds; shading: IQR (25–75%).',
 'Weights (front distance, speed): FD (1.12, 0.28) · BAL (0.84, 0.56) · SP (0.42, 0.98).',sep='\n'))+
 theme_classic(base_size=11,base_family='Arial')+
 theme(text=element_text(colour=ink),axis.text=element_text(colour=ink),
 axis.line=element_line(colour='#9AA6AE',linewidth=.3),axis.ticks=element_line(colour='#9AA6AE',linewidth=.3),
 panel.grid.major.y=element_line(colour='#E9EEF1',linewidth=.3),
 strip.background=element_blank(),strip.text=element_text(face='bold',size=12,hjust=0,margin=margin(b=10,t=9)),
 panel.spacing=grid::unit(10,'mm'),legend.position='top',legend.justification='left',legend.key.width=grid::unit(12,'mm'),
 plot.title=element_text(size=20,face='bold',margin=margin(b=7)),plot.subtitle=element_text(size=11,colour='#63717C',margin=margin(b=8)),
 plot.caption=element_text(size=9,colour='#63717C',hjust=0,lineheight=1.2,margin=margin(t=14)),
 plot.margin=margin(16,18,12,16))
for(ext in c('png','pdf','svg')) {
 dev<-switch(ext,png=ragg::agg_png,pdf=grDevices::cairo_pdf,svg=svglite::svglite)
 ggsave(file.path(out,paste0('ABCD_selected_weights_training.',ext)),p,device=dev,width=11.2,height=7.8,units='in',dpi=300,bg='white')
}
ggsave(file.path(out,'ABCD_selected_weights_training_preview.png'),p,device=ragg::agg_png,width=11.2,height=7.8,units='in',dpi=150,bg='white')
writeLines(c('# Training-curve figure notes',
 'All four arms use only selected weights F1/F4/F7 and 20 training seeds per preference: 240 runs total.',
 'A seeds 3100–3119; B 4100–4119; C/D 4200–4219. D FD/4209 and BAL/4205 remain included despite selection failure.',
 'Input is rollout/ep_rew_mean from the SB3 progress logger: a rolling episode-return mean (up to the most recent 100 episodes), not each raw episode.',
 'Each seed is linearly interpolated to 1k–99k training steps at 1k intervals, strictly within its actual recorded support. No extrapolation or extra smoothing.',
 'Thin lines show every seed on the aligned grid. Thick lines are pointwise medians across 20 seeds. Bands are pointwise 25th–75th percentiles across seeds, not confidence intervals.',
 'Training reached 100k for all runs, but the progress logger need not emit a record exactly at 100k; curves end at 99k to preserve a common observed support.',
 'Displayed weight order is (front distance, speed): FD (1.12, 0.28), BAL (0.84, 0.56), SP (0.42, 0.98). Internal F identifiers and source columns are unchanged.',
 'Shared axes and colors across the four arms. All median curves are solid; colors distinguish preferences. No selection of visually favorable seeds.',
 'Rewards differ by preference and geometry; the figure shows training dynamics, not a common driving-quality ranking or proof of convergence.',
 'Exports produced entirely by R; PNG 300 dpi, preview 150 dpi, editable SVG/PDF. Input hashes and all per-seed/summary data are in source_data/.'
),file.path(out,'figure_notes.md'))
print(counts);print(s %>% filter(timesteps==99000))
