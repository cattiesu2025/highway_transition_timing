#!/usr/bin/env Rscript
# Figure contract: selected-weight sensitivity, not independent held-out inference.
# Figure 1: quantitative grid of seed onset gaps and exposure-paired contrasts.
# Figure 2: quantitative grid of conditional onset times and occurrence counts.
# R-only, same agent colors/three-arm ordering as reference; 20 x 9 inch exports.
# Seed is replication unit; 2000 seed bootstrap draws; preserve missingness/censoring.
.libPaths(c('tmp/r-lib',.libPaths()))
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork)})
stopifnot(requireNamespace('ragg',quietly=TRUE),requireNamespace('svglite',quietly=TRUE))
base <- 'outputs/reward_selected_validation/analysis'
out <- 'outputs/reward_selected_validation/figures'
dir.create(file.path(out,'source_data'),recursive=TRUE,showWarnings=FALSE)
dir.create('tmp/selected_validation_preview',recursive=TRUE,showWarnings=FALSE)
read <- function(name) read.csv(file.path(base,paste0(name,'.csv')),stringsAsFactors=FALSE)
write <- function(d,name) write.csv(d,file.path(out,'source_data',paste0(name,'.csv')),row.names=FALSE)
status <- read('run_status'); cfstatus <- read('counterfactual_status')
stopifnot(nrow(status)==180,all(status$status=='complete'),nrow(cfstatus)==180,all(cfstatus$status=='complete'))
agents <- c('FD','BAL','SP'); arms <- c('A','B','C')
arm_names <- c(A='Single lane',B='Two lanes\nempty target lane',C='Two lanes\noccupied target lane')
colors <- c(FD='#397F96',BAL='#7B65A8',SP='#CA7B22'); ink <- '#283743'
set.seed(12345)
boot <- function(x) {
 x <- x[is.finite(x)]
 if(!length(x)) return(data.frame(mid=NA_real_,lo=NA_real_,hi=NA_real_,n_seed=0))
 b <- replicate(2000,median(sample(x,length(x),replace=TRUE)))
 data.frame(mid=median(x),lo=unname(quantile(b,.025)),hi=unname(quantile(b,.975)),n_seed=length(x))
}
theme_set(theme_classic(base_size=16,base_family='Arial') + theme(
 text=element_text(colour=ink),axis.text=element_text(colour=ink),
 panel.grid.major.y=element_line(colour='#E9EDF0',linewidth=.35),
 strip.background=element_blank(),strip.text=element_text(face='bold',size=17),
 plot.title=element_text(face='bold',size=19),plot.margin=margin(12,18,12,18),
 legend.position='top',legend.title=element_blank()))
save <- function(p,name) {
 for(ext in c('png','pdf','svg')) {
  dev <- switch(ext,png=ragg::agg_png,pdf=grDevices::cairo_pdf,svg=svglite::svglite)
  ggsave(file.path(out,paste0(name,'.',ext)),p,device=dev,width=20,height=9,units='in',dpi=300,bg='white',limitsize=FALSE)
 }
 ggsave(file.path('tmp/selected_validation_preview',paste0(name,'.png')),p,device=ragg::agg_png,width=20,height=9,units='in',dpi=85,bg='white')
}
states <- read('onset_states') %>% mutate(agent=factor(agent,agents),arm=factor(arm,arms),
 observed=tolower(onset_observed)=='true',left=tolower(left_censored)=='true')
stopifnot(nrow(states)==180*36,!anyDuplicated(states[c('arm','agent','seed','exposure_id')]))
coverage <- states %>% group_by(arm,agent,seed) %>% summarise(n_total=n(),n_observed=sum(observed),
 n_left=sum(left),n_finite_uncensored=sum(observed & !left & is.finite(onset_gap_m)),.groups='drop')
write(coverage,'onset_coverage_by_seed')
valid <- states %>% filter(observed,!left,is.finite(onset_gap_m))
seed_gaps <- valid %>% group_by(arm,agent,seed) %>% summarise(gap=median(onset_gap_m),n_exposures=n(),.groups='drop')
centers <- seed_gaps %>% group_by(arm,agent) %>% summarise(mid=median(gap),n_seed=n(),.groups='drop')
write(seed_gaps,'onset_gap_by_seed');write(centers,'onset_gap_centers')
pairs <- list(c('FD','BAL'),c('FD','SP'),c('BAL','SP'))
pair_seed <- bind_rows(lapply(pairs,function(pair) {
 keys <- c('arm','seed','exposure_id','exposure_seed','rollout_id','rollout_seed')
 a <- valid %>% filter(agent==pair[1]) %>% select(all_of(keys),a=onset_gap_m)
 b <- valid %>% filter(agent==pair[2]) %>% select(all_of(keys),b=onset_gap_m)
 inner_join(a,b,by=keys) %>% mutate(delta=a-b) %>% group_by(arm,seed) %>%
 summarise(delta=median(delta),n_pairs=n(),.groups='drop') %>% mutate(contrast=paste(pair,collapse=' - '))
}))
contrast_order <- c('FD - BAL','FD - SP','BAL - SP')
contrasts <- pair_seed %>% group_by(arm,contrast) %>% group_modify(~ {
 d <- boot(.x$delta); pos <- sum(.x$delta>1e-9); neg <- sum(.x$delta< -1e-9)
 d$positive <- pos;d$negative <- neg;d$ties <- sum(abs(.x$delta)<=1e-9)
 d$p <- if(pos+neg) binom.test(pos,pos+neg,p=.5)$p.value else 1
 d
}) %>% ungroup() %>% mutate(contrast=factor(contrast,rev(contrast_order)),
 sig=lo>0 | hi<0,p_label=ifelse(p<.0001,'p < 0.0001',paste0('p = ',formatC(p,format='g',digits=2))),
 label=paste0(positive,'/',negative,'   ',p_label))
write(pair_seed,'paired_onset_gap_by_seed');write(contrasts,'paired_onset_gap_summary')
p1 <- ggplot(seed_gaps,aes(agent,gap,colour=agent)) +
 geom_point(position=position_jitter(width=.12,height=0,seed=77),alpha=.48,size=2.4)+
 geom_errorbar(data=centers,aes(ymin=mid,ymax=mid,y=mid),width=.62,linewidth=1.7)+
 geom_text(data=centers,aes(y=mid,label=sprintf('%.0f m',mid)),vjust=-1,size=4.3,fontface='bold')+
 facet_grid(~arm,labeller=as_labeller(arm_names)) + scale_colour_manual(values=colors)+
 scale_y_continuous(expand=expansion(mult=c(.08,.16)))+
 labs(title='a   Gap at onset: one point per seed',x=NULL,y='Gap to the front vehicle at onset (m)')+
 theme(legend.position='none',panel.spacing.x=grid::unit(9,'mm'))
# Right-side labels live in their own text column, preserving the contrast axis scale.
forest <- ggplot(contrasts,aes(mid,contrast))+
 geom_vline(xintercept=0,colour=ink,linewidth=.6)+
 geom_errorbar(aes(xmin=lo,xmax=hi,colour=sig),orientation='y',width=0,linewidth=1.25)+
 geom_point(aes(colour=sig),size=3.1)+
 facet_grid(arm~.,switch='y',labeller=as_labeller(c(A='Single
lane',B='Two lanes
empty target',C='Two lanes
occupied target')))+
 scale_colour_manual(values=c('FALSE'='#B5BEC5','TRUE'='#217875'),guide='none')+
 labs(title='b   Paired contrasts: bootstrap 95% CI',x='Paired onset-gap difference (m)',y=NULL)+
 theme(strip.placement='outside',strip.text.y.left=element_text(angle=0,size=11,face='bold'),panel.grid.major.x=element_line(colour='#E9EDF0',linewidth=.35),panel.grid.major.y=element_blank())
textcol <- ggplot(contrasts,aes(0,contrast,label=label)) + geom_text(hjust=0,size=3.8,colour=ink)+
 facet_grid(arm~.,labeller=as_labeller(c(A='Single lane',B='Two lanes: empty target',C='Two lanes: occupied target')))+
 coord_cartesian(xlim=c(0,1),clip='off')+theme_void(base_family='Arial')+
 labs(title='+/- seeds     Sign test')+theme(strip.text=element_blank(),plot.title=element_text(size=14),plot.margin=margin(12,12,12,0))
right <- forest+textcol+plot_layout(widths=c(1,.52))
fig1 <- p1+right+plot_layout(widths=c(1,1.12))+
 plot_annotation(title='Selected-weight validation: where the three agents act',
 subtitle='FD: F1 (0.28, 1.12)   |   BAL: F4 (0.56, 0.84)   |   SP: F7 (0.98, 0.42)',
 caption=paste('20 training seeds per agent/arm; 36 matched exposures per seed. Conditional gaps exclude unobserved and t=0 onsets.',
 'Contrasts: within-exposure differences, then median within seed; 2,000 seed bootstrap draws. Positive = first agent acts further back.',
 'Sign tests are two-sided, unadjusted; ties excluded. First 5 seeds reused from weight exploration; development sensitivity analysis.',sep='\n'),
 theme=theme(plot.title=element_text(size=25,face='bold'),plot.subtitle=element_text(size=16),plot.caption=element_text(size=12,hjust=0),plot.margin=margin(18,22,18,22)))
save(fig1,'01_onset_gap_validation')

cf <- read('counterfactual_summary') %>% filter(arm!='A' | record_type=='agent_outcome') %>%
 mutate(arm=factor(arm,arms),agent=factor(agent,agents),
 n_total=ifelse(arm=='A',n,n_episodes),
 n_onset=ifelse(arm=='A',valid_onset_count,actual_lane_change_episode_count),
 seconds=ifelse(arm=='A',median_response_latency_seconds,median_first_actual_lane_change_t/5))
stopifnot(nrow(cf)==720,all(cf$n_total==36),!anyDuplicated(cf[c('arm','agent','seed','counterfactual_variant')]),
 all(cf$n_onset>=0 & cf$n_onset<=36),all(is.finite(cf$seconds[cf$n_onset>0])))
cf$seconds[cf$n_onset==0] <- NA_real_
write(cf,'controls_by_seed')
cf_sum <- cf %>% group_by(arm,agent,counterfactual_variant) %>% group_modify(~ {
 d <- boot(.x$seconds);d$n_onset <- sum(.x$n_onset);d$n_total <- sum(.x$n_total);d
}) %>% ungroup()
write(cf_sum,'controls_summary')
variants <- c('matched-speed-front','far-front','original')
cf <- cf %>% mutate(y=match(counterfactual_variant,variants)+(2-as.numeric(agent))*.19)
cf_sum <- cf_sum %>% mutate(y=match(counterfactual_variant,variants)+(2-as.numeric(agent))*.19)
panels <- lapply(arms,function(a) {
 raw <- cf %>% filter(arm==a,counterfactual_variant!='no-front')
 summary <- cf_sum %>% filter(arm==a,counterfactual_variant!='no-front')
 nf <- cf %>% filter(arm==a,counterfactual_variant=='no-front')
 xmax <- max(raw$seconds,summary$hi,na.rm=TRUE)*1.17
 count_x <- xmax*1.03
 summary$counts <- sprintf('%d/%d; %d seeds',summary$n_onset,summary$n_total,summary$n_seed)
 ggplot(raw,aes(seconds,y,colour=agent,shape=agent))+
 geom_point(position=position_jitter(width=0,height=.035,seed=88),alpha=.3,size=1.8,na.rm=TRUE)+
 geom_errorbar(data=summary,aes(x=mid,xmin=lo,xmax=hi),orientation='y',width=0,linewidth=1.1,na.rm=TRUE)+
 geom_point(data=summary,aes(x=mid),size=3.4,na.rm=TRUE)+
 geom_text(data=summary,aes(x=mid,label=sprintf('%.1f',mid)),vjust=-.85,hjust=-.1,size=3.8,show.legend=FALSE,na.rm=TRUE)+
 geom_text(data=summary,aes(x=count_x,label=counts),hjust=0,size=3.5,colour=ink,show.legend=FALSE)+
 annotate('text',x=count_x,y=3.58,label='Onsets / episodes; seeds with onset',hjust=0,size=3.0,colour=ink)+
 annotate('text',x=0,y=.43,label=sprintf('No front: %d/%d onsets (all agents, 20 seeds each)',sum(nf$n_onset),sum(nf$n_total)),hjust=0,size=3.7,colour='#B9493B')+
 scale_colour_manual(values=colors)+scale_shape_manual(values=c(FD=16,BAL=15,SP=18))+
 scale_y_continuous(breaks=1:3,labels=c('Matched-speed front','Far front','Real slow front'),limits=c(.25,3.75))+
 scale_x_continuous(breaks=pretty(c(0,xmax),n=4),expand=c(0,0))+
 coord_cartesian(xlim=c(0,xmax*1.68),clip='off')+
 labs(title=paste0(tolower(a),'   ',c(A='Single lane',B='Two lanes: empty target',C='Two lanes: occupied target')[a]),
 x='Conditional median onset time (s)',y=NULL)+theme(panel.grid.major.y=element_blank(),panel.grid.major.x=element_line(colour='#EEF1F3',linewidth=.35),
 plot.title=element_text(size=17),axis.text.y=element_text(size=12),plot.margin=margin(12,12,12,12))
})
fig2 <- wrap_plots(panels,nrow=1,guides='collect')+
 plot_annotation(title='Selected-weight validation: responses to hazard controls',
 subtitle='Points: seed medians; large symbols: across-seed median; bars: bootstrap 95% CI. Panel time scales differ.',
 caption=paste('20 training seeds per agent/arm; 36 exposures per condition/seed. Timing is conditional on observed onset; missing onsets are not set to 120 s.',
 'Counts show observed episodes / all episodes and contributing seeds. No-front counts include all three agents.',
 'Standalone rollouts of fixed selected policies; 5/20 seeds reused. Development sensitivity analysis; checkpoint gates also used hazard controls.',sep='\n'),
 theme=theme(plot.title=element_text(size=25,face='bold'),plot.subtitle=element_text(size=16),plot.caption=element_text(size=12,hjust=0),plot.margin=margin(18,22,18,22))) & theme(legend.position='top')
save(fig2,'02_hazard_control_validation')
print(centers);print(contrasts);print(cf_sum)

writeLines(c(
 '# Selected-weight validation figure notes',
 'Complete dataset: 180 policies, 20 seeds per arm/agent, 36 exposures per condition.',
 'Weights: FD=F1 (0.28,1.12); BAL=F4 (0.56,0.84); SP=F7 (0.98,0.42).',
 'Figure 1 conditions gap estimates on finite observed onsets at t>0. Seed medians are shown.',
 'Paired contrasts join exposure ID, exposure seed and rollout identifiers within training seed, then take the median paired difference.',
 'All 9 contrasts contain 20 seed differences. Bootstrap: 2000 draws, seed=12345; sign tests two-sided, unadjusted, ties excluded.',
 'Figure 2 uses conditional seed medians; B/C physical lane-crossing time is recorded in steps and divided by 5 Hz.',
 'Missing onsets are never replaced by a time limit. Occurrence denominators and contributing seed counts are printed.',
 'No-front has 0/2160 onsets in each arm. In C matched-speed conditions, only 19/20 seeds contribute timing medians for each agent.',
 'These are development sensitivity results; five seeds were reused from weight exploration; checkpoint gates already used hazard controls.',
 'Original reference detail does not reproduce exactly: A BAL-SP and C FD-BAL contrasts now exclude zero in the unadjusted analysis.',
 'R-generated 300-dpi PNG and vector SVG/PDF; seed summaries and pairing source tables accompany figures.'
),file.path(out,'figure_notes.md'))
