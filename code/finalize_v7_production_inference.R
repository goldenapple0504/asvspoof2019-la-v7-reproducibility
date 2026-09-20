#!/usr/bin/env Rscript
args<-commandArgs(trailingOnly=TRUE); if(length(args)!=2L||args[2]!="--execute-full")stop("usage: finalize_v7_production_inference.R OUTPUT_ROOT --execute-full")
root<-normalizePath(args[1],winslash="/",mustWork=TRUE); marker<-file.path(root,"FULL_V7_RUN_AUTHORIZED.txt")
if(!file.exists(marker)||trimws(readLines(marker,warn=FALSE)[1])!="FULL_V7_RUN_AUTHORIZED")stop("Full-run authorization absent")
contrast_dir<-file.path(root,"production","contrasts"); output_dir<-file.path(root,"production","inference"); dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
atomic_tsv<-function(x,path){temporary<-paste0(path,".tmp");write.table(x,temporary,sep="\t",row.names=FALSE,quote=FALSE,na="");if(!file.rename(temporary,path))stop("Atomic rename failed: ",path)}
read_parts<-function(pattern,expected=60L){files<-sort(list.files(contrast_dir,pattern=pattern,full.names=TRUE));if(length(files)!=expected)stop("Expected ",expected," files for ",pattern,"; found ",length(files));if(!length(files))return(data.frame());do.call(rbind,lapply(files,read.delim,stringsAsFactors=FALSE,check.names=FALSE))}

acceptance_files<-sort(list.files(file.path(root,"production","diagnostics"),pattern="_acceptance\\.tsv$",full.names=TRUE));if(length(acceptance_files)!=60L)stop("Expected 60 model acceptance files")
acceptance<-do.call(rbind,lapply(acceptance_files,read.delim,stringsAsFactors=FALSE)); if(!all(acceptance$all_acceptance_gates_pass))stop("One or more production model acceptance gates failed")
authorized_status<-acceptance$CR2_status%in%c("VALID_FINITE_PSD_CR2","NOT_APPLICABLE_CROSSED_CLUSTERING")
valid_ok<-acceptance$CR2_status!="VALID_FINITE_PSD_CR2" | (acceptance$CR2_success & acceptance$CR2_covariance_finite & acceptance$CR2_estimates_produced & acceptance$CR2_match_clusters_independent_partitions)
na_ok<-acceptance$CR2_status!="NOT_APPLICABLE_CROSSED_CLUSTERING" | (acceptance$CR2_evidence_complete & acceptance$CR2_utterances_multiple_matches>0 & !acceptance$CR2_estimates_produced & !acceptance$CR2_match_clusters_independent_partitions)
if(!all(authorized_status&valid_ok&na_ok))stop("Unauthorized or incompletely evidenced CR2 outcome")

average<-read_parts("_average_inference_raw\\.tsv$");shape<-read_parts("_shape_inference_raw\\.tsv$");time<-read_parts("_time_inference_raw\\.tsv$")
if(anyDuplicated(average[c("system","feature")])||anyDuplicated(shape[c("system","feature")]))stop("Duplicate average/shape targets")
adjust_inference<-function(average,shape,time){
  average$p_BH_60<-p.adjust(average$p_value,method="BH");average$TOST_p_BH_60<-p.adjust(average$TOST_p,method="BH")
  average$average_classification<-ifelse(average$p_BH_60>=.05,"NOT_SIGNIFICANT",ifelse(average$estimate>0,"SIGNIFICANT_POSITIVE","SIGNIFICANT_NEGATIVE"))
  average$SESOI_classification<-ifelse(average$TOST_p_BH_60<.05,"EQUIVALENT",ifelse(average$CI95_low>.10,"MEANINGFUL_POSITIVE",ifelse(average$CI95_high < -0.10,"MEANINGFUL_NEGATIVE","INCONCLUSIVE")))
  shape$p_BH_60<-p.adjust(shape$p_value,method="BH");shape$shape_classification<-ifelse(shape$p_BH_60<.05,"SIGNIFICANT_SHAPE","NOT_SIGNIFICANT_SHAPE")
  time$p_Bonferroni_within_trajectory<-ave(time$p_value,interaction(time$system,time$feature,drop=TRUE),FUN=function(p)p.adjust(p,method="bonferroni"));time$p_BH_global<-p.adjust(time$p_value,method="BH")
  time$TOST_p_Bonferroni_within_trajectory<-ave(time$TOST_p,interaction(time$system,time$feature,drop=TRUE),FUN=function(p)p.adjust(p,method="bonferroni"));time$TOST_p_BH_global<-p.adjust(time$TOST_p,method="BH")
  time$predefined_followup_significant<-time$p_Bonferroni_within_trajectory<.05&time$p_BH_global<.05;time$SESOI_equivalent<-time$TOST_p_Bonferroni_within_trajectory<.05&time$TOST_p_BH_global<.05
  time$followup_classification<-ifelse(time$predefined_followup_significant,"SIGNIFICANT","NOT_SIGNIFICANT")
  list(average=average,shape=shape,time=time)
}
primary<-adjust_inference(average,shape,time);average<-primary$average;shape<-primary$shape;time<-primary$time
valid_count<-sum(acceptance$CR2_status=="VALID_FINITE_PSD_CR2")
average_cr2<-read_parts("_average_inference_CR2_raw\\.tsv$",valid_count);shape_cr2<-read_parts("_shape_inference_CR2_raw\\.tsv$",valid_count);time_cr2<-read_parts("_time_inference_CR2_raw\\.tsv$",valid_count)
if(!valid_count){average_cr2<-data.frame(system=character(),feature=character(),CR2_status=character());shape_cr2<-average_cr2;time_cr2<-data.frame(system=character(),feature=character(),relative_time_ms=numeric(),CR2_status=character())}
comparison<-average[c("system","feature","average_classification","SESOI_classification")];names(comparison)[3:4]<-paste0(names(comparison)[3:4],"_model_based")
shape_primary<-shape[c("system","feature","shape_classification")];names(shape_primary)[3]<-"shape_classification_model_based";comparison<-merge(comparison,shape_primary,by=c("system","feature"))
status_columns<-acceptance[c("system","feature","CR2_status","CR2_evidence_complete","CR2_utterances_multiple_matches","CR2_proportion_utterances_multiple_matches","CR2_maximum_matches_per_utterance","CR2_match_clusters_independent_partitions","CR2_estimates_produced")]
comparison<-merge(comparison,status_columns,by=c("system","feature"),all.x=TRUE)
comparison$average_classification_CR2<-NA_character_;comparison$SESOI_classification_CR2<-NA_character_;comparison$shape_classification_CR2<-NA_character_;comparison$time_resolved_classification_changes<-NA_integer_
if(valid_count){
  cr2<-adjust_inference(average_cr2,shape_cr2,time_cr2);average_cr2<-cr2$average;shape_cr2<-cr2$shape;time_cr2<-cr2$time
  cr2_class<-merge(average_cr2[c("system","feature","average_classification","SESOI_classification")],shape_cr2[c("system","feature","shape_classification")],by=c("system","feature"))
  time_compare<-merge(time[c("system","feature","relative_time_ms","followup_classification")],time_cr2[c("system","feature","relative_time_ms","followup_classification")],by=c("system","feature","relative_time_ms"),suffixes=c("_model_based","_CR2"))
  changes<-aggregate(I(followup_classification_model_based!=followup_classification_CR2)~system+feature,time_compare,sum);names(changes)[3]<-"time_resolved_classification_changes"
  for(i in seq_len(nrow(cr2_class))){j<-which(comparison$system==cr2_class$system[i]&comparison$feature==cr2_class$feature[i]);comparison[j,c("average_classification_CR2","SESOI_classification_CR2","shape_classification_CR2")]<-cr2_class[i,c("average_classification","SESOI_classification","shape_classification")]}
  comparison<-merge(comparison,changes,by=c("system","feature"),all.x=TRUE,suffixes=c("","_computed"));comparison$time_resolved_classification_changes<-ifelse(is.na(comparison$time_resolved_classification_changes_computed),comparison$time_resolved_classification_changes,comparison$time_resolved_classification_changes_computed);comparison$time_resolved_classification_changes_computed<-NULL
}
comparison$average_classification_changed<-ifelse(comparison$CR2_status=="VALID_FINITE_PSD_CR2",comparison$average_classification_model_based!=comparison$average_classification_CR2,NA)
comparison$shape_classification_changed<-ifelse(comparison$CR2_status=="VALID_FINITE_PSD_CR2",comparison$shape_classification_model_based!=comparison$shape_classification_CR2,NA)
comparison$SESOI_classification_changed<-ifelse(comparison$CR2_status=="VALID_FINITE_PSD_CR2",comparison$SESOI_classification_model_based!=comparison$SESOI_classification_CR2,NA);comparison$sensitivity_only<-TRUE
atomic_tsv(average,file.path(output_dir,"average_specificity.tsv"));atomic_tsv(shape,file.path(output_dir,"joint_shape_specificity.tsv"));atomic_tsv(time,file.path(output_dir,"time_resolved_specificity.tsv"))
atomic_tsv(average_cr2,file.path(output_dir,"average_specificity_CR2_sensitivity.tsv"));atomic_tsv(shape_cr2,file.path(output_dir,"joint_shape_specificity_CR2_sensitivity.tsv"));atomic_tsv(time_cr2,file.path(output_dir,"time_resolved_specificity_CR2_sensitivity.tsv"))
atomic_tsv(comparison,file.path(output_dir,"covariance_sensitivity_classification_comparison.tsv"));atomic_tsv(acceptance,file.path(output_dir,"model_acceptance_gates.tsv"))
