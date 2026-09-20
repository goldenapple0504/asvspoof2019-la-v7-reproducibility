#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L || args[2]!="--execute-full") stop("usage: run_v7_specificity_meta_analysis.R OUTPUT_ROOT --execute-full")
root<-normalizePath(args[1],winslash="/",mustWork=TRUE); libs<-c(Sys.getenv("V7_R_LIB",unset=""),file.path(dirname(root),"rebuild_v6_trajectory","r_libs")); libs<-libs[nzchar(libs)&dir.exists(libs)]; if(length(libs)) .libPaths(c(libs,.libPaths()))
marker<-file.path(root,"FULL_V7_RUN_AUTHORIZED.txt"); if(!file.exists(marker)||trimws(readLines(marker,warn=FALSE)[1])!="FULL_V7_RUN_AUTHORIZED") stop("Full-run authorization absent")
suppressPackageStartupMessages(library(mixmeta)); source(file.path(root,"scripts","v7_meta_functions.R"))
systems<-c("A01","A02","A03","A04","A07","A08","A09","A10","A11","A12"); partitions<-ifelse(systems%in%c("A01","A02","A03","A04"),"TRAIN","EVAL")
features<-c("energy","centroid","tilt","flux","flatness","cpps"); coordinates<-c("constant","ns1","ns2","ns3","ns4")
contrast_dir<-file.path(root,"production","contrasts"); inference_dir<-file.path(root,"production","inference"); output_dir<-file.path(root,"production","meta_analysis"); dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
atomic_tsv<-function(x,path){tmp<-paste0(path,".tmp");write.table(x,tmp,sep="\t",row.names=FALSE,quote=FALSE,na="");if(!file.rename(tmp,path))stop("Atomic rename failed")}
average<-read.delim(file.path(inference_dir,"average_specificity.tsv"),stringsAsFactors=FALSE); model_rows<-list(); heterogeneity_rows<-list(); covariance_rows<-list(); trajectory_rows<-list(); scalar_rows<-list()
for(feature in features){
  estimates<-matrix(NA_real_,10L,5L,dimnames=list(systems,coordinates)); covariances<-vector("list",10L)
  for(i in seq_along(systems)){
    id<-paste(systems[i],feature,sep="_"); b<-read.delim(file.path(contrast_dir,paste0(id,"_specificity_coefficients.tsv")),stringsAsFactors=FALSE)
    V<-as.matrix(read.delim(file.path(contrast_dir,paste0(id,"_specificity_coefficient_covariance.tsv")),header=FALSE,check.names=FALSE))
    if(!identical(as.character(b$coordinate),coordinates)||any(dim(V)!=c(5L,5L)))stop("Coefficient schema failure: ",id)
    estimates[i,]<-b$estimate; covariances[[i]]<-V
  }
  times<-if(feature=="flux")seq(-15,20,5) else seq(-20,20,5); multi<-v7_multivariate_meta(estimates,covariances,systems,partitions,feature,times)
  model_rows[[feature]]<-multi$gates; heterogeneity_rows[[feature]]<-multi$heterogeneity; trajectory_rows[[feature]]<-multi$trajectory
  covariance_rows[[feature]]<-data.frame(feature=feature,row=rep(coordinates,each=5L),column=rep(coordinates,times=5L),value=as.vector(multi$fit$Psi))
  scalar<-average[average$feature==feature,]; if(nrow(scalar)!=10L)stop("Expected ten scalar estimates: ",feature); scalar$partition<-partitions[match(scalar$system,systems)]
  scalar_rows[[feature]]<-v7_scalar_meta(scalar,feature); saveRDS(list(multivariate=multi$fit),file.path(output_dir,paste0(feature,"_specificity_meta_models.rds")),compress="xz")
}
atomic_tsv(do.call(rbind,model_rows),file.path(output_dir,"multivariate_REML_model_gates.tsv")); atomic_tsv(do.call(rbind,covariance_rows),file.path(output_dir,"between_system_Psi.tsv"))
atomic_tsv(do.call(rbind,heterogeneity_rows),file.path(output_dir,"multivariate_heterogeneity.tsv"))
atomic_tsv(do.call(rbind,trajectory_rows),file.path(output_dir,"projected_specificity_trajectories_and_prediction_intervals.tsv")); atomic_tsv(do.call(rbind,scalar_rows),file.path(output_dir,"average_specificity_REML.tsv"))
