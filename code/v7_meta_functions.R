v7_contains_zero <- function(lower, upper) is.finite(lower) & is.finite(upper) & lower <= 0 & upper >= 0

v7_psd_gate <- function(V, tolerance = 1e-8) {
  V <- (as.matrix(V) + t(as.matrix(V))) / 2
  values <- eigen(V, symmetric = TRUE, only.values = TRUE)$values
  list(finite = all(is.finite(V)), minimum_eigen = min(values),
       pass = all(is.finite(V)) && min(values) >= -tolerance * max(1, max(abs(values))))
}

v7_heterogeneity <- function(fit) {
  s <- summary(fit); q <- s$qstat
  data.frame(Q = if (is.null(q)) NA_real_ else as.numeric(q$Q),
             Q_df = if (is.null(q)) NA_integer_ else as.integer(q$df),
             Q_pvalue = if (is.null(q)) NA_real_ else as.numeric(q$pvalue),
             I_squared_percent = if (is.null(s$i2stat)) NA_real_ else as.numeric(s$i2stat))
}

v7_fixed_prediction <- function(fit, predictor_row, outcomes) {
  D <- kronecker(matrix(as.numeric(predictor_row), nrow = 1), diag(outcomes))
  list(estimate = as.numeric(D %*% as.numeric(coef(fit))),
       covariance = D %*% as.matrix(vcov(fit)) %*% t(D), design = D)
}

v7_scalar_meta <- function(data, feature) {
  stopifnot(all(c("system", "partition", "estimate", "SE") %in% names(data)), nrow(data) == 10L)
  data$partition <- factor(data$partition, levels = c("TRAIN", "EVAL"))
  if (any(!is.finite(data$estimate)) || any(!is.finite(data$SE)) || any(data$SE <= 0)) stop("Invalid scalar meta input")
  moderator <- mixmeta::mixmeta(estimate ~ partition, S = SE^2, data = data, method = "reml")
  pooled <- mixmeta::mixmeta(estimate ~ 1, S = SE^2, data = data, method = "reml")
  if (!isTRUE(moderator$converged) || !isTRUE(pooled$converged)) stop("Scalar REML convergence failure")
  make_rows <- function(fit, labels, X, model) {
    tau2 <- as.numeric(fit$Psi[1, 1]); tau <- sqrt(max(0, tau2)); h <- v7_heterogeneity(fit)
    do.call(rbind, lapply(seq_along(labels), function(i) {
      prediction <- v7_fixed_prediction(fit, X[i, ], 1L)
      estimate <- prediction$estimate[1]; se <- sqrt(prediction$covariance[1, 1])
      ci <- estimate + c(-1, 1) * qnorm(.975) * se
      pi <- estimate + c(-1, 1) * qnorm(.975) * sqrt(se^2 + tau2)
      contains <- v7_contains_zero(pi[1], pi[2])
      data.frame(feature = feature, estimand = labels[i], model = model,
                 estimate = estimate, SE = se, CI_lower = ci[1], CI_upper = ci[2],
                 tau = tau, tau_squared = tau2, I_squared_percent = h$I_squared_percent,
                 Q = h$Q, Q_df = h$Q_df, Q_pvalue = h$Q_pvalue,
                 prediction_lower = pi[1], prediction_upper = pi[2], prediction_contains_zero = contains,
                 systems = nrow(data), converged = fit$converged,
                 fixed_covariance_pass = v7_psd_gate(vcov(fit))$pass,
                 between_covariance_pass = v7_psd_gate(fit$Psi)$pass,
                 partition_moderator_retained = model == "partition_adjusted_REML",
                 generality_interpretation = if (contains) "prediction interval includes zero; do not claim cross-system generality" else "prediction interval excludes zero")
    }))
  }
  rbind(make_rows(pooled, "POOLED", matrix(1, 1, 1), "intercept_only_REML"),
        make_rows(moderator, c("TRAIN", "EVAL"), rbind(c(1, 0), c(1, 1)), "partition_adjusted_REML"))
}

v7_multivariate_meta <- function(estimates, covariances, systems, partitions, feature, times) {
  coordinates <- c("constant", "ns1", "ns2", "ns3", "ns4"); estimates <- as.matrix(estimates)
  if (!identical(dim(estimates), c(10L, 5L)) || length(covariances) != 10L) stop("Multivariate meta dimension failure")
  for (V in covariances) if (!identical(dim(as.matrix(V)), c(5L, 5L)) || !v7_psd_gate(V)$pass) stop("Invalid within-system covariance")
  dat <- data.frame(system=systems,partition=factor(partitions,levels=c("TRAIN","EVAL")),estimates); names(dat)[3:7] <- coordinates
  fit <- mixmeta::mixmeta(cbind(constant,ns1,ns2,ns3,ns4)~partition,S=covariances,data=dat,method="reml",bscov="diag")
  if (!isTRUE(fit$converged)) stop("Multivariate specificity REML convergence failure")
  psi_gate <- v7_psd_gate(fit$Psi); fixed_gate <- v7_psd_gate(vcov(fit)); if(!psi_gate$pass||!fixed_gate$pass) stop("Multivariate meta covariance gate failure")
  basis <- cbind(constant=1,splines::ns(times,df=4)); colnames(basis)<-coordinates
  trajectory <- do.call(rbind,lapply(list(TRAIN=c(1,0),EVAL=c(1,1)),function(x){
    p<-v7_fixed_prediction(fit,x,5L); mean<-as.numeric(basis%*%p$estimate); mean_cov<-basis%*%p$covariance%*%t(basis); between_cov<-basis%*%fit$Psi%*%t(basis)
    se<-sqrt(pmax(0,diag(mean_cov))); between_var<-pmax(0,diag(between_cov)); pred_se<-sqrt(se^2+between_var)
    pi_low<-mean-qnorm(.975)*pred_se; pi_high<-mean+qnorm(.975)*pred_se; contains<-v7_contains_zero(pi_low,pi_high)
    data.frame(feature=feature,partition=if(identical(x,c(1,0)))"TRAIN" else "EVAL",relative_time_ms=times,
      projected_mean=mean,mean_SE=se,mean_CI_lower=mean-qnorm(.975)*se,mean_CI_upper=mean+qnorm(.975)*se,
      projected_between_system_variance=between_var,projected_between_system_tau=sqrt(between_var),
      prediction_lower=pi_low,prediction_upper=pi_high,prediction_contains_zero=contains,
      generality_interpretation=ifelse(contains,"prediction interval includes zero; do not claim cross-system generality","prediction interval excludes zero"))
  }))
  h<-v7_heterogeneity(fit); heterogeneity<-data.frame(feature=feature,component=c("overall",coordinates)[seq_len(nrow(h))],h)
  gates<-data.frame(feature=feature,systems=length(systems),outcomes=5L,method="multivariate REML",moderator="TRAIN/EVAL partition",
    between_system_covariance="diagonal (official V6 convention)",converged=fit$converged,logLik=as.numeric(logLik(fit)),Q=h$Q,Q_df=h$Q_df,Q_pvalue=h$Q_pvalue,I_squared_percent=h$I_squared_percent,
    fixed_covariance_pass=fixed_gate$pass,between_covariance_pass=psi_gate$pass,Psi_min_eigen=psi_gate$minimum_eigen)
  list(fit=fit,gates=gates[1,,drop=FALSE],heterogeneity=heterogeneity,trajectory=trajectory,basis=basis)
}
