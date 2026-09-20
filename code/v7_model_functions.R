suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(splines))

V7_SESOI <- 0.10
V7_ALPHA <- 0.05
V7_RANDOM_PRIORITY <- c("utterance_id", "match_id")
V7_MANDATORY_RANDOM <- "token_occurrence_id"

v7_fixed_formula <- function() {
  feature_z ~ authenticity * region * splines::ns(relative_time_ms, df = 4) +
    duration_A_c + duration_B_c + exact_phone_pair
}

v7_random_formula <- function(components) {
  paste(vapply(components, function(x) paste0("(1|", x, ")"), character(1)), collapse = " + ")
}

v7_full_formula <- function(components) {
  as.formula(paste(deparse(v7_fixed_formula()), collapse = " ") |> paste("+", v7_random_formula(components)))
}

v7_prepare_data <- function(x) {
  required <- c("feature_z", "authenticity", "region", "relative_time_ms", "duration_A_c",
                "duration_B_c", "exact_phone_pair", "token_occurrence_id", "match_id", "utterance_id")
  missing <- setdiff(required, names(x))
  if (length(missing)) stop("Missing model columns: ", paste(missing, collapse = ", "))
  if (any(!is.finite(x$feature_z))) stop("Non-finite feature_z")
  x$authenticity <- factor(x$authenticity, levels = c("BF", "TTS"))
  x$region <- factor(x$region, levels = c("Boundary", "Interior_A", "Interior_B"))
  x$exact_phone_pair <- droplevels(factor(x$exact_phone_pair))
  x$token_occurrence_id <- droplevels(factor(x$token_occurrence_id))
  x$match_id <- droplevels(factor(x$match_id))
  x$utterance_id <- droplevels(factor(x$utterance_id))
  if (anyNA(x[, required])) stop("Missing model values after factor preparation")
  x
}

v7_fixed_rank_gate <- function(x) {
  X <- model.matrix(v7_fixed_formula(), data = x)
  rank <- qr(X, tol = 1e-10)$rank
  list(pass = rank == ncol(X), rank = rank, columns = ncol(X), condition = kappa(X), names = colnames(X))
}

v7_capture_fit <- function(form, x) {
  caught <- character()
  fit <- withCallingHandlers(
    lmer(form, data = x, REML = TRUE,
         control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000),
                               check.rankX = "stop.deficient", check.conv.singular = "ignore")),
    warning = function(w) { caught <<- c(caught, conditionMessage(w)); invokeRestart("muffleWarning") }
  )
  list(fit = fit, warnings = unique(caught))
}

v7_variances <- function(fit) {
  vc <- as.data.frame(VarCorr(fit))
  setNames(vc$vcov[is.na(vc$var2)], vc$grp[is.na(vc$var2)])
}

v7_fit_rank_safe <- function(data) {
  x <- v7_prepare_data(data)
  rank_gate <- v7_fixed_rank_gate(x)
  if (!rank_gate$pass) {
    stop(sprintf("RANK_DEFICIENT_FIXED_MATRIX: rank=%d columns=%d", rank_gate$rank, rank_gate$columns))
  }
  active <- c(V7_MANDATORY_RANDOM, "match_id", "utterance_id")
  attempts <- list()
  all_warnings <- character()
  repeat {
    form <- v7_full_formula(active)
    captured <- v7_capture_fit(form, x)
    fit <- captured$fit
    all_warnings <- unique(c(all_warnings, captured$warnings))
    convergence <- fit@optinfo$conv$lme4$messages
    if (length(convergence)) stop("MODEL_NONCONVERGENCE: ", paste(convergence, collapse = " | "))
    variances <- v7_variances(fit)
    residual_variance <- sigma(fit)^2
    threshold <- max(1e-8, 1e-6 * residual_variance)
    zero <- names(variances)[is.finite(variances) & variances <= threshold]
    singular <- isSingular(fit, tol = 1e-5)
    attempts[[length(attempts) + 1L]] <- data.frame(
      attempt = length(attempts) + 1L,
      random_structure = paste(active, collapse = " + "),
      singular = singular,
      zero_variance_components = paste(zero, collapse = ";"),
      variance_components = paste(paste(names(variances), format(variances, digits = 16), sep = "="), collapse = ";"),
      residual_variance = residual_variance,
      zero_threshold = threshold,
      stringsAsFactors = FALSE
    )
    if (!singular) break
    if (V7_MANDATORY_RANDOM %in% zero) {
      stop("SINGULAR_MANDATORY_TOKEN_COMPONENT: token_occurrence_id variance is at boundary")
    }
    removable <- V7_RANDOM_PRIORITY[V7_RANDOM_PRIORITY %in% zero & V7_RANDOM_PRIORITY %in% active]
    if (!length(removable)) {
      stop("UNRESOLVED_SINGULARITY: no prespecified removable zero-variance component")
    }
    active <- setdiff(active, removable[1])
  }
  list(fit = fit, data = x, fixed_rank = rank_gate, attempts = do.call(rbind, attempts),
       final_random_components = active, warnings = all_warnings, variances = v7_variances(fit))
}

v7_fixed_matrix <- function(fit, newdata) {
  fixed <- reformulas::nobars(formula(fit))
  terms_fixed <- delete.response(terms(fixed, data = model.frame(fit)))
  contrasts <- attr(model.matrix(fit), "contrasts")
  levels <- .getXlevels(terms_fixed, model.frame(fit))
  X <- model.matrix(terms_fixed, newdata, contrasts.arg = contrasts, xlev = levels)
  wanted <- names(fixef(fit))
  missing <- setdiff(wanted, colnames(X))
  if (length(missing)) stop("Contrast design missing fitted columns: ", paste(missing, collapse = ", "))
  X[, wanted, drop = FALSE]
}

v7_cell_weights <- function() {
  data.frame(
    authenticity = c("TTS", "BF", "TTS", "BF", "TTS", "BF"),
    region = c("Boundary", "Boundary", "Interior_A", "Interior_A", "Interior_B", "Interior_B"),
    weight = c(1, -1, -0.5, 0.5, -0.5, 0.5),
    stringsAsFactors = FALSE
  )
}

v7_contrast_matrices <- function(fit, times) {
  frame <- model.frame(fit)
  cells <- v7_cell_weights()
  nd <- do.call(rbind, lapply(times, function(time) data.frame(
    authenticity = factor(cells$authenticity, levels = levels(frame$authenticity)),
    region = factor(cells$region, levels = levels(frame$region)),
    relative_time_ms = rep(time, nrow(cells)),
    duration_A_c = 0, duration_B_c = 0,
    exact_phone_pair = factor(rep(levels(frame$exact_phone_pair)[1], nrow(cells)),
                              levels = levels(frame$exact_phone_pair))
  )))
  # Evaluate the spline on the complete official grid at once so its knots and
  # boundary knots are identical to the balanced complete-grid training data.
  X_all <- v7_fixed_matrix(fit, nd)
  L <- matrix(NA_real_, length(times), length(fixef(fit)),
              dimnames = list(as.character(times), names(fixef(fit))))
  for (i in seq_along(times)) {
    X <- X_all[((i - 1L) * nrow(cells) + 1L):(i * nrow(cells)), , drop = FALSE]
    L[i, ] <- as.vector(cells$weight %*% X)
  }
  average <- matrix(colMeans(L), nrow = 1L, dimnames = list("equal_grid_average", colnames(L)))
  centered <- sweep(L, 2L, average[1, ], "-")
  decomposition <- qr(t(centered), tol = 1e-9)
  if (decomposition$rank != 4L) stop("SHAPE_CONTRAST_RANK_NOT_FOUR: ", decomposition$rank)
  rows <- decomposition$pivot[seq_len(decomposition$rank)]
  shape <- centered[rows, , drop = FALSE]
  rownames(shape) <- paste0("shape_basis_", seq_len(nrow(shape)))
  basis <- cbind(constant = 1, splines::ns(times, df = 4))
  coordinates <- solve(crossprod(basis), crossprod(basis, L))
  rownames(coordinates) <- c("constant", paste0("ns", 1:4))
  list(time = L, average = average, shape = shape, curve_coordinates = coordinates,
       cell_weights = cells, basis = basis)
}

v7_wald_scalar <- function(L, beta, V, sesoi = V7_SESOI) {
  estimate <- as.numeric(L %*% beta)
  variance <- as.numeric(L %*% V %*% t(L))
  if (!is.finite(variance) || variance <= 0) stop("NONPOSITIVE_SCALAR_CONTRAST_VARIANCE")
  se <- sqrt(variance)
  z <- estimate / se
  p <- 2 * pnorm(-abs(z))
  p_lower <- 1 - pnorm((estimate + sesoi) / se)
  p_upper <- pnorm((estimate - sesoi) / se)
  tost_p <- max(p_lower, p_upper)
  data.frame(estimate = estimate, SE = se, z = z, p_value = p,
             CI95_low = estimate - qnorm(.975) * se, CI95_high = estimate + qnorm(.975) * se,
             SESOI_lower = -sesoi, SESOI_upper = sesoi,
             TOST_p_lower = p_lower, TOST_p_upper = p_upper, TOST_p = tost_p,
             TOST_equivalent_alpha_0_05 = tost_p < .05)
}

v7_joint_wald <- function(C, beta, V) {
  estimate <- as.vector(C %*% beta)
  covariance <- C %*% V %*% t(C)
  if (qr(covariance, tol = 1e-10)$rank != nrow(C)) stop("RANK_DEFICIENT_SHAPE_COVARIANCE")
  statistic <- as.numeric(t(estimate) %*% solve(covariance, estimate))
  data.frame(chisq = statistic, df = nrow(C), p_value = pchisq(statistic, nrow(C), lower.tail = FALSE))
}

v7_analyze_specificity <- function(fit_result, times, covariance_override = NULL) {
  fit <- fit_result$fit
  beta <- fixef(fit)
  V <- if (is.null(covariance_override)) vcov(fit) else covariance_override
  contrasts <- v7_contrast_matrices(fit, times)
  time <- do.call(rbind, lapply(seq_along(times), function(i) {
    cbind(relative_time_ms = times[i], v7_wald_scalar(contrasts$time[i, , drop = FALSE], beta, V))
  }))
  average <- v7_wald_scalar(contrasts$average, beta, V)
  shape <- v7_joint_wald(contrasts$shape, beta, V)
  curve_beta <- as.vector(contrasts$curve_coordinates %*% beta)
  curve_covariance <- contrasts$curve_coordinates %*% V %*% t(contrasts$curve_coordinates)
  names(curve_beta) <- rownames(contrasts$curve_coordinates)
  list(contrasts = contrasts, time = time, average = average, shape = shape,
       curve_coefficients = curve_beta, curve_covariance = curve_covariance,
       time_covariance = contrasts$time %*% V %*% t(contrasts$time))
}

v7_cr2_cluster_diagnostic <- function(x) {
  required <- c("utterance_id", "match_id")
  if (!all(required %in% names(x)) || anyNA(x[, required])) stop("CR2_METADATA_FAILURE: missing utterance_id or match_id")
  memberships <- unique(data.frame(utterance_id=as.character(x$utterance_id),match_id=as.character(x$match_id),stringsAsFactors=FALSE))
  counts <- table(memberships$utterance_id)
  crossed <- counts > 1L
  list(evidence_complete=TRUE, unique_utterances=length(counts), utterances_multiple_matches=sum(crossed),
       proportion_utterances_multiple_matches=mean(crossed), maximum_matches_per_utterance=max(counts),
       match_clusters_independent_partitions=!any(crossed))
}

v7_cr2_sensitivity <- function(fit_result, times, covariance_function = NULL) {
  fit <- fit_result$fit
  x <- fit_result$data
  diagnostic <- v7_cr2_cluster_diagnostic(x)
  if (!diagnostic$match_clusters_independent_partitions) {
    return(list(status="NOT_APPLICABLE_CROSSED_CLUSTERING", applicable=FALSE, valid=FALSE,
                covariance=NULL, analysis=NULL, cluster="match_id", type="CR2", dimensions=c(NA_integer_,NA_integer_),
                minimum_eigen=NA_real_, estimates_produced=FALSE, diagnostic=diagnostic,
                explanation="Utterances span multiple matched sets, so match_id clusters do not partition utterance-level dependence. Primary mixed-model inference remains mandatory; no CR2 estimates are produced."))
  }
  if (is.null(covariance_function)) {
    if (!requireNamespace("clubSandwich", quietly=TRUE)) stop("CR2_UNAVAILABLE: clubSandwich is not installed")
    covariance_function <- function(model, cluster) clubSandwich::vcovCR(model, cluster=cluster, type="CR2")
  }
  covariance <- tryCatch(
    covariance_function(fit, x$match_id),
    error = function(e) stop("CR2_TECHNICAL_FAILURE: ", conditionMessage(e))
  )
  covariance <- as.matrix(covariance)
  wanted <- names(fixef(fit))
  if (!all(wanted %in% rownames(covariance)) || !all(wanted %in% colnames(covariance))) {
    stop("CR2_DIMENSION_FAILURE: fixed-effect names absent from covariance")
  }
  covariance <- covariance[wanted, wanted, drop = FALSE]
  eigenvalues <- eigen((covariance + t(covariance)) / 2, symmetric = TRUE, only.values = TRUE)$values
  if (any(!is.finite(covariance)) || min(eigenvalues) < -1e-8 * max(1, max(abs(eigenvalues)))) {
    stop("CR2_NONFINITE_OR_NONPSD_COVARIANCE")
  }
  analysis <- v7_analyze_specificity(fit_result, times, covariance_override = covariance)
  list(status="VALID_FINITE_PSD_CR2", applicable=TRUE, valid=TRUE, covariance=covariance, analysis=analysis,
       cluster="match_id", type="CR2", dimensions=dim(covariance), minimum_eigen=min(eigenvalues),
       estimates_produced=TRUE, diagnostic=diagnostic,
       explanation="Every utterance is contained within one matched set; finite PSD match-clustered CR2 covariance produced.")
}
