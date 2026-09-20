options(stringsAsFactors = FALSE, digits = 17)

args <- commandArgs(trailingOnly = TRUE)
package_root <- if (length(args)) normalizePath(args[[1]], mustWork = TRUE) else normalizePath(".", mustWork = TRUE)
result_dir <- file.path(package_root, "rq1_direct_authenticity_results")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)
v7_r_lib <- Sys.getenv("V7_R_LIB", unset = "")
if (nzchar(v7_r_lib)) .libPaths(c(normalizePath(v7_r_lib, mustWork = TRUE), .libPaths()))

regions <- c("Boundary", "Interior_A", "Interior_B")
validation_tolerance <- 1e-10
alpha <- 0.05
analysis_warnings <- character()

capture_warning <- function(expr) {
  withCallingHandlers(
    expr,
    warning = function(w) {
      analysis_warnings <<- unique(c(analysis_warnings, conditionMessage(w)))
      invokeRestart("muffleWarning")
    }
  )
}

write_tsv <- function(x, name) {
  write.table(
    x,
    file = file.path(result_dir, name),
    sep = "\t",
    row.names = FALSE,
    col.names = TRUE,
    quote = FALSE,
    na = "NA",
    fileEncoding = "UTF-8"
  )
}

source(file.path(package_root, "code", "v7_model_functions.R"), local = FALSE)

model_index <- read.delim(file.path(package_root, "MODEL_INDEX.tsv"), check.names = FALSE)
if (nrow(model_index) != 60L) stop("MODEL_INDEX_ROW_COUNT_NOT_60: ", nrow(model_index))
if (anyDuplicated(model_index[c("system", "feature")])) stop("DUPLICATE_SYSTEM_FEATURE_IN_MODEL_INDEX")
model_paths <- file.path(package_root, gsub("/", .Platform$file.sep, model_index$model_file, fixed = TRUE))
if (any(!file.exists(model_paths))) stop("MISSING_MODEL_BUNDLE: ", paste(model_index$model_file[!file.exists(model_paths)], collapse = ", "))

official_specificity <- read.delim(
  file.path(package_root, "reference_inference", "time_resolved_specificity.tsv"),
  check.names = FALSE
)

make_direct_L <- function(fit, times, region, phone_pair_level) {
  frame <- model.frame(fit)
  nd <- do.call(rbind, lapply(times, function(time) {
    data.frame(
      authenticity = factor(c("BF", "TTS"), levels = levels(frame$authenticity)),
      region = factor(rep(region, 2L), levels = levels(frame$region)),
      relative_time_ms = rep(time, 2L),
      duration_A_c = 0,
      duration_B_c = 0,
      exact_phone_pair = factor(rep(phone_pair_level, 2L), levels = levels(frame$exact_phone_pair))
    )
  }))
  X <- v7_fixed_matrix(fit, nd)
  L <- t(vapply(seq_along(times), function(i) {
    X[2L * i, , drop = TRUE] - X[2L * i - 1L, , drop = TRUE]
  }, numeric(ncol(X))))
  dimnames(L) <- list(as.character(times), colnames(X))
  L
}

scalar_results <- function(L, beta, V) {
  estimate <- as.vector(L %*% beta)
  covariance <- L %*% V %*% t(L)
  variance <- diag(covariance)
  if (any(!is.finite(variance)) || any(variance <= 0)) stop("NONPOSITIVE_OR_NONFINITE_DIRECT_VARIANCE")
  SE <- sqrt(variance)
  z <- estimate / SE
  p <- 2 * pnorm(-abs(z))
  data.frame(
    estimate_tts_minus_bf = estimate,
    SE = SE,
    CI95_low = estimate - qnorm(0.975) * SE,
    CI95_high = estimate + qnorm(0.975) * SE,
    wald_z = z,
    p_raw = p
  )
}

joint_wald <- function(C, beta, V, label) {
  tryCatch({
    decomposition <- qr(t(C), tol = 1e-9)
    rank <- decomposition$rank
    if (rank < 1L) stop("ZERO_RANK_CONTRAST_SPACE")
    rows <- decomposition$pivot[seq_len(rank)]
    C_basis <- C[rows, , drop = FALSE]
    estimate <- as.vector(C_basis %*% beta)
    covariance <- C_basis %*% V %*% t(C_basis)
    if (any(!is.finite(covariance))) stop("NONFINITE_COVARIANCE")
    if (qr(covariance, tol = 1e-10)$rank != rank) stop("RANK_DEFICIENT_COVARIANCE")
    statistic <- as.numeric(crossprod(estimate, solve(covariance, estimate)))
    if (!is.finite(statistic) || statistic < -1e-10) stop("INVALID_WALD_STATISTIC")
    data.frame(
      test_type = label,
      df = rank,
      wald_statistic = max(0, statistic),
      p_raw = pchisq(max(0, statistic), df = rank, lower.tail = FALSE),
      status = "PASS"
    )
  }, error = function(e) {
    data.frame(
      test_type = label,
      df = NA_integer_,
      wald_statistic = NA_real_,
      p_raw = NA_real_,
      status = paste0("NOT_ESTIMABLE: ", conditionMessage(e))
    )
  })
}

time_parts <- vector("list", nrow(model_index) * length(regions))
average_parts <- vector("list", nrow(model_index) * length(regions))
trajectory_parts <- vector("list", nrow(model_index) * length(regions) * 2L)
validation_parts <- vector("list", nrow(model_index))
phone_pair_parts <- vector("list", nrow(model_index))
time_position <- 0L
average_position <- 0L
trajectory_position <- 0L

for (model_number in seq_len(nrow(model_index))) {
  index_row <- model_index[model_number, , drop = FALSE]
  system <- as.character(index_row$system)
  feature <- as.character(index_row$feature)
  bundle <- capture_warning(readRDS(model_paths[[model_number]]))

  required_bundle_fields <- c("model", "formula", "acceptance", "times")
  missing_fields <- setdiff(required_bundle_fields, names(bundle))
  if (length(missing_fields)) stop("BUNDLE_FIELDS_MISSING_", system, "_", feature, ": ", paste(missing_fields, collapse = ", "))
  fit <- bundle$model
  if (!inherits(fit, "merMod")) stop("BUNDLE_MODEL_NOT_MERMOD_", system, "_", feature)
  times <- as.numeric(bundle$times)
  if (any(!is.finite(times)) || anyDuplicated(times)) stop("INVALID_TIME_GRID_", system, "_", feature)

  frame <- model.frame(fit)
  if (!identical(levels(frame$authenticity), c("BF", "TTS"))) stop("AUTHENTICITY_REFERENCE_FAILURE_", system, "_", feature)
  if (!identical(levels(frame$region), regions)) stop("REGION_REFERENCE_FAILURE_", system, "_", feature)
  phone_levels <- levels(frame$exact_phone_pair)
  if (length(phone_levels) < 2L) stop("FEWER_THAN_TWO_PHONE_PAIR_LEVELS_", system, "_", feature)

  beta <- lme4::fixef(fit)
  V <- as.matrix(stats::vcov(fit))
  if (any(!is.finite(beta)) || any(!is.finite(V))) stop("NONFINITE_FIXED_EFFECTS_OR_COVARIANCE_", system, "_", feature)
  if (!identical(names(beta), rownames(V)) || !identical(names(beta), colnames(V))) stop("FIXED_COVARIANCE_NAME_MISMATCH_", system, "_", feature)

  # The copied function expects an object with $fit, while transferred bundles
  # store the finalized merMod under $model. This adapter does not mutate or refit.
  production <- capture_warning(v7_analyze_specificity(list(fit = fit), times))
  official <- official_specificity[
    official_specificity$system == system & official_specificity$feature == feature,
    , drop = FALSE
  ]
  official <- official[match(times, official$relative_time_ms), , drop = FALSE]
  if (nrow(official) != length(times) || anyNA(official$relative_time_ms)) stop("OFFICIAL_SPECIFICITY_GRID_MISMATCH_", system, "_", feature)

  direct_L <- setNames(vector("list", length(regions)), regions)
  pair1 <- phone_levels[[1L]]
  pair2 <- phone_levels[[length(phone_levels)]]
  phone_differences <- numeric(length(regions))

  for (region_number in seq_along(regions)) {
    region <- regions[[region_number]]
    L <- make_direct_L(fit, times, region, pair1)
    L_pair2 <- make_direct_L(fit, times, region, pair2)
    phone_differences[[region_number]] <- max(abs(L - L_pair2))
    direct_L[[region]] <- L

    scalar <- scalar_results(L, beta, V)
    time_position <- time_position + 1L
    time_parts[[time_position]] <- data.frame(
      system = system,
      feature = feature,
      region = region,
      relative_time_ms = times,
      scalar,
      stringsAsFactors = FALSE
    )

    average_L <- matrix(colMeans(L), nrow = 1L, dimnames = list("equal_grid_average", colnames(L)))
    average_scalar <- scalar_results(average_L, beta, V)
    average_position <- average_position + 1L
    average_parts[[average_position]] <- data.frame(
      system = system,
      feature = feature,
      region = region,
      average_scalar,
      stringsAsFactors = FALSE
    )

    trajectory_position <- trajectory_position + 1L
    trajectory_parts[[trajectory_position]] <- cbind(
      data.frame(system = system, feature = feature, region = region),
      joint_wald(L, beta, V, "global_trajectory")
    )
    centered_L <- sweep(L, 2L, colMeans(L), "-")
    trajectory_position <- trajectory_position + 1L
    trajectory_parts[[trajectory_position]] <- cbind(
      data.frame(system = system, feature = feature, region = region),
      joint_wald(centered_L, beta, V, "shape")
    )
  }

  phone_pair_parts[[model_number]] <- data.frame(
    system = system,
    feature = feature,
    phone_pair_level_1 = pair1,
    phone_pair_level_2 = pair2,
    regions_and_times_checked = sum(vapply(direct_L, nrow, integer(1))),
    max_abs_contrast_vector_difference = max(phone_differences),
    pass = max(phone_differences) <= validation_tolerance,
    stringsAsFactors = FALSE
  )

  L_specificity <- direct_L$Boundary - 0.5 * (direct_L$Interior_A + direct_L$Interior_B)
  reconstructed_estimate <- as.vector(L_specificity %*% beta)
  reconstructed_SE <- sqrt(diag(L_specificity %*% V %*% t(L_specificity)))
  production_estimate <- as.numeric(production$time$estimate)
  production_SE <- as.numeric(production$time$SE)
  validation_parts[[model_number]] <- data.frame(
    system = system,
    feature = feature,
    relative_time_ms = times,
    official_estimate = official$estimate,
    production_helper_estimate = production_estimate,
    abs_production_estimate_difference = abs(official$estimate - production_estimate),
    reconstructed_estimate = reconstructed_estimate,
    abs_estimate_difference = abs(official$estimate - reconstructed_estimate),
    official_SE = official$SE,
    production_helper_SE = production_SE,
    abs_production_SE_difference = abs(official$SE - production_SE),
    reconstructed_SE = reconstructed_SE,
    abs_SE_difference = abs(official$SE - reconstructed_SE),
    pass_production_estimate = abs(official$estimate - production_estimate) <= validation_tolerance,
    pass_production_SE = abs(official$SE - production_SE) <= validation_tolerance,
    pass_estimate = abs(official$estimate - reconstructed_estimate) <= validation_tolerance,
    pass_SE = abs(official$SE - reconstructed_SE) <= validation_tolerance,
    tolerance = validation_tolerance,
    stringsAsFactors = FALSE
  )

  rm(bundle, fit, frame, beta, V, production, official, direct_L, L_specificity)
  invisible(gc(full = TRUE))
}

time_results <- do.call(rbind, time_parts)
average_results <- do.call(rbind, average_parts)
trajectory_results <- do.call(rbind, trajectory_parts)
validation_results <- do.call(rbind, validation_parts)
phone_pair_results <- do.call(rbind, phone_pair_parts)

region_order <- match(time_results$region, regions)
time_results <- time_results[order(time_results$system, time_results$feature, region_order, time_results$relative_time_ms), ]
trajectory_keys <- interaction(time_results$system, time_results$feature, time_results$region, drop = TRUE)
time_results$p_bonferroni_within_trajectory <- ave(
  time_results$p_raw,
  trajectory_keys,
  FUN = function(p) p.adjust(p, method = "bonferroni")
)
time_results$p_BH_global <- p.adjust(time_results$p_raw, method = "BH")
time_results <- time_results[c(
  "system", "feature", "region", "relative_time_ms", "estimate_tts_minus_bf",
  "SE", "CI95_low", "CI95_high", "wald_z", "p_raw",
  "p_bonferroni_within_trajectory", "p_BH_global"
)]

average_region_order <- match(average_results$region, regions)
average_results <- average_results[order(average_results$system, average_results$feature, average_region_order), ]
average_results$p_BH <- p.adjust(average_results$p_raw, method = "BH")
average_results <- average_results[c(
  "system", "feature", "region", "estimate_tts_minus_bf", "SE",
  "CI95_low", "CI95_high", "wald_z", "p_raw", "p_BH"
)]

trajectory_region_order <- match(trajectory_results$region, regions)
trajectory_results <- trajectory_results[order(
  trajectory_results$system,
  trajectory_results$feature,
  trajectory_region_order,
  match(trajectory_results$test_type, c("global_trajectory", "shape"))
), ]
trajectory_results$p_BH <- NA_real_
for (test_name in c("global_trajectory", "shape")) {
  selected <- trajectory_results$test_type == test_name & trajectory_results$status == "PASS"
  trajectory_results$p_BH[selected] <- p.adjust(trajectory_results$p_raw[selected], method = "BH")
}
trajectory_results <- trajectory_results[c(
  "system", "feature", "region", "test_type", "df", "wald_statistic", "p_raw", "p_BH", "status"
)]

validation_results <- validation_results[order(
  validation_results$system,
  validation_results$feature,
  validation_results$relative_time_ms
), ]
phone_pair_results <- phone_pair_results[order(phone_pair_results$system, phone_pair_results$feature), ]

max_estimate_discrepancy <- max(validation_results$abs_estimate_difference)
max_SE_discrepancy <- max(validation_results$abs_SE_difference)
production_max_estimate_discrepancy <- max(validation_results$abs_production_estimate_difference)
production_max_SE_discrepancy <- max(validation_results$abs_production_SE_difference)
validation_failures <- sum(
  !validation_results$pass_production_estimate |
    !validation_results$pass_production_SE |
    !validation_results$pass_estimate |
    !validation_results$pass_SE
)
specificity_validation_pass <- validation_failures == 0L
phone_pair_invariance_pass <- all(phone_pair_results$pass)

write_tsv(validation_results, "reconstruction_validation.tsv")
write_tsv(phone_pair_results, "exact_phone_pair_invariance.tsv")

if (!specificity_validation_pass) {
  stop(
    "SPECIFICITY_VALIDATION_FAILED: failures=", validation_failures,
    " max_estimate_difference=", format(max_estimate_discrepancy, digits = 17),
    " max_SE_difference=", format(max_SE_discrepancy, digits = 17)
  )
}
if (!phone_pair_invariance_pass) stop("EXACT_PHONE_PAIR_INVARIANCE_FAILED")

expected_time_rows <- 1590L
expected_average_rows <- 180L
expected_trajectory_rows <- 360L
if (nrow(time_results) != expected_time_rows) stop("TIME_ROW_COUNT_FAILURE: ", nrow(time_results))
if (nrow(average_results) != expected_average_rows) stop("AVERAGE_ROW_COUNT_FAILURE: ", nrow(average_results))
if (nrow(trajectory_results) != expected_trajectory_rows) stop("TRAJECTORY_ROW_COUNT_FAILURE: ", nrow(trajectory_results))

finite_columns <- list(
  time = c("estimate_tts_minus_bf", "SE", "CI95_low", "CI95_high", "wald_z", "p_raw", "p_bonferroni_within_trajectory", "p_BH_global"),
  average = c("estimate_tts_minus_bf", "SE", "CI95_low", "CI95_high", "wald_z", "p_raw", "p_BH")
)
if (any(!is.finite(as.matrix(time_results[finite_columns$time])))) stop("NONFINITE_TIME_RESULT")
if (any(!is.finite(as.matrix(average_results[finite_columns$average])))) stop("NONFINITE_AVERAGE_RESULT")
if (any(trajectory_results$status != "PASS")) {
  analysis_warnings <- unique(c(analysis_warnings, "One or more trajectory tests were not estimable; see trajectory_tests.tsv status."))
} else if (any(!is.finite(as.matrix(trajectory_results[c("df", "wald_statistic", "p_raw", "p_BH")])))) {
  stop("NONFINITE_TRAJECTORY_RESULT_WITH_PASS_STATUS")
}

write_tsv(time_results, "time_resolved_authenticity.tsv")
write_tsv(average_results, "average_authenticity.tsv")
write_tsv(trajectory_results, "trajectory_tests.tsv")

summary_groups <- split(average_results, interaction(average_results$feature, average_results$region, drop = TRUE))
feature_region_summary <- do.call(rbind, lapply(summary_groups, function(x) {
  data.frame(
    feature = x$feature[[1L]],
    region = x$region[[1L]],
    systems_positive = sum(x$estimate_tts_minus_bf > 0),
    systems_negative = sum(x$estimate_tts_minus_bf < 0),
    BH_significant_positive = sum(x$estimate_tts_minus_bf > 0 & x$p_BH < alpha),
    BH_significant_negative = sum(x$estimate_tts_minus_bf < 0 & x$p_BH < alpha),
    median_average_effect = median(x$estimate_tts_minus_bf),
    minimum_average_effect = min(x$estimate_tts_minus_bf),
    maximum_average_effect = max(x$estimate_tts_minus_bf)
  )
}))
feature_region_summary <- feature_region_summary[order(
  feature_region_summary$feature,
  match(feature_region_summary$region, regions)
), ]
rownames(feature_region_summary) <- NULL
write_tsv(feature_region_summary, "feature_region_summary.tsv")

system_feature_keys <- unique(average_results[c("system", "feature")])
system_feature_keys <- system_feature_keys[order(system_feature_keys$system, system_feature_keys$feature), ]
system_feature_summary <- system_feature_keys
for (region in regions) {
  avg_region <- average_results[average_results$region == region, ]
  global_region <- trajectory_results[
    trajectory_results$region == region & trajectory_results$test_type == "global_trajectory",
  ]
  shape_region <- trajectory_results[
    trajectory_results$region == region & trajectory_results$test_type == "shape",
  ]
  key <- paste(system_feature_summary$system, system_feature_summary$feature, sep = "|")
  avg_match <- match(key, paste(avg_region$system, avg_region$feature, sep = "|"))
  global_match <- match(key, paste(global_region$system, global_region$feature, sep = "|"))
  shape_match <- match(key, paste(shape_region$system, shape_region$feature, sep = "|"))
  system_feature_summary[[paste0(region, "_average_estimate_tts_minus_bf")]] <- avg_region$estimate_tts_minus_bf[avg_match]
  system_feature_summary[[paste0(region, "_average_BH_significant")]] <- avg_region$p_BH[avg_match] < alpha
  system_feature_summary[[paste0(region, "_global_trajectory_BH_significant")]] <- global_region$status[global_match] == "PASS" & global_region$p_BH[global_match] < alpha
  system_feature_summary[[paste0(region, "_shape_BH_significant")]] <- shape_region$status[shape_match] == "PASS" & shape_region$p_BH[shape_match] < alpha
}
write_tsv(system_feature_summary, "system_feature_summary.tsv")

runtime_packages <- c("lme4", "Matrix", "reformulas", "splines", "Rcpp", "RcppEigen", "minqa", "nloptr")
runtime_versions <- vapply(runtime_packages, function(p) as.character(packageVersion(p)), character(1))
runtime_rows <- paste0("| ", runtime_packages, " | ", runtime_versions, " |")
hash_rows <- paste0(
  "| ", model_index$system,
  " | ", model_index$feature,
  " | `", model_index$model_file,
  "` | `", model_index$sha256, "` |"
)
time_grid_lines <- vapply(sort(unique(model_index$feature)), function(feature) {
  feature_times <- sort(unique(time_results$relative_time_ms[time_results$feature == feature]))
  paste0("- `", feature, "`: ", paste(feature_times, collapse = ", "), " ms")
}, character(1))

runtime_note <- Sys.getenv(
  "V7_RUNTIME_NOTE",
  unset = "Exact recorded package versions were loaded under R 4.6.1."
)
warning_lines <- if (length(analysis_warnings)) paste0("- ", analysis_warnings) else "- None."
readme <- c(
  "# RQ1 direct authenticity post-estimation results",
  "",
  paste0("Generated: ", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "Asia/Ho_Chi_Minh")),
  "",
  "## Task and scope",
  "",
  "This post-estimation extension answers RQ1 using the 60 finalized V7 model bundles. It estimates the population-level fixed-effect contrast `TTS - BF` for every system, feature, region, and official time point. No feature extraction, matching, model fitting, refitting, or finalized production inference was rerun.",
  "",
  "The transferred model, code, metadata, and reference-inference files remained unchanged; their sizes and SHA-256 hashes were verified against `MANIFEST.tsv` before and after execution. All new files were written only under `rq1_direct_authenticity_results/`.",
  "",
  "This is a new post-estimation inference family and was not part of the original finalized V7 specificity analysis.",
  "",
  "## Model and estimand",
  "",
  "- Models loaded: 60 (10 systems x 6 features).",
  "- Formula: `feature_z ~ authenticity * region * splines::ns(relative_time_ms, df = 4) + duration_A_c + duration_B_c + exact_phone_pair` plus each bundle's accepted random structure.",
  "- Authenticity factor levels/reference: `BF`, `TTS`; reference = `BF`.",
  "- Region factor levels/reference: `Boundary`, `Interior_A`, `Interior_B`; reference = `Boundary`.",
  "- Direct estimand: `A_region(t) = TTS - BF`.",
  "- Nuisance settings: `duration_A_c = 0`, `duration_B_c = 0`, and the same valid `exact_phone_pair` level for paired BF/TTS design rows.",
  "- Fixed effects and their full model-based covariance only; no random effects were added to predictions.",
  "- The transferred specification does not explicitly authorize SESOI +/-0.10 for this direct estimand, so no TOST/equivalence classification is reported.",
  "",
  "## Official time grids",
  "",
  time_grid_lines,
  "",
  "## Multiplicity families",
  "",
  "- Time-resolved: raw p-values retained; Bonferroni within each system x feature x region trajectory; BH across all 1,590 direct time-point tests.",
  "- Region averages: BH across all 180 system x feature x region tests.",
  "- Trajectory tests: separate BH families across the 180 global-trajectory tests and 180 shape tests.",
  "",
  "## Trajectory tests",
  "",
  "The global test is a joint Wald test over an independent basis for the complete direct-authenticity trajectory contrast space. The shape test centers the time-point contrast rows by their equal-grid mean and applies a joint Wald test to an independent basis for the non-constant contrast space.",
  "",
  "## Mandatory specificity validation",
  "",
  "For each bundle, the copied `v7_analyze_specificity()` implementation was run using `list(fit = bundle$model)` because the copied helper expects a `$fit` field while the transfer bundle stores the finalized model in `$model`. This is a field-name adapter only; no model was changed or refit.",
  "",
  "Its time-resolved specificity estimates and SEs were compared with `reference_inference/time_resolved_specificity.tsv`. The same finalized specificity was then reconstructed independently as `L_Boundary - 0.5 * (L_Interior_A + L_Interior_B)` using the joint fixed-effect covariance.",
  "",
  paste0("- Tolerance: ", format(validation_tolerance, scientific = TRUE)),
  paste0("- Validation failures: ", validation_failures),
  paste0("- Maximum independent reconstruction estimate discrepancy: ", format(max_estimate_discrepancy, digits = 17)),
  paste0("- Maximum independent reconstruction SE discrepancy: ", format(max_SE_discrepancy, digits = 17)),
  paste0("- Maximum production-helper estimate discrepancy: ", format(production_max_estimate_discrepancy, digits = 17)),
  paste0("- Maximum production-helper SE discrepancy: ", format(production_max_SE_discrepancy, digits = 17)),
  paste0("- Result: ", if (specificity_validation_pass) "PASS" else "FAIL"),
  "",
  "Exact-phone-pair invariance was checked for every model, all three regions, and every official time using the first and last valid pair levels. See `exact_phone_pair_invariance.tsv`.",
  "",
  "## Runtime",
  "",
  paste0("- ", R.version.string),
  paste0("- ", runtime_note),
  "",
  "| Package | Version |",
  "|---|---:|",
  runtime_rows,
  "",
  "## Output files",
  "",
  "- `time_resolved_authenticity.tsv`: 1,590 direct time-point contrasts.",
  "- `average_authenticity.tsv`: 180 joint-covariance equal-grid averages.",
  "- `trajectory_tests.tsv`: 360 joint Wald tests.",
  "- `reconstruction_validation.tsv`: 530 finalized-specificity validation rows.",
  "- `exact_phone_pair_invariance.tsv`: deterministic nuisance-level invariance checks.",
  "- `feature_region_summary.tsv`: descriptive cross-system sign/significance summaries without pooling.",
  "- `system_feature_summary.tsv`: region-wise descriptive support for each system-feature model.",
  "- `run_rq1_direct_authenticity.R`: analysis script used.",
  "",
  "## Warnings",
  "",
  warning_lines,
  "",
  "## Source bundle hashes",
  "",
  "| System | Feature | Model file | SHA-256 |",
  "|---|---|---|---|",
  hash_rows
)
writeLines(readme, file.path(result_dir, "README.md"), useBytes = TRUE)

cat("models loaded: ", nrow(model_index), "\n", sep = "")
cat("time-resolved rows: ", nrow(time_results), "\n", sep = "")
cat("average rows: ", nrow(average_results), "\n", sep = "")
cat("trajectory-test rows: ", nrow(trajectory_results), "\n", sep = "")
cat("specificity validation: ", if (specificity_validation_pass) "PASS" else "FAIL", "\n", sep = "")
cat("maximum estimate discrepancy: ", format(max_estimate_discrepancy, digits = 17), "\n", sep = "")
cat("maximum SE discrepancy: ", format(max_SE_discrepancy, digits = 17), "\n", sep = "")
cat("exact-phone-pair invariance: ", if (phone_pair_invariance_pass) "PASS" else "FAIL", "\n", sep = "")
cat("model refit: NO\n")
cat("frozen transferred files modified: NO\n")
cat("result directory: ", normalizePath(result_dir, winslash = "/", mustWork = TRUE), "\n", sep = "")
cat("warnings: ", if (length(analysis_warnings)) paste(analysis_warnings, collapse = " | ") else "none", "\n", sep = "")
