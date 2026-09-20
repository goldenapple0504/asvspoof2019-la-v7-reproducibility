#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) stop("usage: fit_v7_production_models.R OUTPUT_ROOT --execute-full [MODEL_ID]")
root <- normalizePath(args[1], winslash = "/", mustWork = TRUE)
library_candidates <- c(Sys.getenv("V7_R_LIB", unset = ""),
                        file.path(dirname(root), "rebuild_v6_trajectory", "r_libs"))
library_candidates <- library_candidates[nzchar(library_candidates) & dir.exists(library_candidates)]
if (length(library_candidates)) .libPaths(c(library_candidates, .libPaths()))
if (args[2] != "--execute-full") stop("Production fitting is locked; --execute-full required")
marker <- file.path(root, "FULL_V7_RUN_AUTHORIZED.txt")
if (!file.exists(marker) || trimws(readLines(marker, warn = FALSE)[1]) != "FULL_V7_RUN_AUTHORIZED") {
  stop("Production fitting authorization marker absent or invalid")
}
requested <- if (length(args) >= 3L) args[3] else ""
here_arg <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", here_arg[grepl("^--file=", here_arg)][1])
source(file.path(dirname(normalizePath(script_file, winslash = "/")), "v7_model_functions.R"))

input_dir <- file.path(root, "production", "model_data")
output_dir <- file.path(root, "production", "models")
contrast_dir <- file.path(root, "production", "contrasts")
diagnostic_dir <- file.path(root, "production", "diagnostics")
for (directory in c(output_dir, contrast_dir, diagnostic_dir)) dir.create(directory, recursive = TRUE, showWarnings = FALSE)

files <- sort(list.files(input_dir, pattern = "^[A][0-9]{2}_.+\\.tsv\\.gz$", full.names = TRUE))
if (nzchar(requested)) files <- files[sub("\\.tsv\\.gz$", "", basename(files)) == requested]
if (!length(files)) stop("No production model inputs selected")

write_tsv <- function(x, path, row.names = FALSE, col.names = TRUE) {
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  temporary <- paste0(path, ".tmp")
  write.table(x, temporary, sep = "\t", quote = FALSE, row.names = row.names, col.names = col.names, na = "")
  if (!file.rename(temporary, path)) stop("Atomic rename failed: ", path)
}

atomic_save_rds <- function(x, path) {
  temporary <- paste0(path, ".tmp")
  saveRDS(x, temporary, compress = "xz")
  if (!file.rename(temporary, path)) stop("Atomic RDS rename failed: ", path)
}

for (path in files) {
  model_id <- sub("\\.tsv\\.gz$", "", basename(path))
  pieces <- strsplit(model_id, "_", fixed = TRUE)[[1]]
  system <- pieces[1]
  feature <- paste(pieces[-1], collapse = "_")
  checkpoint <- file.path(diagnostic_dir, paste0(model_id, "_acceptance.tsv"))
  model_path <- file.path(output_dir, paste0(model_id, ".rds"))
  if (file.exists(checkpoint) && file.exists(model_path)) {
    old <- read.delim(checkpoint, stringsAsFactors = FALSE)
    if (nrow(old) == 1L && isTRUE(old$all_acceptance_gates_pass[1])) { cat("CHECKPOINT SKIP", model_id, "\n"); next }
  }
  export_resume <- FALSE
  if (file.exists(model_path)) {
    bundle <- readRDS(model_path)
    if (is.null(bundle$acceptance) || is.null(bundle$times)) {
      stop("Existing model object lacks the prespecified post-fit export-resume checkpoint: ", model_id)
    }
    result <- list(fit = bundle$model, attempts = bundle$attempts)
    inference <- bundle$inference
    cr2 <- bundle$CR2_sensitivity
    acceptance <- bundle$acceptance
    times <- bundle$times
    export_resume <- TRUE
  } else {
    data <- read.delim(gzfile(path), stringsAsFactors = FALSE)
    result <- v7_fit_rank_safe(data)
    times <- sort(unique(data$relative_time_ms))
    inference <- v7_analyze_specificity(result, times)
    cr2 <- v7_cr2_sensitivity(result, times)
    cr2_valid <- identical(cr2$status, "VALID_FINITE_PSD_CR2")
    cr2_not_applicable <- identical(cr2$status, "NOT_APPLICABLE_CROSSED_CLUSTERING")
    cr2_authorized <- cr2_valid || (cr2_not_applicable && isTRUE(cr2$diagnostic$evidence_complete)
      && cr2$diagnostic$utterances_multiple_matches > 0L && !isTRUE(cr2$estimates_produced))
    fixed_pass <- result$fixed_rank$pass
    singular_pass <- !isSingular(result$fit, tol = 1e-5)
    convergence_pass <- !length(result$fit@optinfo$conv$lme4$messages)
    covariance_eigen <- eigen((inference$curve_covariance + t(inference$curve_covariance)) / 2,
                              symmetric = TRUE, only.values = TRUE)$values
    covariance_pass <- all(is.finite(covariance_eigen)) && min(covariance_eigen) >= -1e-8 * max(1, max(abs(covariance_eigen)))
    acceptance <- data.frame(
      model_id = model_id, system = system, feature = feature,
      fixed_columns = result$fixed_rank$columns, fixed_rank = result$fixed_rank$rank,
      fixed_full_rank = fixed_pass, fixed_condition_number = result$fixed_rank$condition,
      final_singular = !singular_pass, converged = convergence_pass,
      final_random_structure = paste(result$final_random_components, collapse = " + "),
      curve_covariance_PSD = covariance_pass,
      CR2_status = cr2$status, CR2_cluster = cr2$cluster, CR2_type = cr2$type,
      CR2_applicable = cr2$applicable, CR2_evidence_complete = cr2$diagnostic$evidence_complete,
      CR2_unique_utterances = cr2$diagnostic$unique_utterances,
      CR2_utterances_multiple_matches = cr2$diagnostic$utterances_multiple_matches,
      CR2_proportion_utterances_multiple_matches = cr2$diagnostic$proportion_utterances_multiple_matches,
      CR2_maximum_matches_per_utterance = cr2$diagnostic$maximum_matches_per_utterance,
      CR2_match_clusters_independent_partitions = cr2$diagnostic$match_clusters_independent_partitions,
      CR2_estimates_produced = cr2$estimates_produced,
      CR2_rows = cr2$dimensions[1], CR2_columns = cr2$dimensions[2],
      CR2_covariance_finite = if (cr2_valid) all(is.finite(cr2$covariance)) else NA,
      CR2_covariance_min_eigen = cr2$minimum_eigen,
      CR2_success = cr2_valid, CR2_authorized_outcome = cr2_authorized,
      all_acceptance_gates_pass = fixed_pass && singular_pass && convergence_pass && covariance_pass && cr2_authorized,
      warnings = paste(result$warnings, collapse = " | "), stringsAsFactors = FALSE
    )
    if (!acceptance$all_acceptance_gates_pass) {
      write_tsv(acceptance, checkpoint)
      stop("Production model acceptance gate failed: ", model_id)
    }
    atomic_save_rds(list(model = result$fit, formula = formula(result$fit), attempts = result$attempts,
                         inference = inference, CR2_sensitivity = cr2, acceptance = acceptance, times = times,
                         specification = "V7-production-implementation-2-cr2-preflight"), model_path)
  }
  write_tsv(result$attempts, file.path(diagnostic_dir, paste0(model_id, "_random_hierarchy.tsv")))
  write_tsv(inference$contrasts$cell_weights, file.path(contrast_dir, paste0(model_id, "_cell_weights.tsv")))
  write_tsv(data.frame(relative_time_ms = times, inference$contrasts$time, check.names = FALSE),
            file.path(contrast_dir, paste0(model_id, "_time_contrast_matrix.tsv")))
  write_tsv(data.frame(target = "equal_grid_average", inference$contrasts$average, check.names = FALSE),
            file.path(contrast_dir, paste0(model_id, "_average_contrast_matrix.tsv")))
  write_tsv(data.frame(target = rownames(inference$contrasts$shape), inference$contrasts$shape, check.names = FALSE),
            file.path(contrast_dir, paste0(model_id, "_shape_contrast_matrix.tsv")))
  write_tsv(cbind(system = system, feature = feature, inference$time),
            file.path(contrast_dir, paste0(model_id, "_time_inference_raw.tsv")))
  write_tsv(cbind(system = system, feature = feature, inference$average),
            file.path(contrast_dir, paste0(model_id, "_average_inference_raw.tsv")))
  write_tsv(cbind(system = system, feature = feature, inference$shape),
            file.path(contrast_dir, paste0(model_id, "_shape_inference_raw.tsv")))
  write_tsv(data.frame(model_id=model_id,system=system,feature=feature,status=cr2$status,
    evidence_complete=cr2$diagnostic$evidence_complete,unique_utterances=cr2$diagnostic$unique_utterances,
    utterances_multiple_matches=cr2$diagnostic$utterances_multiple_matches,
    proportion_utterances_multiple_matches=cr2$diagnostic$proportion_utterances_multiple_matches,
    maximum_matches_per_utterance=cr2$diagnostic$maximum_matches_per_utterance,
    match_clusters_independent_partitions=cr2$diagnostic$match_clusters_independent_partitions,
    estimates_produced=cr2$estimates_produced,explanation=cr2$explanation),
    file.path(diagnostic_dir,paste0(model_id,"_CR2_status.tsv")))
  if (cr2_valid) {
    write_tsv(cbind(system=system,feature=feature,cr2$analysis$time),file.path(contrast_dir,paste0(model_id,"_time_inference_CR2_raw.tsv")))
    write_tsv(cbind(system=system,feature=feature,cr2$analysis$average),file.path(contrast_dir,paste0(model_id,"_average_inference_CR2_raw.tsv")))
    write_tsv(cbind(system=system,feature=feature,cr2$analysis$shape),file.path(contrast_dir,paste0(model_id,"_shape_inference_CR2_raw.tsv")))
    write_tsv(cr2$covariance,file.path(contrast_dir,paste0(model_id,"_fixed_effect_covariance_CR2.tsv")),row.names=FALSE,col.names=FALSE)
  }
  write_tsv(data.frame(coordinate = names(inference$curve_coefficients), estimate = inference$curve_coefficients),
            file.path(contrast_dir, paste0(model_id, "_specificity_coefficients.tsv")))
  write_tsv(inference$curve_covariance,
            file.path(contrast_dir, paste0(model_id, "_specificity_coefficient_covariance.tsv")),
            row.names = FALSE, col.names = FALSE)
  write_tsv(inference$time_covariance,
            file.path(contrast_dir, paste0(model_id, "_time_covariance.tsv")), row.names = FALSE, col.names = FALSE)
  write_tsv(acceptance, checkpoint)
  cat(if (export_resume) "EXPORT-RESUME PASS" else "PASS", model_id,
      "random=", acceptance$final_random_structure, "\n")
}
