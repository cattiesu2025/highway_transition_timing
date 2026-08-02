#!/usr/bin/env Rscript

# Single-panel training diagnostic for the multi-lane lane-change experiment.
#
# Figure contract
# - Core conclusion: FD, BAL, and SP follow distinct reward-optimization
#   trajectories in the multi-lane lane-change environment.
# - Archetype: single quantitative comparison panel.
# - Interpretation risk: the agents optimize different reward weights, so the
#   absolute returns are diagnostic and are not direct performance rankings.
# - Source data: one progress.csv per agent under training_logs/<agent>/.

local_lib <- file.path("tmp", "r-lib")
if (dir.exists(local_lib)) {
  .libPaths(c(normalizePath(local_lib), .libPaths()))
}

local_cache <- file.path("tmp", "r-cache")
dir.create(local_cache, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(XDG_CACHE_HOME = normalizePath(local_cache))

required_packages <- c("ggplot2", "svglite", "ragg")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "Missing required R packages: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages(library(ggplot2))

parse_args <- function(args) {
  options <- list(
    run_dir = "outputs/multilane_open_lane_change_mixed_100k",
    out = "outputs/report_figures/multilane_training_diagnostics",
    metric = "rollout/ep_rew_mean"
  )

  index <- 1L
  while (index <= length(args)) {
    flag <- args[[index]]
    if (!flag %in% c("--run-dir", "--out", "--metric")) {
      stop("Unknown argument: ", flag, call. = FALSE)
    }
    if (index == length(args)) {
      stop("Missing value after ", flag, call. = FALSE)
    }
    key <- switch(
      flag,
      "--run-dir" = "run_dir",
      "--out" = "out",
      "--metric" = "metric"
    )
    options[[key]] <- args[[index + 1L]]
    index <- index + 2L
  }

  options
}

read_agent_progress <- function(run_dir, agent, metric) {
  path <- file.path(run_dir, "training_logs", agent, "progress.csv")
  if (!file.exists(path)) {
    stop("Required training log is missing: ", path, call. = FALSE)
  }

  progress <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  required_columns <- c("time/total_timesteps", metric)
  missing_columns <- setdiff(required_columns, names(progress))
  if (length(missing_columns) > 0) {
    stop(
      "Missing column(s) in ", path, ": ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  data.frame(
    agent = agent,
    timesteps = as.numeric(progress[["time/total_timesteps"]]),
    mean_episode_reward = as.numeric(progress[[metric]]),
    stringsAsFactors = FALSE
  )
}

save_figure <- function(plot, stem, width_mm = 89, height_mm = 70, dpi = 600) {
  dir.create(dirname(stem), recursive = TRUE, showWarnings = FALSE)
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4

  svglite::svglite(
    paste0(stem, ".svg"),
    width = width_in,
    height = height_in,
    bg = "white"
  )
  print(plot)
  invisible(grDevices::dev.off())

  grDevices::cairo_pdf(
    paste0(stem, ".pdf"),
    width = width_in,
    height = height_in,
    family = "Arial",
    bg = "white"
  )
  print(plot)
  invisible(grDevices::dev.off())

  ragg::agg_tiff(
    paste0(stem, ".tiff"),
    width = width_in,
    height = height_in,
    units = "in",
    res = dpi,
    compression = "lzw",
    background = "white"
  )
  print(plot)
  invisible(grDevices::dev.off())

  ragg::agg_png(
    paste0(stem, ".png"),
    width = width_in,
    height = height_in,
    units = "in",
    res = 300,
    background = "white"
  )
  print(plot)
  invisible(grDevices::dev.off())
}

options <- parse_args(commandArgs(trailingOnly = TRUE))
agents <- c("FD", "BAL", "SP")
agent_colours <- c(FD = "#3B7C8F", BAL = "#626B75", SP = "#C77C2B")

training_data <- do.call(
  rbind,
  lapply(
    agents,
    function(agent) read_agent_progress(options$run_dir, agent, options$metric)
  )
)
training_data <- training_data[
  is.finite(training_data$timesteps) &
    is.finite(training_data$mean_episode_reward),
  ,
  drop = FALSE
]
training_data$agent <- factor(training_data$agent, levels = agents)

if (nrow(training_data) == 0) {
  stop("No finite training observations were found.", call. = FALSE)
}

last_points <- do.call(
  rbind,
  lapply(
    agents,
    function(agent) {
      rows <- training_data[training_data$agent == agent, , drop = FALSE]
      if (nrow(rows) == 0) {
        stop("No valid observations found for agent ", agent, call. = FALSE)
      }
      rows[which.max(rows$timesteps), , drop = FALSE]
    }
  )
)

x_max <- max(training_data$timesteps)
x_upper <- ceiling(x_max / 20000) * 20000
x_breaks <- seq(0, x_upper, by = 20000)

diagnostic_note <- paste0(
  "Diagnostic only: FD, BAL, and SP optimize different reward weights;\n",
  "absolute returns are not direct performance rankings."
)

plot <- ggplot(
  training_data,
  aes(
    x = .data$timesteps,
    y = .data$mean_episode_reward,
    colour = .data$agent,
    group = .data$agent
  )
) +
  geom_line(linewidth = 0.8, lineend = "round") +
  geom_point(
    data = last_points,
    size = 1.8,
    stroke = 0,
    show.legend = FALSE
  ) +
  scale_colour_manual(values = agent_colours, breaks = agents) +
  scale_x_continuous(
    breaks = x_breaks,
    limits = c(0, x_upper),
    labels = function(value) {
      ifelse(value == 0, "0", paste0(value / 1000, "K"))
    },
    expand = expansion(mult = c(0, 0.015))
  ) +
  scale_y_continuous(
    breaks = seq(200, 700, by = 100),
    limits = c(140, 800),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = "Multi-lane lane-change agents",
    x = "Training timesteps",
    y = "Mean episode reward",
    colour = NULL,
    caption = diagnostic_note
  ) +
  theme_classic(base_size = 7.2, base_family = "Arial") +
  theme(
    axis.line = element_line(linewidth = 0.35, colour = "#222222"),
    axis.ticks = element_line(linewidth = 0.35, colour = "#222222"),
    axis.text = element_text(colour = "#222222"),
    axis.title = element_text(colour = "#222222"),
    panel.grid.major = element_line(linewidth = 0.35, colour = "#E5E7EB"),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.direction = "horizontal",
    legend.justification = "center",
    legend.key.width = grid::unit(10, "pt"),
    legend.key.height = grid::unit(5, "pt"),
    legend.text = element_text(size = 6.8),
    plot.title = element_text(
      size = 9.0,
      face = "plain",
      hjust = 0.5,
      margin = margin(b = 2)
    ),
    plot.caption = element_text(
      size = 5.7,
      colour = "#555555",
      hjust = 0,
      lineheight = 1.05,
      margin = margin(t = 5)
    ),
    plot.margin = margin(4, 5, 4, 5)
  ) +
  guides(
    colour = guide_legend(
      override.aes = list(linewidth = 0.9),
      nrow = 1,
      byrow = TRUE
    )
  )

source_path <- paste0(options$out, "_source_data.csv")
dir.create(dirname(source_path), recursive = TRUE, showWarnings = FALSE)
write.csv(training_data, source_path, row.names = FALSE, na = "")
save_figure(plot, options$out)

message("Wrote figure files with stem: ", options$out)
message("Wrote source data: ", source_path)
