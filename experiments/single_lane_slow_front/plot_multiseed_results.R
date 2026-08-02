#!/usr/bin/env Rscript

# Publication-style summaries for the five-seed 100K single-lane experiment.
# Run from the repository root:
#   R_LIBS_USER=tmp/r-lib Rscript \
#     experiments/single_lane_slow_front/plot_multiseed_results.R

local_lib <- file.path("tmp", "r-lib")
if (dir.exists(local_lib)) {
  .libPaths(c(normalizePath(local_lib), .libPaths()))
}
local_cache <- file.path("tmp", "r-cache")
dir.create(local_cache, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(XDG_CACHE_HOME = normalizePath(local_cache))

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(scales)
  library(svglite)
  library(ragg)
})

args <- commandArgs(trailingOnly = TRUE)
outputs_root <- if (length(args) >= 1) args[[1]] else "outputs"
figure_dir <- if (length(args) >= 2) {
  args[[2]]
} else {
  file.path(
    outputs_root,
    "single_lane_slow_front_stratified_100k_multiseed",
    "figures"
  )
}
source_dir <- file.path(figure_dir, "source_data")
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)

seeds <- 0:4
agents <- c("FD", "BAL", "SP")
variants <- c("original", "no-front", "matched-speed-front", "far-front")

agent_colours <- c(
  FD = "#3D8398",
  BAL = "#7566A8",
  SP = "#C87825"
)
agent_shapes <- c(FD = 16, BAL = 15, SP = 18)
neutral_dark <- "#27313A"
neutral_mid <- "#69737C"
neutral_light <- "#EFF1F3"
signal_blue <- "#3A6780"

theme_nature_r <- function(base_size = 7.2, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.text = element_text(colour = neutral_dark),
      axis.title = element_text(colour = neutral_dark),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.4),
      legend.key.width = grid::unit(9, "pt"),
      plot.title = element_text(
        size = base_size + 1.0,
        face = "bold",
        colour = neutral_dark
      ),
      plot.subtitle = element_text(
        size = base_size - 0.1,
        colour = neutral_mid,
        margin = margin(b = 5)
      ),
      plot.caption = element_text(
        size = base_size - 0.7,
        colour = neutral_mid,
        hjust = 0,
        margin = margin(t = 6)
      ),
      plot.tag = element_text(
        size = base_size + 1.1,
        face = "bold",
        colour = neutral_dark
      ),
      panel.grid = element_blank(),
      strip.background = element_blank(),
      plot.margin = margin(4, 5, 4, 5)
    )
}

save_publication_figure <- function(
  plot,
  stem,
  width_mm,
  height_mm,
  dpi = 600
) {
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

run_dir <- function(seed) {
  file.path(
    outputs_root,
    sprintf("single_lane_slow_front_stratified_100k_seed%d", seed)
  )
}

require_file <- function(path) {
  if (!file.exists(path)) {
    stop("Required file is missing: ", path, call. = FALSE)
  }
  path
}

# ---------------------------------------------------------------------------
# Figure 1: training convergence across five independent seeds
# ---------------------------------------------------------------------------

training_raw <- bind_rows(lapply(seeds, function(seed) {
  bind_rows(lapply(agents, function(agent) {
    path <- require_file(
      file.path(run_dir(seed), "training_logs", agent, "progress.csv")
    )
    raw <- readr::read_csv(path, show_col_types = FALSE)
    tibble(
      seed = seed,
      agent = agent,
      timesteps = raw[["time/total_timesteps"]],
      episode_reward = raw[["rollout/ep_rew_mean"]]
    )
  }))
}))

interpolation_grid <- seq(2500, 97500, by = 2500)

interpolate_metric <- function(x, y, xout) {
  keep <- is.finite(x) & is.finite(y)
  x <- x[keep]
  y <- y[keep]
  ordering <- order(x)
  x <- x[ordering]
  y <- y[ordering]
  if (length(unique(x)) < 2) {
    return(rep(NA_real_, length(xout)))
  }
  stats::approx(x, y, xout = xout, rule = 1, ties = "ordered")$y
}

training_grid <- bind_rows(lapply(seeds, function(seed) {
  bind_rows(lapply(agents, function(agent) {
    data <- training_raw %>%
      filter(.data$seed == .env$seed, .data$agent == .env$agent)
    tibble(
      seed = seed,
      agent = agent,
      timesteps = interpolation_grid,
      episode_reward = interpolate_metric(
        data$timesteps,
        data$episode_reward,
        interpolation_grid
      )
    )
  }))
}))

training_summary <- training_grid %>%
  filter(is.finite(.data$episode_reward)) %>%
  group_by(.data$agent, .data$timesteps) %>%
  summarise(
    n_seeds = n(),
    median = median(.data$episode_reward),
    q1 = quantile(.data$episode_reward, 0.25),
    q3 = quantile(.data$episode_reward, 0.75),
    .groups = "drop"
  )

readr::write_csv(training_raw, file.path(source_dir, "training_progress_raw.csv"))
readr::write_csv(
  training_grid,
  file.path(source_dir, "training_progress_interpolated.csv")
)
readr::write_csv(
  training_summary,
  file.path(source_dir, "training_progress_summary.csv")
)

make_training_panel <- function(
  agent,
  y_label,
  panel_tag = NULL
) {
  individual <- training_grid %>%
    filter(.data$agent == .env$agent, is.finite(.data$episode_reward))

  aggregate <- training_summary %>%
    filter(.data$agent == .env$agent)

  plot <- ggplot() +
    geom_line(
      data = individual,
      aes(
        x = .data$timesteps / 1000,
        y = .data$episode_reward,
        group = factor(.data$seed)
      ),
      colour = agent_colours[[agent]],
      alpha = 0.22,
      linewidth = 0.32
    ) +
    geom_ribbon(
      data = aggregate,
      aes(
        x = .data$timesteps / 1000,
        ymin = .data$q1,
        ymax = .data$q3
      ),
      fill = agent_colours[[agent]],
      alpha = 0.18,
      colour = NA
    ) +
    geom_line(
      data = aggregate,
      aes(x = .data$timesteps / 1000, y = .data$median),
      colour = agent_colours[[agent]],
      linewidth = 0.8,
      lineend = "round"
    ) +
    scale_x_continuous(
      limits = c(0, 100),
      breaks = c(0, 25, 50, 75, 100),
      expand = expansion(mult = c(0, 0.01))
    ) +
    scale_y_continuous(
      limits = c(100, 720),
      breaks = seq(100, 700, by = 100),
      expand = expansion(mult = c(0, 0))
    ) +
    labs(
      x = expression(paste("Training steps (", "\u00D7", 10^3, ")")),
      y = y_label,
      title = agent,
      tag = panel_tag
    ) +
    theme_nature_r() +
    theme(
      plot.title = element_text(hjust = 0.5),
      plot.tag.position = c(0, 1),
      plot.tag = element_text(hjust = -0.6, vjust = 1.2)
    )
  plot
}

training_figure <- (
  make_training_panel(
    "FD",
    "Mean episode reward",
    panel_tag = "a"
  ) |
    make_training_panel("BAL", NULL) |
    make_training_panel("SP", NULL)
) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Double DQN training across five seeds",
    subtitle = paste(
      "Thin lines: individual seeds; thick line: median;",
      "shading: interquartile range"
    ),
    theme = theme_nature_r() +
      theme(
        plot.title = element_text(size = 9.2, hjust = 0.5),
        plot.subtitle = element_text(size = 7.0, hjust = 0.5)
      )
  )

training_stem <- file.path(figure_dir, "training_convergence_5seed")
save_publication_figure(
  training_figure,
  training_stem,
  width_mm = 183,
  height_mm = 76
)

# ---------------------------------------------------------------------------
# Figure 2: rollout counterfactual occurrence and timing
# ---------------------------------------------------------------------------

counterfactual_outcomes <- bind_rows(lapply(seeds, function(seed) {
  bind_rows(lapply(variants, function(variant) {
    path <- require_file(
      file.path(
        run_dir(seed),
        "rollout_counterfactual_front_vehicle",
        variant,
        "analysis",
        "episode_outcomes.csv"
      )
    )
    readr::read_csv(path, show_col_types = FALSE) %>%
      mutate(seed = seed, counterfactual_variant = variant)
  }))
}))

initial_actions <- bind_rows(lapply(seeds, function(seed) {
  path <- require_file(
    file.path(
      run_dir(seed),
      "counterfactual_front_vehicle",
      "counterfactual_initial_actions.csv"
    )
  )
  readr::read_csv(path, show_col_types = FALSE) %>%
    mutate(seed = seed)
}))

occurrence_summary <- counterfactual_outcomes %>%
  group_by(.data$counterfactual_variant, .data$agent_condition) %>%
  summarise(
    n = n(),
    observed = sum(.data$episode_outcome == "valid_onset"),
    terminal_failure = sum(.data$episode_outcome == "terminal_failure"),
    occurrence_rate = .data$observed / .data$n,
    .groups = "drop"
  )

timing_summary <- counterfactual_outcomes %>%
  filter(.data$episode_outcome == "valid_onset") %>%
  group_by(.data$counterfactual_variant, .data$agent_condition) %>%
  summarise(
    n_observed = n(),
    median_latency = median(.data$response_latency),
    q1_latency = quantile(.data$response_latency, 0.25),
    q3_latency = quantile(.data$response_latency, 0.75),
    .groups = "drop"
  )

initial_action_summary <- initial_actions %>%
  count(
    .data$counterfactual_variant,
    .data$agent_condition,
    .data$action,
    name = "n"
  ) %>%
  group_by(.data$counterfactual_variant, .data$agent_condition) %>%
  mutate(rate = .data$n / sum(.data$n)) %>%
  ungroup()

readr::write_csv(
  occurrence_summary,
  file.path(source_dir, "counterfactual_occurrence_summary.csv")
)
readr::write_csv(
  timing_summary,
  file.path(source_dir, "counterfactual_timing_summary.csv")
)
readr::write_csv(
  initial_action_summary,
  file.path(source_dir, "counterfactual_initial_action_summary.csv")
)

variant_levels <- c(
  "far-front",
  "matched-speed-front",
  "no-front",
  "original"
)
variant_labels <- c(
  "far front",
  "matched\nspeed",
  "no front",
  "slow front"
)

occurrence_plot_data <- occurrence_summary %>%
  mutate(
    agent_condition = factor(.data$agent_condition, levels = agents),
    counterfactual_variant = factor(
      .data$counterfactual_variant,
      levels = variant_levels
    ),
    count_label = sprintf("%d/%d", .data$observed, .data$n),
    text_colour = if_else(.data$occurrence_rate >= 0.55, "white", neutral_dark)
  )

occurrence_plot <- ggplot(
  occurrence_plot_data,
  aes(
    x = .data$agent_condition,
    y = .data$counterfactual_variant,
    fill = .data$occurrence_rate
  )
) +
  geom_tile(colour = "white", linewidth = 0.65) +
  geom_text(
    aes(label = .data$count_label, colour = .data$text_colour),
    size = 2.65,
    fontface = "bold",
    family = "Arial"
  ) +
  scale_fill_gradient(
    low = neutral_light,
    high = signal_blue,
    limits = c(0, 1),
    guide = "none"
  ) +
  scale_colour_identity() +
  scale_y_discrete(labels = variant_labels) +
  coord_fixed(ratio = 0.82, clip = "off") +
  labs(
    x = NULL,
    y = NULL,
    title = "Stable-slowdown occurrence",
    tag = "a"
  ) +
  theme_nature_r() +
  theme(
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(size = 7.2),
    axis.text.y = element_text(size = 7.0),
    plot.title = element_text(size = 7.8, hjust = 0.5),
    plot.tag.position = c(0, 1),
    plot.tag = element_text(hjust = -0.8, vjust = 1.1),
    plot.margin = margin(5, 8, 5, 8)
  )

timing_plot_data <- timing_summary %>%
  filter(.data$counterfactual_variant != "no-front") %>%
  mutate(
    agent_condition = factor(.data$agent_condition, levels = agents),
    variant_y = case_when(
      .data$counterfactual_variant == "original" ~ 3,
      .data$counterfactual_variant == "matched-speed-front" ~ 2,
      .data$counterfactual_variant == "far-front" ~ 1,
      TRUE ~ NA_real_
    ),
    agent_offset = case_when(
      .data$agent_condition == "FD" ~ 0.18,
      .data$agent_condition == "BAL" ~ 0,
      .data$agent_condition == "SP" ~ -0.18,
      TRUE ~ 0
    ),
    plot_y = .data$variant_y + .data$agent_offset,
    median_label = format(
      round(.data$median_latency, 1),
      trim = TRUE,
      nsmall = 0
    )
  )

timing_plot <- ggplot(timing_plot_data) +
  geom_segment(
    aes(
      x = .data$q1_latency,
      xend = .data$q3_latency,
      y = .data$plot_y,
      yend = .data$plot_y,
      colour = .data$agent_condition
    ),
    linewidth = 1.05,
    alpha = 0.72,
    lineend = "butt"
  ) +
  geom_point(
    aes(
      x = .data$median_latency,
      y = .data$plot_y,
      colour = .data$agent_condition,
      shape = .data$agent_condition
    ),
    size = 2.35,
    stroke = 0.25
  ) +
  geom_text(
    aes(
      x = .data$median_latency + 2.2,
      y = .data$plot_y,
      label = .data$median_label,
      colour = .data$agent_condition
    ),
    family = "Arial",
    size = 2.45,
    hjust = 0,
    show.legend = FALSE
  ) +
  annotate(
    "text",
    x = 0,
    y = 0.55,
    label = "No front: 0/180 events for every agent",
    hjust = 0,
    size = 2.45,
    family = "Arial",
    colour = neutral_mid
  ) +
  scale_colour_manual(values = agent_colours) +
  scale_shape_manual(values = agent_shapes) +
  scale_x_continuous(
    limits = c(-5, 120),
    breaks = seq(0, 120, by = 20),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(0.38, 3.5),
    breaks = c(1, 2, 3),
    labels = c("far front", "matched\nspeed", "slow front"),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    x = "First stable-slowdown latency (policy steps)",
    y = NULL,
    title = "Timing among observed events",
    tag = "b",
    colour = NULL,
    shape = NULL
  ) +
  theme_nature_r() +
  theme(
    legend.position = "top",
    legend.justification = "right",
    legend.box.margin = margin(0, 0, -2, 0),
    axis.text.y = element_text(size = 7.0),
    plot.title = element_text(size = 7.8, hjust = 0.5),
    plot.tag.position = c(0, 1),
    plot.tag = element_text(hjust = -0.8, vjust = 1.1),
    panel.grid.major.x = element_line(
      colour = "#DCE1E4",
      linewidth = 0.35
    ),
    plot.margin = margin(5, 5, 5, 8)
  )

matched_slow_counts <- initial_actions %>%
  filter(
    .data$counterfactual_variant == "matched-speed-front",
    .data$action == "SLOWER"
  ) %>%
  count(.data$agent_condition, name = "n_slow")
matched_slow_total <- sum(matched_slow_counts$n_slow)

counterfactual_figure <- occurrence_plot + timing_plot +
  plot_layout(widths = c(0.9, 1.55)) +
  plot_annotation(
    title = "Counterfactual slowdown occurrence and timing",
    subtitle = "Pooled across five independently trained seeds",
    caption = paste0(
      "n = 180 held-out exposures per agent and condition ",
      "(36 per seed × 5). Points show medians; horizontal ranges show IQR. ",
      "Initial matched-speed SLOWER: ",
      matched_slow_total,
      "/540 across all agents."
    ),
    theme = theme_nature_r() +
      theme(
        plot.title = element_text(size = 9.2, hjust = 0.5),
        plot.subtitle = element_text(size = 7.0, hjust = 0.5),
        plot.caption = element_text(size = 6.3)
      )
  )

counterfactual_stem <- file.path(
  figure_dir,
  "counterfactual_slowdown_5seed"
)
save_publication_figure(
  counterfactual_figure,
  counterfactual_stem,
  width_mm = 183,
  height_mm = 98
)

metadata <- c(
  "Core conclusion:",
  paste(
    "Across five training seeds, episode reward converges while",
    "matched-speed and no-front conditions",
    "do not trigger an immediate SLOWER action; matched-speed rollouts",
    "slow only after a delayed catch-up."
  ),
  "",
  "Figure archetype: quantitative grid",
  "Backend: R (ggplot2 + patchwork)",
  "Training figure size: 183 x 76 mm",
  "Counterfactual figure size: 183 x 98 mm",
  "Training statistics: median and IQR across five independent seeds",
  paste(
    "Counterfactual statistics: n=180 exposures per agent/condition;",
    "median and IQR among observed events"
  ),
  "Image integrity: vector-native plots; no raster source images",
  paste(
    "Reviewer risk:",
    "the shared reward axis supports consistent visual scaling, but reward",
    "weights differ and absolute rewards are not directly comparable."
  )
)
writeLines(metadata, file.path(figure_dir, "figure_contract_and_qa.txt"))

message("Wrote figures and source data to: ", normalizePath(figure_dir))
