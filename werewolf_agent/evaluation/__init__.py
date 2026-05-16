"""Evaluation and benchmark tooling.

Design doc §14: evaluation system for batch games, metrics aggregation,
model/persona/RAG strategy comparisons, leakage rate, illegal action rate,
cost/latency statistics, growth curves, and leaderboard JSON reports.

Evaluation is replayable from initial_seed + ruleset_snapshot + event_log.
Evaluation never mutates rule truth.
"""
