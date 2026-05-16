# Phase 4 Plan - RAG And Memory

## Goal

Add strategy retrieval and memory without contaminating rule truth.

## Deliverables

- external high-end game case ingestion;
- source metadata and permission fields;
- strategy retriever;
- project game memory;
- review memory;
- visibility-filtered injection;
- observability of RAG hits.

## Required Boundary

RAG answers how to play, not what the rule is.

External god-view reviews can only be used during review or moderator views, not live player contexts.

## Done Means

- Every RAG hit has source, quality grade, ruleset, phase, visibility boundary, and review status.
- No base rules are indexed as RAG truth.
- Self-play examples are lower priority than external high-quality examples unless manually promoted.
