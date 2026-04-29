# Prompt comparison (test split)

| Prompt variant | Accuracy | Macro F1 | Mean P@k (labeled) | Mean coherence |
|----------------|----------|----------|--------------------|----------------|
| zero_shot | 0.500 | 0.573 | 0.292 | 0.592 |
| few_shot | 0.357 | 0.300 | 0.292 | 0.391 |
| chain_of_thought | 0.571 | 0.457 | 0.292 | 0.667 |

Best accuracy variant: `chain_of_thought` (see `figures/confusion_matrix.png`).
Model: `phi3:mini`; retrieval k=4.

Per-variant figures: `figures/confusion_matrix_zero_shot.png`, `figures/confusion_matrix_few_shot.png`, `figures/confusion_matrix_chain_of_thought.png`

## Inference latency (offline batch)

Wall-clock time per example includes retrieval + local LLM generation (`run_turn`). Units: milliseconds.

| Prompt variant | Mean ms | p50 ms | p95 ms |
|----------------|---------|--------|--------|
| zero_shot | 6574.0 | 5840.2 | 9672.3 |
| few_shot | 5463.8 | 5237.9 | 6922.4 |
| chain_of_thought | 10609.7 | 10550.9 | 12275.6 |