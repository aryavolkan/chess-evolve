# Checkpoints

Compressed genome snapshots from training runs.

## Naming convention

```
{backend}/hof_{date}_{count}g_{bench_win_pct}.json.gz
```

- **backend**: `neat` or `fixed` (network topology)
- **date**: snapshot date (YYYY-MM-DD)
- **count**: number of genomes (white + black combined)
- **bench_win_pct**: avg win rate vs fixed random benchmark

## Usage

```bash
# Decompress to use as seed
gunzip -k checkpoints/neat/hof_2025-03-10_100g_10pct.json.gz -c > neat_best_genomes.json

# Train seeded from checkpoint
python train_wandb.py --config '{"seed_genome_path": "neat_best_genomes.json"}'
```

## Format

Each `.json.gz` file contains:
- `white_hof`: array of white Hall-of-Fame NEAT genomes
- `black_hof`: array of black Hall-of-Fame NEAT genomes
- `bench_avg_win_rate`: benchmark performance at time of snapshot
