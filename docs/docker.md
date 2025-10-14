# Docker Usage for redused resourses

### Dataset Preparation

```bash
docker compose -f docker-compose.dataset.yml up --build
```

**Resource limits:**
- CPU: 2-4 cores
- RAM: 3-6 GB
- Shared memory: 2 GB
- Mode: CPU only (no GPU required for dataset prep)

### Training

```bash
docker compose -f docker-compose.training.yml up --build
```

**Resource limits:**
- CPU: 4-6 cores
- RAM: 6-8 GB
- Shared memory: 4 GB
- Batch size: 8 (reduced from 16)

## Why Docker with Limits?

Prevents laptop crashes by:
1. Limiting CPU usage
2. Capping RAM consumption
3. Preventing memory leaks
4. Isolating GPU memory allocation

## Native Run (Alternative)

If Docker issues, run natively with reduced batch size:

```bash
uv run scripts/prepare_dataset.py
uv run scripts/train_model.py
```

Already optimized: `batch_size=8` for RTX 3060 Laptop.

## Monitoring

`nvtop` - GPU metrics
`btop` - CPU metrics
`docker stats worker-pkl-training` - watch Docker resources

## Troubleshooting

**Out of Memory:**
- Reduce batch_size to 4 in `scripts/train_model.py:32`
- Reduce imgsz to 512

**CPU Overload:**
- Lower CPU limits in compose files
- Set `OMP_NUM_THREADS=2`

**Laptop Overheating:**
- Pause training periodically
- Use laptop cooling pad
- Reduce epochs or batch size
