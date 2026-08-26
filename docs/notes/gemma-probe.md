# Gemma 4 E2B MLX probe (local caption + text)

**Date:** 2026-08-26  
**Model:** `mlx-community/gemma-4-e2b-it-mxfp4` via ModelScope

## Status

- ModelScope download: ✅ ~3.41GB
- `mlx_vlm.generate` text-only: ✅
- `mlx_vlm.generate` with image: ✅
- Default providers: `mlx_gemma` for LLM + Vision (DeepSeek remains manual fallback)

## Quick probe

```bash
cd wenmai-assistant
chmod +x scripts/probe_gemma_mlx.sh
./scripts/probe_gemma_mlx.sh
```

Runtime wiring: `GemmaMlxServerManager` lazy-starts `mlx_vlm.server` on port **8120**. Captioner, Enricher, and `POST /ask` all call the same server — text requests omit images.

## Memory note

Do not load Gemma server alongside PaddleOCR mlx server (8111) or `bge-m3` embedding in the same session on 16GB RAM.
