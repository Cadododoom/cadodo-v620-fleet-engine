# Install (placeholder — Phase 9 fills this)

Target: a V620 owner goes from clone to first token in < 30 minutes.

1. ROCm 6.x pinned build of llama.cpp (HSA_ENABLE_SDMA=0 for gfx1031).
2. `setup.sh`: detect ROCm, locate/build llama-server, write config template.
3. Launch app → detector lists V620s → pick a model → serve.

Phase 9 delivers the full doc with measured tuning (TUNING.md).
