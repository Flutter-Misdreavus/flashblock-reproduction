# Preparation
Run bash repro/setup_env.sh then bash repro/build_flash_attn.sh under the controlled task service. Use isolated PyTorch2.4.1/cu124 and CUDA12.8; freeze actual resolved versions. No borrowed dependency symlinks. Custom FlashAttention emits compute_86 PTX; warm up driver JIT before timing. Never substitute a generic wheel.

repro/fetch_model.py pins revision d350259d64abf9219362ccb1435d68910744eea7 and verifies LFS SHA256 and small-file git blob hashes. Run long commands foreground using DSH managed background jobs, not nohup plus ampersand: the latter was reaped in prep1. Data/weights stay outside Git. See PROTOCOL.md and current task.
