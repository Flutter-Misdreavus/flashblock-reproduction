# FlashBlock 最小复现

原始 TraDo4B BF16、单张共享 RTX3090、无蒸馏/训练/量化。完成机制、固定工作量效率及固定128题GSM8K质量对照。研究范围和预先固定的判定规则见 [PROTOCOL.md](PROTOCOL.md)，完整结果及局限见 [REPORT.zh.md](REPORT.zh.md)。本实验不等同于论文8B蒸馏模型、32k+上下文或A100主表复现。

## 结果与审查入口

- [数值准备与模型冒烟](experiments/flashblock-20261023-prep-r3/supervisor-review.md)
- [机制：按提示聚合、独立参考与复用计数](experiments/flashblock-20261023-mechanism/supervisor-review.md)
- [效率：60条计时、失败保留与剖析分类纠正](experiments/flashblock-20261023-efficiency/supervisor-review.md)
- [质量：冻结128题及独立评分复核](experiments/flashblock-20261023-quality/supervisor-review.md)
- [最终版本和资源记录](FINAL_PROVENANCE.json)

代码/小型脱敏指标/配置/图保存在私有仓库。权重、数据、生成原文、token IDs和原始DSH日志不提交；它们保存在远程项目的忽略目录runs/或项目外runtime/。报告中的实际运行SHA与后续修复/审查HEAD分别记录。

## 固定依赖与数据

上游[Caesarhhh/Flash_Block](https://github.com/Caesarhhh/Flash_Block)，commit a322fb80f29df0d4cc842fbc011929bc85ba8f36，来源清单[UPSTREAM.json](UPSTREAM.json)。实际环境Python3.10、torch2.4.1+cu121、triton3.0、numpy1.26.4、transformers4.52.4、FlashInfer0.2.9、liger-kernel0.5.10；CUDA toolkit12.8，自定义FlashAttention源码编译sm86/PTX。初始文档里的cu124计划不代表实际安装环境；以[环境快照](experiments/flashblock-20261023-prep-r3/environment.json)和[版本锁](experiments/flashblock-20261023-prep-r3/locks.json)为准。

模型Gen-Verse/TraDo-4B-Instruct revision d350259d64abf9219362ccb1435d68910744eea7；两片权重LFS SHA256和小文件Git blob hash均验证。数据openai/gsm8k main revision740312add88f781978c0658806c59bc2815b9866；训练/测试parquet SHA256及最早冻结的128测试ID见版本锁和formal-test-ids.json。质量提示、完整评分器哈希和当前资源配置见[quality-config.json](experiments/flashblock-20261023-quality/quality-config.json)与[input-manifest.json](experiments/flashblock-20261023-quality/input-manifest.json)。

已有服务器环境位于`/data/research/flashblock-reproduction-runtime/venv`，权重位于同一runtime的models/TraDo-4B-Instruct。环境/数据不随Git克隆提供。在新的隔离环境重建可参考[setup_env.sh](repro/setup_env.sh)、[build_flash_attn.sh](repro/build_flash_attn.sh)、[fetch_model.py](repro/fetch_model.py)和[freeze_data.py](repro/freeze_data.py)，安装后必须核对实际CUDA wheel及锁定哈希；禁止替换成通用FlashAttention wheel。

## 重算已保存的结果（不运行GPU）

使用安装numpy、matplotlib的独立Python环境。实际分析版本为Python3.10.14/numpy1.26.3/matplotlib3.9.4；脚本SHA、输入SHA、执行时commit均写入输出independent-review.json。

```bash
python repro/independent_mechanism.py \
  --input experiments/flashblock-20261023-mechanism/mechanism-per-prompt.json \
  --out /tmp/flashblock-mechanism-review
python repro/independent_efficiency.py \
  --input experiments/flashblock-20261023-efficiency/efficiency-paired-runs.json \
  --profile experiments/flashblock-20261023-efficiency/separate-profile.json \
  --out /tmp/flashblock-efficiency-review
python repro/independent_quality.py \
  --input experiments/flashblock-20261023-quality/quality-per-item.json \
  --out /tmp/flashblock-quality-review
```

独立效率脚本对5对记录穷举3125个bootstrap样本；同时保留执行端预先固定的1万次MonteCarlo区间。剖析修正按完整内核名区分`void flash::`注意力和`flashinfer::sampling::`采样，不把通用名称flash当作注意力证据。质量脚本复算精确有理数EM、1万次成对bootstrap及离散精确bootstrap区间，不修改评分规则。

## 真实执行命令与重新实验

各阶段task.json、commands-and-provenance.json、run-commits.json和supervisor-review.md记录实际命令与执行SHA。驱动入口包括`flashblock_prep_cli.py`、`mechanism_cli.py`、`efficiency_cli.py`、`quality_cli.py`；使用`--help`查看命令。质量评分器[quality_scorer.py](repro/quality_scorer.py)及61个[自编用例](repro/quality_scorer_tests.py)在测试生成前冻结。

历史执行目录已经包含完成结果，部分旧驱动会写固定run_id路径。重新运行GPU实验时，应在独立checkout中创建新的run_id、冻结配置和任务并提交，沿用输入哈希与科学参数，不能直接对历史runs/覆盖执行。按[research-orchestrator技能](skills/research-orchestrator/SKILL.md)调度和审查；仅使用指定GPU，先检查实时显存/其他作业、容量与预算。当前回放命令和路径是实验记录，不是要求立即再跑整套实验。

本次修复包括源码兼容性、缓存写入/批量buffer维度、模型生命周期与分块参数检查。所有失败均保留；曾有显存越界，不能用最终成功运行的峰值宣称整个开发过程始终满足资源约束。详见各阶段审查。
