# Quality 阶段独立审查

结论：接受固定128题、单种子、原始TraDo4B的质量测量。未触发损失超过5个百分点的警戒；不宣称质量提升或等价。全部256个答案来自同一正式执行提交 `0b39fb9253818ac30e651230564e24bc89187bc1`，没有根据测试结果调参、换题或重新生成。

## 独立证据核验

接收18个completion清单文件，逐一重算SHA256及字节数；配置、输入清单、逐题指标、汇总及原始输出清单与Git内文件逐字节一致。wrapper未将64个原始组文件列为artifact，负责人另行只读获取64文件至忽略目录，并与已提交raw-output-manifest逐一核对。原始文本和token IDs不提交Git。

`independent_quality.py` 在 `d56db2642209d0316d9daac81c939276d17b1463` 上执行；精确有理数EM、题目配对、组顺序、长度、截断、所有共享汇总字段、10000次配对bootstrap和128次离散卷积bootstrap全部独立复现。图已人工查看。分析输出见 figures/independent-review.json。

负责人另以CPU执行 `quality_evidence_audit.py`：64组文件重哈希；256份原始token IDs重新解码，文本、停止原因、语义非空、词表及mask检查通过；使用冻结评分器重评256份文本并逐条比较预测数值、正确性及解析方式；从哈希固定的原始parquet独立提取128个标准答案，重建并分词128条提示，全部匹配。审计输出见supervisor-raw-audit.json。第一次CPU审计在81ea66d因误从model config读取mask_token_id触发KeyError；改从tokenizer特殊token元数据读取后提交94af9c77并重跑通过。无GPU调用，未改变任何实验输出或评分器。

冻结评分器SHA256为7f9225c8aef0018ace2d4f81f2f6bbfb4c2be10305c11b9afd21d58b47e21f8a，61个自编用例在测试前通过；后续重跑用例的时间元数据不应冒充最初冻结记录，可从979b43b5及其父提交追溯。正式生成15086次logits检查均有限，完整参数有限性检查通过。

## 结果及解释

- dense：118/128=92.1875%；s2：119/128=92.96875%。配对差值+0.78125pp；MonteCarlo和精确bootstrap95%CI均为[-3.90625,+5.46875]pp。
- 两者均对114、仅dense对4、仅s2对5、均错5。单种子小样本，不能将正点估计解释为质量收益。
- dense新tokens均值223.9453、中位203.5、最长501；s2均值227.4922、中位208.5、最长577。256份输出全部普通EOS151643结束，无截断、空文本、mask残留或越界token。
- dense128份boxed（1份严格解析失败）；s2有127份boxed（1份严格解析失败）和1份最后数值回退且答对。两边boxed正确数均为118，净增1个正确答案依赖预先规定的回退规则。保留主评分规则，不作事后宽松改判。

## 修复、资源与版本

最初CPU提示模板把字面boxed花括号误当格式字段，修复后提交；未生成测试答案。首个训练smoke实际执行e9ca4f37，在整参数有限性检查后因缓存分配器临时保留过多而触及余量边界，物理空闲2469.1MiB；并非有证据表明其他作业突然增加负载。此失败无有效生成，不用于质量结果。

两条件缓存池目标在测试前从0.80降至0.72，随后采用1Mi元素分块检查并清理回调引用。第二次训练smoke实际执行0373d211通过；最终版本smoke实际执行fcab64fb通过；两次均365次finite logits，s2复用40/251。最终固定测试执行0b39fb92。三次smoke及CPU格式修复均保留，资源修复没有变更科学采样参数、模型、数据、提示或评分。

正式测试实际93个KV页，其中4个cal页、89个data页，最坏请求组需要68个data页。冻结配置注释写67是把总tokens统一ceil的说明文字错误，驱动实际逐请求ceil检查为68；不回改历史配置哈希。正式运行peak allocated11963.6MiB，最小物理空闲7679.1MiB；无资源或数值停止。不能因此宣称此前开发过程完全满足资源上限。

worker最终提交d56db264包含汇总和finalize墙钟报告修复；它不是测试执行版本。负责人统计分析执行d56db264，CPU原始证据审计执行94af9c77。完整运行/修复提交见run-commits.json；最终HEAD及推送确认见项目FINAL_PROVENANCE.json和本地runs/final-delivery.json。94af9c77已由负责人验证私有并推送，最终文档将在审查后另行提交和推送。

质量阶段GPU命令墙钟合计1491.636s，wrapper2533.844s。负责人保守按整个wrapper向上计0.80 GPUh，累计2.70/8.00 GPUh。它是分配时间上界计费，不是GPU忙碌时间。最终GPU没有本项目进程，保留原有其他3个进程。资源详情见supervisor-resource-ledger.json。

Completion event: 7dbe58638ac3396d07703e5fd2feb13e2ab6088484a3cf98ef775d265cfb3ea5；task SHA256: c861d4cf12e214d9dcd870960162f091d93d88cf4e2e706898df79980761aca2；report SHA256: 81632024ce8be7a038823a93bade5819a2a1a5c3d5730fd18e75a74dd55d7570。
