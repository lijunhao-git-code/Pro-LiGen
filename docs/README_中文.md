# 当前论文代码与数据：阅读和运行指引

当前公开版本v0.3.0同时包含原实验和新增三属性对照。先看`CURRENT_RESULTS_TO_FILES.md`，不要根据老脚本文件名推断当前正文的图表编号。

1. 只核对论文数字：运行`python reproduce/reproduce_current_paper.py --out 一个新目录`，需要NumPy/Pandas，会检查原A/B/C及新1500位置、24项检验和新对接，不调用大模型。
2. 核对最终分子是否由真实候选选出：运行`python reproduce/verify_extended_selection.py --out 一个新目录`，对原始精度的1500份候选逐次重放；不依赖CSV舍入破同分。
3. 核对GIN与Morgan如何选例：运行`python reproduce/retrieve_property_references.py --out 一个新目录`，需要RDKit、PyTorch和PyG。加`--recompute-library`会重算两套全库图嵌入，不训练。环数的原入口仍为`retrieve_references.py`。
4. 核对训练输入：运行`python reproduce/train_property_adaptations.py --out 一个新目录`；默认只核对四套三元组和一次前向损失，没有优化器更新。显式加`--run-training`才会在新目录执行原训练。
5. 重画数值结果图：在第1条加`--figures`，安装原绘图依赖后会重画历史定量图及当前图5、图6。当前PNG/PDF/SVG和逐点坐标也可直接查看。

新材料集中在`extensions/20261102/`：`data`为统计来源，`gin`为权重与训练输入/原代码，`lossless`为科学字段的原始精度导出，`scoring`为受体、逐任务评分和姿态，`archive`为原始运行及补充分析。完整含文献原文的请求/对话和账号信息不公开；公开输入与提示模板、分子及计算记录可以追溯科学结果。

T/M/G/R是输入范例配置；离线随机最终选择是另一项分析。初始375位置与新减环500位置分开保存。1500个计划运行只有1469次合格交回，不能写成1500个不同分子。11靶点的759条对接观测包含确定性库返回，不将其复制五次。

## 投稿版补充（v0.4.0，2026-11-10）

新增材料位于`extensions/20261110/`，包括ChatDrug对照750位置/4390候选、远程实际运行代码、数据整理与分子形式映射、API提示模板与资源记录、两个对接姿态案例和MDE敏感性分析。运行`python reproduce/verify_submission_supplement.py --out 一个新目录`核对新增主要材料，无需调用API。原A–C与GIN重选入口继续使用。当前正文表3是外部方法对比，表4是原对接表；详见新增目录README中的对应表。自有代码采用MIT，第三方内容保留原许可。
