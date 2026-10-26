# 文件与运行指引

本仓库按“可运行入口、原始实现、数据、权重、结果和来源”组织。原服务器目录保持原样，发布的是经过核对的副本；没有为整理仓库重新生成分子、重新训练模型或重新对接。

## 先看哪些文件

| 目的 | 从哪里开始 |
| --- | --- |
| 了解每张论文表图的数据来源 | `docs/RESULTS_TO_FILES.md` |
| 查看目录组织和服务器来源 | `docs/SOURCE_MAP.md`、`provenance/server_source_index.json` |
| 人工查看分子记录 | `data/csv/`；精确复算使用 `data/SI_records/` 中的 JSON |
| 查看原始 A 组输出 | `data/raw_outputs/study_a/` |
| 查看 GIN 训练材料 | `data/training/`：三元组、源 SMILES、50 轮曲线和原始日志 |
| 查看实际部署权重与推理代码 | `models/best_gin_rc.pth`、`proligen/gin.py` |
| 查看训练、提示和调用的原实现 | `archive/`；这些文件保留原历史路径，实际复算用 `reproduce/` |
| 查看运行检查结果 | `verification/` |

## 运行顺序

先按根目录 README 安装相应依赖。数值统计原有验证环境为 Python 3.12；GIN 另有 Python 3.10 的精确验证配置。可以使用独立虚拟环境，不需要远程服务器账号或 API 密钥来执行下面的检查。

```text
python reproduce/offline_analysis.py --out reanalysis_run_01
python reproduce/retrieve_references.py --out gin_run_cached
python reproduce/retrieve_references.py --recompute-library --out gin_run_full
python reproduce/verify_training_inputs.py --out gin_run_training_inputs
python reproduce/render_figures.py --out regenerated_figures_01
```

每次输出目录都应使用一个新名字。前三类检查分别回答：表中数字能否从逐条记录算出来；GIN/Morgan 能否重新选出原参考；完整库从 SMILES 重新编码后是否仍得到相同参考。训练输入检查进一步核对 15,461 条三元组的环数、源分子与库的一致性，以及训练日志和曲线。

所有上述命令均不训练模型、不调用付费生成接口、不运行新的 docking。可选训练命令另见 `docs/GIN_REPRODUCTION.md`，不要将它当作日常数据复算步骤。

## 怎样读验证结果

`status: passed` 表示该入口列出的检查通过。GIN 的 `all_GIN_top3_exact`、`all_Morgan_top3_exact` 对照分子身份和顺序；`saved_reference_score_max_abs_difference` 是分数的数值差；`inputs_unchanged` 表示核验过程中输入没有被改写。软件版本和输入 SHA256 一并记录。

原始训练三元组与曲线已经提供；原训练器采用未固定切分种子的随机 80/20 划分。复现论文中的选例使用已发布权重，不需要先重新训练。完整文学上下文对话和账户凭据没有加入公开包，外部 API 生成与本仓库的离线核验入口分开说明。
