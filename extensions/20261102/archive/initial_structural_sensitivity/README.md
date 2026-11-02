# P0 第四、第五项离线补充分析

先阅读 `结果说明与人工核查.md`，再用浏览器打开 `gallery/index.html` 查看实际分子。图片和分子图均可离线查看。

## 内容

- `inputs/`：从公开 v0.2.0 包原样复制的六份输入；来源和 SHA-256 见 `protocol.json`。
- `protocol.json` / `protocol.md`：计算前固定的本轮补充分析口径；不是原实验预注册。
- `code/analyze.py`：实际运行的数值分析代码；`retained_statistics.py` 是从公开分析脚本原样提取的统计函数。
- `code/reproduce_fresh.py`：在新目录运行的入口，避免覆盖交付结果。
- `code/report_figures.py`、`gallery.py`、`audit.py`、`write_report.py`：图、实际分子核查页、独立审计和中文报告的可执行脚本。
- `results/`：逐分子特征、全部候选、450 个原输出及检索输出位置、任务汇总、完整结构门槛重选记录及验证 JSON。
- `figures/`：三张补充图，各有 PNG 和可编辑 SVG；附全部绘图坐标和数据源哈希。
- `gallery/`：按任务与方法查看实际分子对，着色原子来自 MCS 匹配；支持查看全部 450 个计划位置。
- `originals_before.json`、`results/originals_integrity.json`：原论文和原公开包未被改动的文件指纹核查。
- `frozen_inputs_and_code.json`：本轮数值计算前固定的输入、口径和核心代码指纹。
- `file_manifest.json`：交付包各文件的最终哈希，排除自身和运行缓存。

## 运行

本次实测环境为 Python 3.10.7、RDKit 2026.03.4、NumPy 1.23.5、pandas 2.0.3、Matplotlib 3.7.2。没有安装新的完整 Python 环境；使用本机已有运行时和既有隔离依赖。所需 Python 包版本列在 requirements.txt；本轮代码不需要 PyTorch、GPU、网络或 API 密钥。

在本文件所在目录运行：

```text
python code/reproduce_fresh.py --output ../P045_rerun --figures --audit --gallery
```

输出目录必须是一个尚不存在的新目录。脚本先核对固定输入与核心代码指纹，再重新执行数值计算、绘图、独立审计、报告及分子图。

若只需重新验证交付的数据与绘图坐标，而不重新计算 MCS：

```text
python code/audit.py
```

该审计会更新自身的验证记录及两个派生核查 CSV，不改输入、论文或公开包。完整重跑会重新搜索 MCS；极少数复杂分子的搜索完成情况可能随 CPU/限时而变化，必须保留 complete 标记与上下界，不能把超时下界当作精确 MCS。交付结果保留本次实际搜索缓存，便于逐条核查。

## 范围与文件安全

本轮生成 API 调用为 0，费用为 0 元；没有新训练和新对接。结果没有写入论文，也没有推送到 GitHub。原论文及公开 v0.2.0 文件保持原样。图表整理代码在核心数值计算后加入，其最终指纹见 file_manifest.json；没有改动计算前固定的输入、方法口径或核心分析代码。

远程结果目录为 `/root/JCIM_P0_4_5_20261028`，配套压缩包为 `/root/JCIM_P0_4_5_20261028.zip`。本地旁置同步回执记录实际上传及逐文件核验结果。
