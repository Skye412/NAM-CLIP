# TF-MoECLIP 缺陷修复执行说明（2026-11-05）

本次改动涉及三个文件，原文件备份在 `_backup_20261105/`。所有新开关的默认值都保持原行为，不加开关时训练结果与之前一致；唯一默认生效的变化是 AUPRO 的计算方式。

| 文件 | 改动 |
| :--- | :--- |
| `src/utils/metrics.py` | 新增标准定义的 AUPRO（`aupro_rank_invariant`），旧实现保留为 `Pixel-AUPRO-legacy`；新增逐类 AP、逐类 AUPRO、按区域面积三分位的 AUPRO |
| `src/train_eval.py` | 新增 `--eval_only`、`--run_tag`、`--sfr_patch_label`、`--cls_px_balance`、`--group_prior` |
| `src/models/tf_moeclip.py` | 新增 `group_prior_bias`（默认 0.0，不改变原行为） |

## 发现的两个问题

**问题一：AUPRO 实现依赖分数分布，不只依赖排序。** 原 `fast_aupro` 把 40 个阈值均匀放在分数区间上，再把保留下来的 FPR 区间拉伸到 [0, 1]。异常图越尖锐，FPR ≤ 0.3 对应的阈值越集中在最低的一两个，积分实际只覆盖很窄的低 FPR 段。纯 Python 复现的合成实验中，同一份排序只做单调变换：

| 变换 | 原实现 | 标准定义 |
| :--- | :---: | :---: |
| v（弥散图） | 63.68 | 64.87 |
| v³ | 58.64 | 64.87 |
| v⁸（尖锐图） | 28.43 | 64.87 |

所以现有各表的 P-AUPRO、Mean-7、Mean-5 都混入了"图有多尖"这个因素。旧序列 A2 的 AUPRO 11.18、本方法低于 Bayes-PFL 的 49.77 对 71.31，都需要用修正后的实现重算才能判断。

**问题二：12 锚点的 patch 级监督把细裂缝标成了背景。** `train_eval.py` 中 patch 标签规则是"patch 内病害像素占比 ≥ 20% 才算病害"。一个 patch 是 14×14 像素，1 到 2 像素宽的裂缝穿过它只占 7% 到 14%，被标为背景，随后两项背景损失把它推向正常锚点或干扰锚点。这与像素级 Focal + Dice 损失的目标相反，对应"加入 SFR-Graph 后四个场景 Crack 全部下降"的现象。

## 第 0 步：自检（1 分钟）

```bash
python src/utils/metrics.py
```

应打印四行，`quantile-v2` 一列的极差小于 0.5，`legacy` 一列明显变化。断言失败就先停下。

## 第 1 步：不重训，重算全部已有权重的指标（约 10 分钟）

```bash
python src/train_eval.py --eval_only 1 --train_on CODEBRIM --variants M0,M1,M1_PAA,M1_HET,M1_WT,M2,M3 --output_dir <official_full_v2_s111 目录> --cache_dir <特征缓存目录>
```

```bash
python src/train_eval.py --eval_only 1 --train_on S2DS --variants M0,M1,M2,M3 --output_dir <official_full_v2_s111 目录> --cache_dir <特征缓存目录>
```

- 结果写入 `metrics_<变体>_<数据集>_reeval.json` 和 `summary_tables_v2_reeval.md`，不覆盖原文件。
- 每个 JSON 新增：`Pixel-AUPRO`（修正值）、`Pixel-AUPRO-legacy`（原值，应与旧表一致，用来确认权重加载正确）、`AUPRO-By-Size`、`Per-Class` 里的 `*_AP` 和 `*_AUPRO`。
- **外部方法必须用同一个函数重算 AUPRO。** 如果外部方法的 AUPRO 也是用原 `fast_aupro` 算的，需要用它们保存的异常图调用 `aupro_rank_invariant` 重算；如果是各自代码库算的，口径本来就不一致，同样要统一。

这一步之后需要看的四个数：M0、M1、M2 的修正 AUPRO；`AUPRO_small` 对 `AUPRO_large`；`Crack_AP`；`Crack_AUPRO`。如果 M1 的 `AUPRO_small` 和 `Crack_AUPRO` 明显低于 M0，问题二得到数据确认。

## 第 2 步：候选修正，每次只改一处（每组约 15 分钟）

都在 M2 上做，只训练 Bridge512。`--run_tag` 让权重和指标文件带后缀，不覆盖默认运行。

```bash
python src/train_eval.py --train_on CODEBRIM --variants M2 --sfr_patch_label ignore --run_tag ign --output_dir <输出目录> --cache_dir <特征缓存目录>
```

```bash
python src/train_eval.py --train_on CODEBRIM --variants M2 --sfr_patch_label any --run_tag any --output_dir <输出目录> --cache_dir <特征缓存目录>
```

```bash
python src/train_eval.py --train_on CODEBRIM --variants M2 --cls_px_balance 1 --run_tag bal --output_dir <输出目录> --cache_dir <特征缓存目录>
```

```bash
python src/train_eval.py --train_on CODEBRIM --variants M2 --group_prior 1 --run_tag gp --output_dir <输出目录> --cache_dir <特征缓存目录>
```

| 开关 | 含义 | 预期 |
| :--- | :--- | :--- |
| `--sfr_patch_label ignore` | 占比在 0 到 20% 之间的 patch 不参与 patch 级监督 | 首选。去掉矛盾监督，不引入晕圈 |
| `--sfr_patch_label any` | patch 内有任一病害像素即为病害 | 对细裂缝更激进，可能降低 P-AP |
| `--cls_px_balance 1` | 逐类像素 BCE 正负样本分开平均 | 改变损失量级，风险较高 |
| `--group_prior 1` | 病害组能量加 log(7/5) | 温度较高时影响小，优先级最低 |

**判定规则**

- 用验证集判定：看各 JSON 里 `best_val_metrics` 的 `Per-Class.Crack_AP`、`Per-Class.Crack_AUPRO`、`Pixel-AUPRO` 和 Sel-6，不看测试集。
- 运行间波动约 0.2 到 0.35 个 Mean-7 点。接受标准：验证集 `Crack_AP` 或 `Pixel-AUPRO` 提升 ≥ 3 个点，且 Sel-6 下降 ≤ 0.3。
- 达不到就放弃该候选，不反复调参。
- 单项通过后再试组合，例如 `--sfr_patch_label ignore --cls_px_balance 1 --run_tag ign_bal`。

## 第 3 步：定稿配置完整重跑

patch 标签规则属于 SFR-Graph 的损失，所以消融链上所有含 SFR-Graph 的变体都要用同一个规则重训，否则 M1 → M2 的增量混入了规则变化。

```bash
python src/train_eval.py --train_on BOTH --variants M1,M2 --sfr_patch_label <选定规则> --run_tag <标签> --output_dir <输出目录> --cache_dir <特征缓存目录>
```

```bash
python src/train_eval.py --train_on CODEBRIM --variants M1_PAA,M1_HET,M1_WT --sfr_patch_label <选定规则> --run_tag <标签> --output_dir <输出目录> --cache_dir <特征缓存目录>
```

M0 不含 SFR-Graph，不需要重训，用第 1 步的重算结果。

## 尚未验证的部分

- 本机没有 numpy、scipy、torch，三个文件只通过了语法检查，没有实际运行。第 0 步自检和第 1 步的 `Pixel-AUPRO-legacy` 与旧表一致性，是这次改动的两个运行验证点。
- 问题二目前由代码逻辑和"单加 SFR-Graph 后 Crack 全降"的现象支持，修正是否有效要看第 2 步结果。
