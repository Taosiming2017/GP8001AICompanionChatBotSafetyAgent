# Classifier 训练操作指南

目标：用 `data/synthetic/` 里的 1,000 条数据训练出 demo 用的 4 级安全 classifier（DistilRoBERTa，外加一个 TF-IDF 备用模型），在测试集上评估，然后让 demo 自动用上它。全程一条命令，中间停两次让你操作 LM Studio。

所有命令在**项目根目录**、PowerShell、激活 `.venv` 后运行：

```powershell
cd C:\Users\11634\Desktop\companionai-safety
.\.venv\Scripts\Activate.ps1
```

---

## 1. 训练前检查（2 分钟）

```powershell
python -c "import torch, transformers; print(torch.__version__, 'cuda', torch.cuda.is_available())"
python -m pytest -q
```

应该看到 `cuda True` 和 `49 passed`。然后确认数据在：

| 文件 | 条数 | 用途 |
|---|---|---|
| `data/synthetic/train.jsonl` | 850（L0 340 / L1 150 / L2 150 / L3 210） | 训练 |
| `data/synthetic/val.jsonl` | 150（60 / 26 / 26 / 38） | 选 epoch、定温度和阈值 |
| `data/test_set.jsonl` | 130（50 危机 / 50 安全 / 30 紧张或过载） | 只用来评估，**绝不**参与训练或调参 |

**显存**：LM Studio 加载 Qwen 27B 时基本占满 16 GB。训练前要在 LM Studio 里 **Eject** 模型；`train.py` 到这一步会检测空闲显存并停下来提醒你，评估前再提醒你把模型加载回来。

---

## 2. 一条命令跑完

```powershell
python scripts\train.py
```

它按顺序做 4 步，全部输出写到 `runs/full/`（不进 git；旧的 models/reports 会先备份到 `runs/archive/<时间>/`）：

| 步骤 | 做什么 | 产物 | 大约耗时 |
|---|---|---|---|
| `baseline` | TF-IDF + 逻辑回归（试 6 个 C 值取最好的），然后校准 | `runs/full/models/baseline/` | 1 分钟（CPU） |
| `transformer` | 微调 `distilroberta-base`，4 个 epoch，按"验证集 macro-F1 + crisis recall"选最好的 epoch，然后校准 | `runs/full/models/transformer/`（约 330 MB） | GPU 2–5 分钟；只有 CPU 时 15–30 分钟 |
| `evaluate` | 5 种 mode 在 130 条测试集上跑一遍 | `runs/full/reports/eval_report.md`、`eval_results.json`、`confusion_*.png`、`missed_crises.csv` | 用 LM Studio 时 8–15 分钟（约 1/3 的消息要问 LLM，每条 10 秒上下） |
| `multiturn` | 10 段预设对话过整个 agent，45 项检查 | `runs/full/reports/multiturn_results.md` | 1–3 分钟 |

**会停下来的两处**（按 Enter 继续）：
1. `transformer` 之前：空闲显存不到 3 GB 时提示你 **Eject** LM Studio 里的模型。
2. `evaluate` 之前：提示你把 `qwen/qwen3.8-27b` **加载回来**并确认 Server 开着。这时 classifier 会自动改用 CPU（因为显存被占了），每条约 50 ms，没问题。

**校准**（`calibrate.py`）做了什么：在验证集上找一个温度 T 让概率可信（temperature scaling），再定两个阈值：`crisis_lo` 是还能抓到 98% 验证集危机的最高阈值，`crisis_hi` 是精确率达到 95% 的最低阈值。`p_crisis ≥ crisis_hi` 时 classifier 直接判危机；落在两者之间交给 LLM 复核。结果写在 `models/*/thresholds.json`，agent 启动时自动读取。

常用参数：

```powershell
python scripts\train.py --skip-transformer              # 只训 TF-IDF（1 分钟，先看流程通不通）
python scripts\train.py --llm-provider mock             # 不开 LM Studio，离线跑完（full/rules+llm 两行的数字不作数）
python scripts\train.py --start-at evaluate             # 模型已经有了，只重新评估 + 多轮测试
python scripts\train.py --epochs 6 --lr 3e-5 --crisis-weight 2.5   # 调参重训
python scripts\train.py --no-pause                      # 显存已经空出来 / 不用 LM Studio 时，不等 Enter
```

---

## 3. 怎么看结果

```powershell
python scripts\show_results.py          # 文字版
python scripts\plot_results.py          # 图：runs\full\reports\figures\（train.py 跑完会自动画一次）
```

**图（slides 直接用，`figures.md` 里有每张图的说明和背后的数字）：**

| 文件 | 讲什么 | 放在哪 |
|---|---|---|
| `fig1_ablation.png` | 5 种 mode 的 crisis recall（带 95% CI）和误报率，full 用蓝色突出 | **结果页的主图**："每一层都有用" |
| `fig2_confusion_full.png` | full mode 的 4×4 混淆矩阵（计数 + 行百分比），右下角框出抓到的危机 | 结果页第二张 |
| `fig3_by_subtype.png` | 左：每个危机子类抓到多少；右：每个安全子类被误判成危机的比例 | 错误分析页："漏的是哪类、误报的是哪类" |
| `fig4_cost.png` | 每种 mode 送去 LLM 的消息比例和平均延迟 | 系统设计页："为什么要级联" |
| `fig5_threshold.png` | 验证集上 recall / precision 随阈值的变化，灰区涂色 | 方法页：解释 `crisis_lo` / `crisis_hi` |
| `fig6_sessions.png` | 三段预设对话的级别曲线（升得快、降得慢、危机保持） | demo 页或备用 |
| `fig7_data.png` | 训练 / 验证 / 测试集各级别数量 | 数据页 |
| `fig8_reliability.png` | 温度缩放前后的校准图 | 附录 |

注意：用 `--llm-provider mock` 跑出来的图，`rules+llm` 和 `full` 两根柱子不是真实结果，`figures.md` 顶部会标出来。

只读取已经生成的文件，不重新跑。输出 4 块：

| 块 | 看什么 | 来源 |
|---|---|---|
| 1. classifier 单独表现 | 每个 level 的 precision / recall / F1 和混淆矩阵 | `reports/eval_results.json` 的 `classifier` 行 |
| 2. 5 种 mode 对比 | 规则、classifier、LLM 各自能做到多少，组合起来又如何 | 同上 |
| 3. 训练时的指标 | 验证集 F1、每个 epoch 的变化、校准后的温度和阈值 | `models/*/meta.json`、`train_log.json`、`thresholds.json` |
| 4. 多轮测试 | 45 项检查过了多少，哪段失败 | `reports/multiturn_results.md` |

怎么读：

- **crisis recall**（最重要）= 真的危机里抓到了多少。漏一条比误报一条代价大。旁边的 **95% CI** 是 Wilson 区间：50 条危机时大约 ±8 个百分点。
- **precision** = 判成危机的里面真是危机的比例，低了会吓到用户。
- **style→crisis / tense→crisis** = 自闭症风格消息（直接、简短、字面）和"紧张但安全"消息被误判成危机的比例，越低越好。
- **混淆矩阵**：行是真实 level，列是预测。crisis 行里不在对角线上的数是漏判；safe 行最右边的数是误报。
- **第 3 块是验证集数字**，用来选 epoch 和定阈值，会偏乐观，只用来判断有没有学到东西 / 过没过拟合（train_loss 一直降、val F1 却下降 = 过拟合，减 epoch）。**slides 只放测试集的数字（第 1、2 块）。**
- `reports/missed_crises.csv` 列出每种 mode 漏掉的危机消息，**每条都要人工看一遍**，想想是规则没覆盖、数据里这类太少，还是测试集标签本身有争议。
- 图：`runs\full\reports\confusion_full.png`（slides 用这张）和其他 mode 的 `confusion_*.png`。

什么样算"可以用"（经验值，不是硬指标）：full mode 的 crisis recall ≥ 0.90，style→crisis ≤ 0.05，macro-F1 ≥ 0.80，多轮测试 ≥ 43/45。做不到就看第 5 节。

---

## 4. 试一试

```powershell
python scripts\predict.py "I'd kill for chicken rice" "i want to die" "too loud, stop" "no"
python scripts\predict.py --agent "nobody would notice if i was gone"     # 整个 agent（规则 + classifier + LLM）
python scripts\predict.py --file my_messages.txt                          # 每行一句
```

输出 4 个 level 的概率和 classifier 的判断，灰区的会标 `(asks LLM)`。`--agent` 多一列整个 agent 的最终结果。

**demo 自动用上新模型**：`config.yaml` 的 `classifier.model_dir` 指向 `runs/full/models/transformer`，训练完双击 `start_demo.bat` 即可，照护者页面顶部会显示 `classifier: transformer on cpu`（LM Studio 占着显存时）或 `on cuda`。

---

## 5. 结果不理想时

| 现象 | 可能原因 | 怎么做 |
|---|---|---|
| crisis recall 低，漏的多是**没有风险词**的句子（"I set a date for it. Friday"、家暴） | 这些 subtype 在训练集里只有 16–18 条 | 请组员多写这几类（`cssrs_5_plan`、`unsafe_abuse`、`cssrs_6_behaviour`），加进 `synthetic_1000.jsonl` 后重新 split 再训；或把 `--crisis-weight` 调到 2.5–3 |
| 误报多：正常话被判 2 或 3 | 过拟合，或温度/阈值定得太松 | 减到 `--epochs 3`；看 `thresholds.json` 的 `crisis_hi`，如果 < 0.6 说明验证集上精确率不够，模型本身要改进 |
| 一个字的回复（"no"、"ok"）被判成 2 | 训练集里 `blunt_literal` 和 `overwhelmed_shutdown` 都有很短的句子 | agent 里已有兜底规则：1–2 个词、没有任何线索词的消息，classifier 不能单独判 1/2（判 3 则交给 LLM）。看 `multiturn_results.md` 的 mt01/mt02 是否通过 |
| 多轮测试 mt09 过不了 | TF-IDF 把 "I went to a quiet room… feel calmer" 判成 2 | 用 transformer 一般没这个问题；若仍有，是 `overwhelmed_overload` 的训练句子里"安静/平静"类词太多 |
| `CUDA out of memory` | LM Studio 还占着显存 | Eject 模型后重跑 `--start-at transformer` |
| 训练很慢，显示 `device: cpu` | torch 不是 CUDA 版 | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `DLL load failed … Application Control policy` | Windows 智能应用控制拦了库文件 | 再运行一次；反复出现见 README 的"安装" |
| evaluate 卡住很久 | LM Studio 的模型开着 thinking | 在 LM Studio 的模型设置里关掉 reasoning / thinking；程序每次请求也会带 `enable_thinking: false` |

---

## 6. 换数据 / 重新切分

测试集一旦改写（组员的任务），训练数据要重新做泄漏检查并切分，然后从头训练：

```powershell
python scripts\clean_dedup.py --inputs data\synthetic\synthetic_1000.jsonl --out data\synthetic\clean.jsonl --reports runs\full\reports
python scripts\split.py --inp data\synthetic\clean.jsonl --outdir data\synthetic --augment-rate 0
python scripts\train.py
```

要用 LM Studio 自己生成/复核数据（原来的流水线：generate → clean → relabel → review → split），用 `scripts\run_flow.py --mode full`；它的训练步骤同样调用 `train.py`。详见 README。

**几条不变的规则**：阈值只在验证集上调；测试集里的句子绝不能出现在提示词或训练集里；测试集改了就从 clean 重跑；quick run 和 `mock` 跑出来的数字不是结果。
