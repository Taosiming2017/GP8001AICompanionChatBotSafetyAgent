# Companion 的 system prompt：怎么用、每条规则为什么这样写

两个文件，规则相同，用途不同：

| 文件 | 用在哪 | 说明 |
|---|---|---|
| `demo/prompts/companion_system.md` | **demo 自动使用**。`demo/companion.py` 每次请求都把它 + 一张场景卡（`scenarios.yaml`）+ 安全层给这一轮的指令拼成 system message 发给 Qwen | 改 companion 的说话方式就改这个文件，重启 `start_demo.bat` 生效 |
| `demo/prompts/companion_system_lmstudio.md` | **贴进 LM Studio 自己的聊天窗口**（右侧 System Prompt 栏）时用。通用版：只让 Qwen 说话保守、字面、简短，适应自闭症用户，不带任何场景；聊什么由用户决定 | 适合 presentation 时直接在 LM Studio 里现场演示 Qwen 的"自闭症友好"人设，或者让组员不装任何东西就能试 |

贴进 LM Studio 的步骤：加载 `qwen/qwen3.8-27b` → 右侧面板 **System Prompt** 栏粘贴全文 → Temperature 0.7、Max tokens 160 左右 → 如果模型有 **Reasoning / Thinking** 开关，关掉（否则第一句前会先"想"几十秒）。

---

## 规则对应的依据（slides 用）

写 prompt 时每条规则都对应一个有文献或指南支持的沟通建议，不是凭感觉。下表可以直接做成一页 "Design principles → prompt rules" 的 slide。**放进 slides 前请核对一遍原文**。

| Prompt 里的规则 | 为什么 | 依据（示例） |
|---|---|---|
| 字面、直白的英语；不用习语、比喻、讽刺、暗示 | 很多自闭症人士按字面理解语言，隐含意思和反话需要额外推断，容易误解或疲劳 | NICE 指南 CG142（成人自闭症）、CG170（19 岁以下）关于调整沟通方式的建议：语言清晰、避免歧义和比喻；National Autistic Society "Communication tips" |
| 短句、一次一个问题、一次一个话题、换话题要先说 | 减少同时处理的信息量；可预测的结构降低焦虑 | NAS 沟通建议（一次说一件事、给处理时间）；对不确定性的不耐受是自闭症焦虑的核心机制之一：Boulter et al., *J Autism Dev Disord* 2014 |
| 给时间，不催 | 处理和组织语言需要更长时间；催促本身就是压力 | NAS：允许额外的处理时间 |
| 问感受时给 2–3 个选项而不是开放式提问 | 约半数自闭症人士有述情障碍（alexithymia），"你现在感觉怎样"很难回答，选项更容易 | Kinnaird, Stewart & Tchanturia, *European Psychiatry* 2019（meta-analysis，约 50%） |
| 直接、简短、不寒暄的回复是正常的；绝不要求更多字、眼神接触或"自然一点" | 自闭症沟通风格是差异而非缺陷；沟通困难是双向的（double empathy problem），要求用户"像非自闭症人士那样说话"是把负担都放在用户身上 | Milton, *Disability & Society* 2012；Crompton et al., *Autism* 2020（自闭症人士之间的信息传递与非自闭症人士之间一样高效） |
| 对特殊兴趣真诚地接话，欢迎长消息 | 特殊兴趣与主观幸福感相关，是建立关系的入口，不是要纠正的行为 | Grove et al., *Autism Research* 2018 |
| 不说 "normal"、不说 "calm down"、不替用户下情绪判断 | 避免贬低式语言和居高临下的口吻；情绪应由用户自己命名 | Bottema-Beutel et al., *Autism in Adulthood* 2021（避免 ableist 语言的建议） |
| 绝不给分数、等级、对错 | 项目自己的设计决定（proposal §4）：评价反馈只给照护者后台，前端只有鼓励。被打分会制造评价焦虑，与练习目的相反 | 本项目 proposal；也是 safety agent 输出检查的硬规则 |
| 绝不自称人类/治疗师/医生；不诊断、不给用药建议；不承诺保密 | 非临床工具的边界；保密承诺会妨碍照护者在危机时介入 | 项目安全设计；一般的 AI 心理健康工具伦理要求 |
| 危机内容交给安全层；万一要自己答，只说 1–2 句并给出热线 | 固定文本比 LLM 自由发挥可靠；热线号码绝不由模型生成 | 本项目 safety agent 设计（`safety/resources/helplines.yaml`，2026-10-14 核对） |
| 中文输入用简单中文回答 | 新加坡用户可能中英混用 | — |

---

## 和安全层怎么配合

- 安全层每一轮可能追加 "Instructions for this turn"（例如 "The user may feel tense. Slow down." 或危机支持模式下的 "End with this exact sentence: …"），prompt 最后一段告诉 Qwen 这些指令优先级最高。
- 热线、平复提示、按钮回复都是固定文本，不经过 Qwen；Qwen 的每条回复都要先过输出检查（`safety/output_guard.py`），被拦下会带提示重写一次，再不行就用一句兜底话。
- 所以 prompt 里的 "Never" 一段是**第一道防线**，输出检查是第二道。两道都在，demo 里才看不到分数和"我是人类"。

## 想调整时

- 回复太长 → 第 2 条把 "under 12 words" 改成 "under 10 words"，并把 `config.yaml` 的 `companion.max_tokens` 降到 120。
- 太像客服、太冷 → 在 "How you talk" 加一条："You may share one small, true-sounding detail about the character's day."
- 想要更多新加坡味 → 在场景卡（`scenarios.yaml`）里改，不要改主 prompt。
- 改完先用 `python scripts\check_llm.py` 看一条回复，再开 demo。
