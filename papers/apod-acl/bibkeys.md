# Bibliography keys for the ACL paper (refs.bib)

Built 2026-11-12. `refs.bib` has 73 entries: 65 arXiv papers and 8 web, software, model or dataset entries.

**How the entries were checked**
- **arXiv entries.** Title, the full author list and the first-version date come from the arXiv API
  (`export.arxiv.org/api/query?id_list=...`; script `/tmp/bib/fetch.py`, raw output in `/tmp/bib/{a,b,c}.json`).
  Titles in `refs.bib` are the API strings, copied exactly. Author lists longer than 12 names are cut to 12 plus "others".
- **Venues.** A venue is listed only when the arXiv comment or journal_ref states it. One exception:
  `rutherford2024noregrets` takes its venue from the NeurIPS 2024 proceedings page. All other entries are cited as
  `arXiv preprint arXiv:<id>`.
- **Allowed claim.** The "Claim the paper may attribute" column is taken from the abstract, which I read this session. The
  "Do not claim" column carries caveats from `notes/apod-critique-verify.md` §4–5.
- **Compile test.** I ran `\nocite{*}` with `acl.sty` + `acl_natbib.bst` under tectonic (in `/tmp/bibtest`). All 73
  `\bibitem`s were produced, with no BibTeX warnings.

"verified" = yes means the API or page was read this session and title, first author and year match.

## A. On-policy post-training and distillation (background and closest work)

| Key | Short title | arXiv / URL | Use in the paper | Claim the paper may attribute (from the abstract) | Do not claim | Verified |
|---|---|---|---|---|---|---|
| agarwal2024gkd | GKD (Agarwal, ICLR 2024) | https://arxiv.org/abs/2306.13649 | Our objective (reverse KL on the student's own samples); TRL `GKDTrainer` | Trains the student on its self-generated sequences with teacher feedback, which addresses train–inference distribution mismatch, and allows alternative divergences | — | yes |
| gu2024minillm | MiniLLM (Gu, ICLR 2024) | https://arxiv.org/abs/2306.08543 | Reverse KL for generative LMs | Replaces forward KL with reverse KL so the student does not overestimate low-probability regions of the teacher; on-policy optimisation | — | yes |
| ko2024distillm | DistiLLM (Ko, ICML 2024) | https://arxiv.org/abs/2402.03898 | Cost of student-generated outputs | Student-generated outputs "significantly escalated computational costs"; skew-KL plus adaptive off-policy use gives up to 4.3× speedup | — | yes |
| ko2025distillm2 | DistiLLM-2 (Ko, ICML 2025) | https://arxiv.org/abs/2503.07067 | Related work: loss design by data type | Contrastive loss that raises the likelihood of teacher responses and lowers that of student responses | — | yes |
| hinton2015distilling | Distilling the Knowledge (Hinton 2015) | https://arxiv.org/abs/1503.02531 | Origin of KD | Distils an ensemble's knowledge into a single model | — | yes |
| kim2016sequence | Sequence-Level KD (Kim & Rush, EMNLP 2016) | https://arxiv.org/abs/1606.07947 | Off-policy sequence KD contrast | Sequence-level KD for NMT; student much smaller with little BLEU loss | — | yes |
| yang2025qwen3 | Qwen3 Technical Report | https://arxiv.org/abs/2505.09388 | Standard practice: strong-to-weak on-policy phase for small models | Uses knowledge from flagship models to "significantly reduce the computational resources required to build smaller-scale models" | Table 21 numbers (OPD 74.4 vs RL 67.6; 1,800 vs 17,920 GPU-h) come from the full text (LIT read HTML), not the abstract | yes |
| lu2025opd | On-Policy Distillation (Lu & Thinking Machines Lab, 2025 blog) | https://thinkingmachines.ai/blog/on-policy-distillation/ (DOI 10.64434/tml.20251026) | Standard practice: train on every rollout; per-token reverse KL | Samples student trajectories and has a teacher grade each token (per-token reverse KL), which combines on-policy relevance with dense reward | The teacher used on the page is Qwen3-8B (critique-verify L17). The BibTeX on the page has no month; the plain-text citation says Oct 2025 | yes (page read) |
| zhao2026opsd | OPSD / Self-Distilled Reasoner (Zhao 2026) | https://arxiv.org/abs/2601.18734 | Source of our training set | On-policy self-distillation: one model acts as both teacher (with privileged verified traces) and student, minimising per-token divergence on the student's rollouts | — | yes |
| li2026rethinking | Rethinking OPD (Yaxuan Li 2026) | https://arxiv.org/abs/2604.13016 | Closest diagnosis: when OPD works; teacher-aligned prompt template | Success needs compatible thinking patterns plus new teacher capability; successful OPD shows progressive alignment on a small shared token set holding 97–99% of the mass; cold start and teacher-aligned prompt selection | — | yes |
| fu2026revisiting | Revisiting OPD (Fu 2026) | https://arxiv.org/abs/2603.25562 | Related work: OPD failure modes | The sampled-token log-ratio signal is fragile on long rollouts whose prefixes drift; their fix gives +19.8% over sampled-token OPD | — | yes |
| zhao2026semiopd | When Do We Need OPD? Semi-OPD (Zhao 2026) | https://arxiv.org/abs/2610.11291 | Alignment between teacher and student governs OPD | Semi-OPD beats OPD in 14 of 17 pairs; OPD helps only when teacher and student are highly aligned (output-token overlap) | — | yes |
| piskorz2026onoff | On-Policy or Off-Policy? (Piskorz 2026) | https://arxiv.org/abs/2609.35259 | KL direction and learning rate matter | Rollout policy is not necessarily central; KL direction shapes performance; LR governs forgetting and update sparsity | — | yes |
| hong2026dynamics | Learning Dynamics of OPD (Hong 2026) | https://arxiv.org/abs/2611.08016 | Feedback loop; limited responsiveness | Limited student responsiveness despite large disagreement; degeneration when updates raise unsuccessful continuations | — | yes |
| huang2026offpolicyteacher | Off-Policy Teacher in OPD (Huang 2026) | https://arxiv.org/abs/2609.38360 | Teacher reliability on student prefixes | Trajectories are on-policy for the student but off-policy for the teacher | — | yes |
| li2026demystifying | Demystifying OPD (Yiping Li 2026) | https://arxiv.org/abs/2610.19145 | Where the teacher's advantage lives along trajectories | Controlled token flips locate the teacher's advantage; argues for schedules that learn valuable corrections while keeping distributions stable | — | yes |
| yang2026shape | Distilling the Shape of Uncertainty (Shiping Yang 2026) | https://arxiv.org/abs/2610.39020 | Entropy structure and OPD gains | OPD learns where to branch as well as how; this drives the gains | — | yes |
| jin2026eopd | Entropy-Aware OPD (Jin, ICML 2026) | https://arxiv.org/abs/2603.07079 | Reverse KL is mode-seeking; entropy | Mode-seeking reverse KL hurts diversity; accounting for teacher uncertainty helps (Pass@8 gains) | — | yes |
| chen2026proposing | Proposing Better Rollouts (Chen 2026) | https://arxiv.org/abs/2610.36477 | Rollout policy as a design choice; length | +6.91 over OPD on Qwen3-4B-Base with 41% shorter generations | — | yes |
| dong2026rlstarts | RL Starts before RL (Dong 2026) | https://arxiv.org/abs/2609.28145 | OPD as preparation for RL | Students initialised with OPD do better under later RL | — | yes |
| zhang2026fullrollouts | Are Full Rollouts Necessary for OPD? (Zhang 2026) | https://arxiv.org/abs/2605.31490 | Rollout cost; truncated horizons | TOPD matches OPD with 10% of the rollout horizon; POPD up to 3× efficiency | — | yes |
| yu2026mismatch | Mismatch Tolerance of OPD (Cong Yu 2026) | https://arxiv.org/abs/2611.11450 | vLLM rollout / trainer mismatch | OPD tolerates some training–inference mismatch; FP8 recipes keep quality | — | yes |

## B. Data, prompt and rollout selection in OPD (closest prior work to the aligned band)

| Key | Short title | arXiv / URL | Use in the paper | Claim the paper may attribute (from the abstract) | Do not claim | Verified |
|---|---|---|---|---|---|---|
| hou2026whatmatters | What Matters in OPD? (Hou 2026) | https://arxiv.org/abs/2609.05198 | Hard examples and long CoT drive OPD gains, not token entropy | 1-shot OPD works; gains come from the longer CoT of hard problems, not high token entropy; selecting only hard examples works | It is 1-shot OPD on single problems with caps of 1,024–2,048 tokens (critique-verify L16). Transfer to our 8k/16k setting is inference | yes |
| wang2026beyondprompt | Beyond Prompt Count (Jiaxuan Wang 2026) | https://arxiv.org/abs/2609.37377 | Targeted selection vs uniform random in OPD | "Targeted selection does not consistently outperform uniform random sampling"; prompt utility is relational (depends on the teacher) | Say "did not beat random (36.94 vs 37.65 ± 0.38)", not "below random" (L9). Their selector uses MC signed log-ratios, not exact KL | yes |
| li2026everycoin | Every Coin Has Two Sides (Zhaoyi Li 2026) | https://arxiv.org/abs/2608.16647 | Difficulty in OPD | "Training difficulty barely matters, and even problems the teacher never solves are useful"; same-origin vs cross-origin pairs differ | — | yes |
| yang2026correcting | Correcting or Continuing? (Shuo Yang 2026) | https://arxiv.org/abs/2610.18011 | Question level: both-wrong questions hurt | OPD restricted to problems the student fails degrades accuracy (all 8 configs, sign test p = 0.0078); +4.6 pts when not concentrated on failures | — | yes |
| zhu2026sharedlr | Shared LR Not Neutral (Zhu 2026) | https://arxiv.org/abs/2609.22109 | Scope / regime caveat: selector verdicts depend on LR | Selector–rate entanglement: selective arms swing with LR (up to 17.7 pp) while dense supervision is flat | Selection is at token positions, with LoRA on GSM8K. It is "one possible contributor", not the explanation of our regime effect (L8). First version dated 2026-08-17 | yes |
| xu2026paced | PACED (Xu 2026) | https://arxiv.org/abs/2603.11178 | ZPD weighting in distillation (closest to the aligned-band intuition) | Gradient SNR is bell-shaped over student pass rate and collapses at both extremes; weighting by p(1−p) concentrates on the "zone of proximal development" | Their axis is pass rate per problem, not per-rollout KL | yes |
| lee2026sead | SEAD (Lee 2026) | https://arxiv.org/abs/2606.28562 | Competence-dependent waste in OPD | Teacher supervision quality depends on student competence: "incoherent rollouts yield noisy gradients; already-mastered tokens yield redundant ones" | Not checked whether it uses an equal-budget random baseline | yes |
| liu2026sweetspot | Restoring the Sweet Spot (Liu 2026) | https://arxiv.org/abs/2605.27765 | KL-based (self-)distillation lacks difficulty awareness | SDPO's KL-based advantage lacks GRPO's implicit focus on intermediate difficulty; weighting by √(p(1−p)) helps | It is self-distillation (SDPO), not a separate teacher | yes |

## C. Learnability / zone-of-proximal-development curricula (motivation for the aligned band)

| Key | Short title | arXiv / URL | Use in the paper | Claim the paper may attribute (from the abstract) | Do not claim | Verified |
|---|---|---|---|---|---|---|
| tzannetos2023proximal | ProCuRL (Tzannetos, TMLR 2023) | https://arxiv.org/abs/2304.12877 | ZPD principle in RL curricula | Inspired by ZPD: "learning progress is maximized when picking tasks that are neither too hard nor too easy for the learner" | — | yes |
| rutherford2024noregrets | No Regrets / SFL (Rutherford, NeurIPS 2024) | https://arxiv.org/abs/2408.15099 (proceedings: https://proceedings.neurips.cc/paper_files/paper/2024/hash/1d0ed12c3fda52f2c241a0cebcf739a6-Abstract-Conference.html) | Origin of learnability sampling | Training directly on scenarios the agent "can sometimes solve, but not always" (high learnability) beats UED methods | The venue was confirmed from a search result showing the NeurIPS proceedings URL; the arXiv comment is empty | yes |
| foster2025learnability | Frontier of Learnability (Foster 2025) | https://arxiv.org/abs/2502.12272 | Learnability sampling for LLM RL | Many questions are solved by all or by no attempts; prioritising high variance of success boosts training | The v6 API title has no "LILO:" prefix (earlier PDFs do). Use the API title | yes |
| bae2025difficulty | Online Difficulty Filtering (Bae 2025) | https://arxiv.org/abs/2504.03380 | Intermediate difficulty, theory | Expected policy improvement is lower-bounded by the variance of task success; intermediate difficulty maximises learning efficiency | arXiv comment: "To appear in EACL 2026". Cited as arXiv | yes |
| chen2025sec | Self-Evolving Curriculum (Xiaoyin Chen 2025) | https://arxiv.org/abs/2505.14970 | Adaptive curricula via learning gain | Bandit curriculum that uses absolute advantage as a proxy for immediate learning gain | A web snippet named a different first author; the API gives Xiaoyin Chen | yes |
| mahrooghi2026goldilocks | Goldilocks RL (Mahrooghi 2026) | https://arxiv.org/abs/2602.14868 | "Neither too easy nor too hard" in LLM RL | Selector predicts reward std per question and prioritises high variability; up to 78% fewer steps than GRPO | — | yes |
| parashar2025e2h | E2H Reasoner (Parashar 2025) | https://arxiv.org/abs/2506.06632 | Easy-to-hard curricula for small LLMs | Scheduling tasks easy to hard helps 1.5B–3B models; easy tasks must be faded out to avoid overfitting | — | yes |

## D. Selection in RL / SFT and the entropy proxy

| Key | Short title | arXiv / URL | Use in the paper | Claim the paper may attribute (from the abstract) | Do not claim | Verified |
|---|---|---|---|---|---|---|
| wang2025beyond8020 | Beyond the 80/20 Rule (Shenzhi Wang, NeurIPS 2025) | https://arxiv.org/abs/2506.01939 | Entropy as a selection signal (forking tokens) | High-entropy minority tokens act as forks; training on 20% of tokens matches full updates on Qwen3-8B and beats them on Qwen3-32B (+11.04 AIME'25) | Token-level, RLVR; not rollout selection | yes |
| yu2025dapo | DAPO (Qiying Yu 2025) | https://arxiv.org/abs/2503.14476 | Dynamic sampling in RLVR | Decoupled clip plus dynamic sampling; 50 points on AIME 2024 with Qwen2.5-32B | — | yes |
| shao2024deepseekmath | DeepSeekMath / GRPO (Shao 2024) | https://arxiv.org/abs/2402.03300 | GRPO reference (optional) | Introduces GRPO, a PPO variant | — | yes |
| deepseek2025r1 | DeepSeek-R1 (Nature 2025) | https://arxiv.org/abs/2501.12948 | Context: RL reasoning and distillation to small models (optional) | Reasoning patterns of large models "can be systematically harnessed to guide and enhance" smaller models | — | yes |
| xu2025pods | PODS (Xu 2025) | https://arxiv.org/abs/2504.13818 | Rollout down-sampling in RL (closest RL analogue) | Generate many rollouts, train on a max-variance subset; peak GRPO accuracy ≥1.7× faster | TMLR 2026 per the arXiv comment; cited as arXiv | yes |
| wang2025oneshot | 1-shot RLVR (Yiping Wang 2025) | https://arxiv.org/abs/2504.20571 | Few-example training can move MATH-500 a lot | One example lifts Qwen2.5-Math-1.5B on MATH500 from 36.0% to 73.6% | — | yes |
| xia2024random | Random Selection Is Almost All You Need (Xia 2024) | https://arxiv.org/abs/2410.09335 | Uniform-k is a strong baseline | Nearly all self-scoring selection methods fail to beat random on million-scale SFT pools; diversity matters | SFT, not OPD | yes |
| kung2023active | Active Instruction Tuning (Kung, EMNLP 2023) | https://arxiv.org/abs/2311.00288 | Uncertainty-based selection (contrast) | Prompt-uncertain tasks improve generalisation; difficult tasks give no benefit | Its uncertain-is-good finding has the opposite sign to our r3/r4 result. Do not call it agreement (L13) | yes |

## E. Length, termination and budgeted evaluation (Beat 5)

| Key | Short title | arXiv / URL | Use in the paper | Claim the paper may attribute (from the abstract) | Do not claim | Verified |
|---|---|---|---|---|---|---|
| chen2024overthinking | Overthinking (Xingyu Chen 2024) | https://arxiv.org/abs/2412.21187 | Overthinking in o1-like models | o1-like models spend excessive compute on simple problems for little benefit | — | yes |
| wu2025whenmore | When More Is Less (Wu 2025) | https://arxiv.org/abs/2502.07266 | Longer CoT is not always better | Longer CoT is not always better; an optimal length depends on task and model | — | yes |
| fatemi2025concise | Concise Reasoning via RL (Fatemi 2025) | https://arxiv.org/abs/2504.05185 | Verbosity origin | Verbosity stems from RL loss minimisation on incorrect answers | — | yes |
| yang2026eos | When EOS Tokens Disagree (Yuxiao Yang 2026) | https://arxiv.org/abs/2609.20511 | Length inflation in OPD; termination | Termination-token mismatch between student and teacher causes OPD length inflation; aligning the stopping set alone is insufficient | Whether our EOS handling covers their "functionally equivalent EOS" issue is unverified | yes |
| jiang2026crater | CRATER (Boen Jiang 2026) | https://arxiv.org/abs/2611.05860 | Unfinished = censored; finish-rate framing | Treats correct/incorrect completion as competing risks and unfinished responses as right-censored, under token budgets, on MATH | — | yes |
| desouza2026budget | Budget-Dependent Rankings (de Souza 2026) | https://arxiv.org/abs/2608.12150 | Rankings can change with the token cap | Model rankings reverse across token budgets of 64–4,096 | Budgets were only 64–4,096. That rankings reverse at 8k–32k is untested (L12) | yes |
| weng2026lostinrounding | Lost in Rounding (Weng 2026) | https://arxiv.org/abs/2611.12127 | Our teacher family (Qwen3.5 hybrid GDN); cap hits | With a bf16 recurrent state, Qwen3.5-9B cap hits rise from 118 to 223 and accuracy falls from 85.6% to 70.8% at 32K | Whether our vLLM run used an fp32 SSM state is unverified | yes |

## F. Numerical precision (Beat 6, regime)

| Key | Short title | arXiv / URL | Use in the paper | Claim the paper may attribute (from the abstract) | Do not claim | Verified |
|---|---|---|---|---|---|---|
| micikevicius2018mixed | Mixed Precision Training (ICLR 2018) | https://arxiv.org/abs/1710.03740 | fp32 master copy as standard | Half-precision training with techniques to keep accuracy, about 2× memory reduction | — | yes |
| zamirai2020bf16 | Revisiting BFloat16 Training (Zamirai 2020) | https://arxiv.org/abs/2010.06192 | Rounding cancels small updates | Stochastic rounding and Kahan summation fix accuracy loss in 16-bit-FPU training | — | yes |
| qi2025fp16 | Defeating Mismatch via FP16 (Qi 2025) | https://arxiv.org/abs/2510.26788 | Precision and RL train/infer mismatch | Uniform FP16 gives more stable optimisation and faster convergence in RL fine-tuning | — | yes |
| gao2026bf16 | bf16 Rounding Discards RL Updates (Gao 2026) | https://arxiv.org/abs/2610.41073 | Most small updates round away in bf16 | On Qwen3-1.7B only 0.4–0.5% of served weights change per step; the policy lags the optimiser by 50–150 steps | — | yes |
| singh2026fp32 | No fp32 Master Erases Updates (Singh 2026) | https://arxiv.org/abs/2610.31859 | Pure-bf16 AdamW erases updates | Without an fp32 copy, ≥79% of bf16 weight updates are erased | Report 4.3 → 49.3% only with "at a ten-times lower learning rate, where both models are undertrained" (L11). VLA domain | yes |
| nazemi2026sr | BF16–FP16 Gap with SR (Nazemi 2026) | https://arxiv.org/abs/2611.13545 | Weight sticking at low LR | Round-to-nearest bf16 causes "weight sticking" at low RL learning rates; stochastic rounding fixes it | — | yes |

## G. Evaluation rigour, seed noise, benchmarks

| Key | Short title | arXiv / URL | Use in the paper | Claim the paper may attribute (from the abstract) | Do not claim | Verified |
|---|---|---|---|---|---|---|
| hochlehnert2025sober | A Sober Look (Hochlehnert, COLM 2025) | https://arxiv.org/abs/2504.07086 | Seed and decoding sensitivity of math benchmarks | Math benchmarks are highly sensitive to seeds, decoding, prompt format and hardware; most RL gains are modest | — | yes |
| madaan2024variance | Quantifying Variance in Eval Benchmarks (Madaan 2024) | https://arxiv.org/abs/2406.10229 | Seed variance yardstick | Defines and measures benchmark variance metrics, including seed variance across initialisations | — | yes |
| miller2024errorbars | Adding Error Bars to Evals (Miller 2024) | https://arxiv.org/abs/2411.00640 | Paired comparisons and CIs | Formulas for analysing eval data and differences between two models | — | yes |
| dodge2020finetuning | Fine-Tuning Seeds (Dodge 2020) | https://arxiv.org/abs/2002.06305 | Seed-to-seed swings in fine-tuning | Different random seeds with the same hyperparameters lead to substantially different results | — | yes |
| hendrycks2021math | MATH (Hendrycks, NeurIPS 2021) | https://arxiv.org/abs/2103.03874 | MATH dataset | 12,500 competition mathematics problems with step-by-step solutions | — | yes |
| lightman2023verify | Let's Verify Step by Step (Lightman 2023) | https://arxiv.org/abs/2305.20050 | Source of the MATH-500 split | Evaluates on "a representative subset of the MATH test set" (the 500-problem split) | Cited as arXiv; the venue was not checked | yes |
| h4math500 | MATH-500 (HF dataset) | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 | Exact eval file we load (`apod/datasets/load.py:17`, split `test`) | Public, ungated; file `test.jsonl` | Author and year are the HF org and last-modified year (2025); no paper | yes (HF API) |

## H. Models, data and software

| Key | Short title | arXiv / URL | Use in the paper | Claim the paper may attribute | Do not claim | Verified |
|---|---|---|---|---|---|---|
| qwen2026qwen35 | Qwen3.5 blog (Qwen Team, Feb 2026) | https://qwen.ai/blog?id=qwen3.5 | Model family citation; the official BibTeX on both model cards | — (BibTeX copied from the model-card citation block) | The blog page itself was not fetched | yes (via card) |
| qwen2026qwen35_2b | Qwen3.5-2B model card | https://huggingface.co/Qwen/Qwen3.5-2B | Student | 2B parameters, 24 layers, hybrid Gated DeltaNet + gated attention, Apache-2.0. Card: "Qwen3.5-2B is more prone to entering thinking loops … which may prevent it from terminating generation properly" | The loops warning holds even with the recommended parameters | yes |
| qwen2026qwen35_9b | Qwen3.5-9B model card | https://huggingface.co/Qwen/Qwen3.5-9B | Frozen teacher | 9B parameters, 32 layers, same hybrid layout, Apache-2.0 | — | yes |
| guha2025openthoughts | OpenThoughts (Guha 2025) | https://arxiv.org/abs/2506.04178 | Upstream source of the training questions | Open datasets for reasoning models (OpenThoughts2-1M, OpenThoughts3) | The exact upstream subset behind the 30k OPSD file is not documented on the card. Say "derived from OpenThoughts" | yes |
| zhao2026openthoughtsdata | Openthoughts_math_30k_opsd (HF dataset) | https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd | Training questions (29,434 rows; train split; parquet; public, ungated) | Schema: source, problem, solution, messages, generated_token_count, correct, Question, COT_Reason, Answer | The card has **no license field and no description**. Author "Siyan Zhao" is **inferred** from the uploader handle `siyanzhao` matching the OPSD first author | yes (HF API); author inferred |
| vonwerra2020trl | TRL | https://github.com/huggingface/trl | `GKDTrainer` (beta = 1, lmbda = 0); we pin trl 1.10.0 (`pyproject.toml`) | — (BibTeX from the TRL README) | — | yes |
| kwon2023vllm | vLLM / PagedAttention (Kwon, SOSP 2023) | https://arxiv.org/abs/2309.06180 | Rollout and eval engine (vllm 0.26.0 pinned) | PagedAttention-based serving gives 2–4× throughput | — | yes |
| kydlicek2025mathverify | Math-Verify | https://github.com/huggingface/Math-Verify | Answer checking (`apod/verification/verify.py`; `math-verify[antlr4_13_2]>=0.5.2`) | A mathematical-expression evaluator for LLM outputs | No citation block in the repo. Author is from `pyproject.toml` (Hynek Kydlíček); year 2025 = repo creation (2025-01-17) | yes (repo read) |

## Not included (and why)

- **Not re-checked this session, so left out:** Lane-C-only OPD selection papers from `apod-lit-review.md`: FiRe-OPD 2606.02684,
  PG-OPD 2606.21994, ReNIO 2606.23104, C-OPSD 2610.25842, TA-OPD 2605.26844, TIP 2604.14084. Also Lewis & Gale 1994, BADGE,
  Spurious Rewards 2506.10947, Reasoning-or-Memorization 2507.10532, s1, L1, Ghosal, Yeo, Kalamkar, Gupta,
  Mosbach, Bouthillier, Henderson, Agarwal-2021, Hariri, Sui, and the vLLM cache docs.
  If the writer wants any of them, re-run `/tmp/bib/fetch.py <ids>`. That takes about 10 s per batch.
- **No fabricated or unverifiable entries.** Every requested item was found and verified.

## Required-list check (from the task)

GKD ✔ agarwal2024gkd · MiniLLM ✔ · DistiLLM ✔ · Qwen3 ✔ · Thinking Machines ✔ lu2025opd · Qwen3.5-2B/9B cards ✔ ·
2604.13016 ✔ · 2609.05198 ✔ · 2609.37377 ✔ · 2608.16647 ✔ · 2610.18011 ✔ · 2609.22109 ✔ · 2506.01939 ✔ · 2503.14476 ✔ ·
2410.09335 ✔ · 2412.21187 ✔ · 2609.20511 ✔ · 2611.05860 ✔ · 2610.41073 ✔ · 2610.31859 ✔ · 2611.13545 ✔ · 2504.07086 ✔ ·
2406.10229 ✔ · MATH ✔ · 2305.20050 ✔ · TRL ✔ · vLLM ✔ · Math-Verify ✔ · OpenThoughts ✔ · siyanzhao dataset ✔ · 2601.18734 ✔ ·
learnability/ZPD: 7 papers (tzannetos2023proximal, rutherford2024noregrets, foster2025learnability, bae2025difficulty,
chen2025sec, mahrooghi2026goldilocks, parashar2025e2h), plus 3 in the OPD setting (xu2026paced, lee2026sead, liu2026sweetspot) ✔

## Added in revision (2026-11-12, reviewer W10; identities checked against export.arxiv.org API this session)

| key | paper | url | used for | allowed claim (from title/abstract) |
|---|---|---|---|---|
| mindermann2022rholoss | Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt (ICML 2022) | https://arxiv.org/abs/2206.07137 | closest conceptual precedent of the hypothesis | prioritises training points that are learnable, worth learning and not yet learnt |
| lin2024rho1 | Rho-1: Not All Tokens Are What You Need | https://arxiv.org/abs/2404.07965 | reference-model-scored token selection | selects tokens scored against a reference model |
| li2025smallmodels | Small Models Struggle to Learn from Strong Reasoners | https://arxiv.org/abs/2502.12143 | learnability gap | small students struggle to learn from much stronger reasoners |
| mirzadeh2020takd | Improved Knowledge Distillation via Teacher Assistant (AAAI 2020) | https://arxiv.org/abs/1902.03393 | teacher-student capacity gap | intermediate teacher assistant narrows the gap |
| xu2025skd | Speculative Knowledge Distillation (ICLR 2025) | https://arxiv.org/abs/2410.11325 | teacher-student gap via interleaved sampling | interleaved teacher-student sampling |
