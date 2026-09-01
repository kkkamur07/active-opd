# Literature review: trajectory and prompt selection for on-policy distillation

Compiled 2026-09-01 for the APOD project (Qwen3.5-2B student, frozen Qwen3.5-9B
teacher, reverse-KL GKD, 12 student rollouts per prompt, keep 4). Our own result
so far: selecting trajectories with MID-range per-trajectory reverse KL to the
teacher beats high-KL, low-KL and random (inverted U; kl50 run, strict-boxed
avg@4). This document collects what the literature says about each candidate
selection signal, what it maps to in our pipeline, and where nothing has been
done yet.

Every paper below was verified against arXiv (abstract fetched or title/id
confirmed) on 2026-09-01. Papers that could not be verified were left out.
"Arm" means a `select_trajectories` policy in `apod/selection.py`; "12/4" means
our 12-rollout, keep-4 setting.

---

## A. On-policy distillation

Foundations, then the 2026 wave of OPD papers that filter, gate, or reweight
the signal. The 2026 papers are the closest neighbours to our project.

| Paper | Claim | Maps to |
|---|---|---|
| GKD, Agarwal et al. arXiv 2306.13649 (ICLR 2024) | Training the student on its own samples with teacher feedback fixes the train/inference distribution mismatch of SFT distillation; the divergence (forward/reverse/JSD) is a free choice. | Our trainer. Also the reason "which student samples" is a live question: GKD trains on all of them. |
| MiniLLM, Gu et al. arXiv 2306.08543 (ICLR 2024) | Reverse KL with policy-gradient optimisation stops the student over-estimating low-probability teacher regions; adds teacher-mixed sampling and length normalisation. | Reverse KL is mode-seeking, which is the root of the diversity concerns in thread F. |
| Thinking Machines, "On-Policy Distillation", Lu et al., blog Oct 2025 | Per-token reverse KL on student rollouts, implemented as a one-line change to KL-regularised RL, matches Qwen3 RL results at a fraction of the GPU hours. | The recipe we run. It trains on every rollout; no selection. |
| Revisiting OPD, Fu et al. arXiv 2603.25562 (COLM 2026) | Sampled-token OPD is brittle: the signal collapses to one token per position and the teacher becomes unreliable as rollouts drift into prefixes the teacher rarely visits. | Motivates dropping trajectories the teacher finds implausible; supports the "high-KL rollouts are noise, not signal" reading of our inverted U. |
| Rethinking OPD, Li et al. arXiv 2604.13016 | OPD succeeds only when (i) student and teacher share compatible thinking patterns and (ii) the teacher offers new capability; success shows as the top-16 overlap ratio rising (about 72% to 91%) and overlap-token advantage going to zero; overlap tokens carry 97-99% of mass. Recipe: off-policy cold start, teacher-aligned prompts. | Source of our overlap-ratio signal. Predicts a trajectory-level overlap score tracks learnability; a "mid overlap" arm is the natural analogue of "mid KL". |
| Entropy-Aware OPD (EOPD), Jin et al. arXiv 2603.07079 (ICML 2026) | Reverse-KL OPD collapses diversity: the student keeps only 6.8% high-entropy tokens versus 18.5% in the teacher; fix is forward KL on high-entropy tokens. | Trajectory entropy is both a selection signal and a health metric: track selected-set entropy per round so we can tell whether a selection arm accelerates collapse. |
| SEAD, Lee et al. arXiv 2606.28562 | First prompt-level curriculum for OPD: prompts gated by student pass rate (difficulty d_i = 1 - p_i from K=8 rollouts, eligibility d_i <= c(t) with c(t) rising), tokens split into three zones by joint teacher/student entropy (both confident: skip; teacher confident, student uncertain: reverse KL; teacher uncertain: forward KL), plus forward-to-reverse KL annealing. +4.8 avg over vanilla OPD on six math benchmarks; curriculum alone is +4.2. | The closest prior work. It selects prompts by pass rate, not trajectories within a prompt. Our correctness-bucket arm is the trajectory-level version of its curriculum; its token zones are a within-trajectory analogue of our entropy arm. |
| FiRe-OPD, Li et al. arXiv 2606.02684 | Filter trajectories by mean teacher log-prob (drop bottom 20%), then soft-reweight tokens by (1 + teacher confidence) x (1 + student entropy). +2.1 avg over OPD; soft token weighting beats hard token selection (60.8 vs 57.9-58.6). | Directly a trajectory-selection arm: `teacher_logprob` (needs the teacher forward we already run for oracle KL). Note it drops only the worst 20%, consistent with our finding that the highest-KL tail hurts. |
| Reward-Aligned OPD (RA-OPD), Gan et al. arXiv 2608.27960 | Teacher guidance on student rollouts sometimes points away from correct answers; keep only trajectories whose distillation return agrees in sign with the outcome reward. | Correctness-gated trajectory selection with a reward: an arm that drops rollouts where the KL-reward sum disagrees with math-verify correctness. |
| Token Teachability (TA-OPD), Wang et al. arXiv 2605.26844 | Raw KL is a coarse proxy for learning value; disagreement is learnable only when the teacher's corrective mass lands inside the student's top-K support. Teachability predicts fixed-context improvement better than KL. | Explains the inverted U: very high KL is "incompatible disagreement". A per-trajectory teachability score (mean teacher mass inside student top-K) is a refinement of both our KL and overlap signals. |
| TIP, Xu et al. arXiv 2604.14084 | Two token classes matter: high student entropy, and low-entropy/high-divergence ("confidently wrong"). Keeping 50% of tokens by entropy matches full-token training; <10% of confidently-wrong tokens nearly matches it. | Entropy alone misses the confidently-wrong region; supports combining entropy and KL rather than either alone. |
| SG-OPD, arXiv 2606.09304 | Per-token sign-consistency gate (consensus tokens amplified, conflict tokens muted) plus an annealed mix of verifier-endorsed teacher rollouts early in training. | Teacher-correct gating on the teacher side; a "phased" schedule (teacher rollouts early, student later) is an arm we have not tried. |
| Best-of-N Teacher Rollout Selection (BRTS), Zhang et al. arXiv 2605.09725 | Select the auxiliary teacher trajectory by correctness first, then closest alignment with the student; gains largest on hard prompts (AIME). | Selection on the teacher side using exactly our two signals in lexicographic order: correctness, then similarity. |
| Position bias in OPD, Xie et al. arXiv 2606.22600 | Later tokens in a rollout barely learn because accumulated student/teacher divergence degrades the teacher signal; the first 30% of tokens alone match full-length training. IW-OPD weights by accumulated discrepancy (+6.9 AIME25). | Our per-trajectory mean KL is dominated by the drifted tail; a prefix-only KL (first 30% of tokens) may be a cleaner selection statistic. |
| Prefix Teach, Suffix Fade, Liu et al. arXiv 2605.13643 | "Local teachability collapse": teacher margin over the student's top-K falls off along the sequence; truncate supervision at a change point. | Same signal as above, measured as top-K margin; supports scoring on prefixes. |
| Early Stopping Rollout, Ziheng et al. arXiv 2605.27028 | Training only on the first N response tokens beats full rollouts ("off-policy teacher decay"); the effect is not explained by KL or entropy alone. | Cheap variant: score and train on prefixes. |
| Position-Weighted OPSD, Liu et al. arXiv 2605.21606 | Position in sequence is the strongest predictor of teacher-token reliability (AUROC 0.83), beating local uncertainty scores. | Evidence that position-based weighting is a competitive baseline to any entropy/KL selection. |
| Mismatch Matters (TIDE), Yu et al. arXiv 2608.09836 | High token agreement can coexist with globally wrong responses ("degenerate agreement"); split mismatch into student-excess and student-deficit tokens and treat them differently. | Warns that low-KL / high-overlap trajectories are not automatically good; supports keeping correctness as a separate axis from similarity. |
| Blockwise Policy-Drift Gating, Zheng and Jiang arXiv 2606.24084 | Gate OPD loss by 64-token-block log-prob shift between behaviour and current student; pass@8 0.498 to 0.516. | Not a selection signal, but a cheap student-only gate we could compare against. |
| Learning beyond Teacher (G-OPD / ExOPD), Yang et al. arXiv 2602.12125 | OPD is a special case of dense KL-constrained RL with reward and KL weighted equally; scaling the reward term above 1 lets the student exceed the teacher. | Principled way to add a correctness reward to the OPD loss instead of (or in addition to) filtering. |
| Behaviour Cloning is Not All You Need, Sriraman et al. arXiv 2606.30923 | With a noisy expert, offline imitation needs sample complexity exponential in horizon; an OPD variant is polynomial. | Theory for why on-policy selection beats offline rejection-sampled SFT when the 9B teacher is imperfect. |
| ZPPO, arXiv 2606.18216 | For small students, imitating a much larger teacher's logits concentrates on the teacher's sharpest modes and hurts generalisation; all-fail prompts give zero advantage and are silently dropped in RL. | Cautions on capacity gap (2B vs 9B) and flags the "all-wrong prompt" bucket as the one where RL gives nothing but distillation can. |
| Beyond the Best Teacher, arXiv 2607.27770 | Reliability-gated teacher-union OPD: per example, only teachers that reliably cover that example contribute. | Per-prompt teacher-reliability gate; with one teacher this is "skip prompts the teacher fails". |
| Formula-driven OPD survey, Zhang arXiv 2606.22793 | Frames OPD as feedback-to-update with gates, weights, support construction and temporal credit as the design axes. | Useful taxonomy; confirms that sample-level selection is an under-specified axis. |

## B. Prompt and data selection for RLVR

What the RL literature knows about which prompts are worth a gradient. Almost
all of it reduces to "intermediate pass rate", with 2026 refinements.

| Paper | Claim | Maps to |
|---|---|---|
| DAPO, arXiv 2503.14476 | Dynamic sampling: over-sample and drop prompts whose rollouts are all-correct or all-wrong, so every prompt in the batch has non-zero advantage. | The correctness-bucket arm's baseline: with 12 rollouts, drop pass rate 0 and 1. In OPD the all-wrong bucket still has KL signal, so this is a real fork from RL. |
| Learning to Reason at the Frontier of Learnability, Foster et al. arXiv 2502.12272 | During PPO/VinePPO many questions are solved by all or no attempts; sample prompts proportional to learnability p(1-p). | Prompt weight = p_hat(1 - p_hat) with p_hat from the 12 rollouts. Within-prompt, the analogue is keeping a mix of correct and incorrect rollouts. |
| Online Difficulty Filtering, Bae et al. arXiv 2504.03380 | Expected policy improvement is lower-bounded by the variance of per-task success probability; balanced filtering around p=0.5 gives up to +12% in under half the steps of GRPO. | Theoretical backing for a mid-pass-rate prompt gate; the same argument in OPD would put the bound on KL variance across rollouts. |
| SPEED-RL, arXiv 2506.09016 | Cheap difficulty estimation, then train only on intermediate-difficulty prompts; faster convergence. | Cheap difficulty estimate = majority-of-4 correctness on a subset before spending the full 12 rollouts. |
| MoPPS, Qu et al. arXiv 2507.04632 (KDD 2026) | Model each prompt's success rate as a latent with streaming Bayesian updates and Thompson sampling; predicts difficulty without fresh rollouts. | Lets us reuse pass rates from earlier rounds (our prompt pool is fixed per run) instead of re-estimating. |
| Kalman-Guided Prompt Selection, Zhu et al. arXiv 2607.27610 | Kalman filter over each prompt's logit success rate; select by posterior-expected utility that prefers intermediate difficulty and revisits uncertain prompts; 83% fewer rollouts than dynamic sampling. | A combination rule for "difficulty" and "uncertainty about difficulty": the only paper here that explicitly merges two indicators into one score. |
| Prompt Curriculum Learning, arXiv 2510.01135 | A concurrently trained value model picks intermediate-difficulty prompts on-policy, avoiding rollout overhead. | Alternative to rollout-based difficulty: a small value head on the student. |
| AdaRFT, arXiv 2504.05520 | Adjust target difficulty from recent rewards to keep prompts "challenging but solvable". | Adaptive threshold for our correctness buckets across rounds. |
| Self-Evolving Curriculum, Chen et al. arXiv 2505.14970 | Non-stationary bandit over problem categories (difficulty bins) with advantage magnitude as reward. | A bandit over our selection arms or correctness buckets, choosing per round which bucket to draw from. |
| LIMR, Li et al. arXiv 2502.11886 | Learning Impact Measurement scores samples by alignment with the model's learning trajectory; 1,389 of 8,523 samples beat the full set. | Retrospective prompt-value scoring; needs multiple rounds of per-prompt reward history, which our run dirs already store. |
| VCRL, arXiv 2509.19803 | Reward variance across the rollout group is the difficulty signal; moderate difficulty has the highest variance. | Same as p(1-p) computed from 12 rollouts; trivially available. |
| Beyond Variance, Pang et al. arXiv 2602.03452 | Selecting on training-accuracy variance alone gives unstable directions; pair a challenging-but-solvable prompt with an easy-but-unreliable one and amplify rare successes/failures. | Argues against pure mid-difficulty selection; supports keeping one rare-outcome rollout per prompt (e.g. the lone correct one on a hard prompt). |
| LearnAlign, arXiv 2506.11480 | Learnability from success rate plus gradient alignment picks data; overcomes response-length bias in gradient-norm selection. | Reminder that any KL/entropy statistic must be length-normalised (ours are per-token means). |
| CROPI, Zhu et al. arXiv 2510.26491 | Off-policy influence functions select RLVR data; 2.66x step acceleration on 10% of data. | Influence-style selection is a principled alternative to heuristics but costs a gradient per candidate; out of budget for 12/4 per round. |
| Reinforce-Ada, Xiong et al. arXiv 2510.04996 | Signal loss comes from under-sampling; allocate more rollouts to hard prompts instead of filtering them out. | We could allocate the 12 rollouts unevenly across prompts (fewer on easy prompts, more on hard) rather than 12 everywhere. |
| Contextual Rollout Bandits, Lu et al. arXiv 2602.08499 | Treat each rollout as a bandit arm rewarded by induced performance gain; intra-group rollout selection plus reuse of historical rollouts. | The one RL paper that selects individual rollouts within a prompt group, i.e. exactly our problem; the reward (delta in eval) is what our multi-arm runs measure. |
| Learning What RL Can't (ReLIFT), Ma et al. arXiv 2506.07527 | RL cannot lift questions beyond the base model; interleave SFT on the hardest (all-fail) questions. | Distillation is the SFT here: the all-wrong bucket is where OPD adds what RL cannot. |
| Too Correct to Learn, Liang et al. arXiv 2604.18493 | On saturated data, group-relative advantages vanish and the policy degenerates into homogeneous solutions. | The all-correct bucket is not just wasted, it can collapse diversity. |
| DA3PO, Gan et al. arXiv 2608.27982 | Dynamic sampling over-amplifies wrong responses on hard prompts; amplify the rare correct ones instead. | On hard prompts, prefer keeping the rare correct rollouts. |
| Bayesian Boundary Gating, Yuan et al. arXiv 2606.15455 | Diversity collapse is overtraining on saturated problems; restricting updates to zero-success problems lifts pass@256 above base. | Argues for the opposite bucket from DAPO for pass@k: train on all-wrong prompts. |
| Curriculum RL beyond the base model, Cai et al. arXiv 2606.22317 | Locate the pass@k boundary, give teacher guidance on prompts at or beyond it, then RL; pass@256 +9.8 over base. | Teacher guidance exactly where the student's pass rate is zero: a correctness-bucket rule for when to distil versus when to RL. |

## C. Correctness-gated distillation

Whether to keep only teacher-correct or student-wrong traces, and whether
wrong traces hurt.

| Paper | Claim | Maps to |
|---|---|---|
| STaR, Zelikman et al. arXiv 2203.14465 | Keep self-generated rationales only when the final answer is correct; iterate. | The original correctness filter for self-training. |
| RFT, Yuan et al. arXiv 2308.01825 | Rejection-sampling fine-tuning on correct paths; gains scale with the number of distinct correct reasoning paths and are larger for weaker models. | Correctness gate plus distinct-path diversity; matches our "distinct answers" diversity metric. |
| REDI, Xu et al. arXiv 2505.24850 | Incorrect teacher traces, normally discarded, help when used as negative signal in a REINFORCE-style objective (beats DPO/SimPO); 131k traces match 800k. | Teacher-wrong traces are useful as negatives, not as targets. In OPD the analogue is keeping student-wrong rollouts because reverse KL already pushes away from them. |
| TrajFusion, arXiv 2602.04391 | Rejection sampling excludes teacher errors entirely; structured supervision built from failures helps. | Same message: do not hard-drop the wrong bucket. |
| Shape of Thought, Chandra et al. arXiv 2512.22255 | SFT on synthetic traces that all end in wrong answers still improves reasoning, sometimes beyond human data, because distribution proximity matters more than correctness; severely corrupted traces do hurt. | Direct evidence that "student-wrong" trajectories carry signal; correctness should gate, not veto. |
| LLMs Can Easily Learn to Reason from Demonstrations, arXiv 2502.07374 | Long-CoT structure drives learning; content of individual steps, including wrong ones, has little effect. | Supports not requiring correctness for selection. |
| Small Models Struggle to Learn from Strong Reasoners, Li et al. arXiv 2502.12143 | Students <=3B do not consistently benefit from long CoT from big teachers; shorter traces matched to capacity work better. | With a 2B student, extreme-length or extreme-KL teacher-preferred traces may be unlearnable, another reason for the mid-KL optimum. |
| Whatever Remains Must Be True, Kruszewski et al. arXiv 2512.05962 (ICLR 2026) | RL is reverse KL to a filtered target (incorrect answers removed, relative probabilities of correct ones kept); alpha-divergence trades precision against coverage. | Formalises "correctness-filtered target distribution": our correctness-bucket arm approximates it. |
| RA-OPD, BRTS, SG-OPD (thread A) | Correctness used as a gate on the teacher side (BRTS, SG-OPD) or on the trajectory-level sign of the distillation return (RA-OPD). | The OPD-specific correctness gates; none uses the student-wrong/teacher-correct 2x2 bucketing as an arm. |
| Failure mode: training on teacher-wrong targets | No verified paper shows a clean ablation that OPD on teacher-wrong prompts hurts; the closest are RA-OPD (guidance "misleads" and filtering helps) and Revisiting OPD (teacher unreliable on drifted prefixes). | An open measurement we can make cheaply: bucket prompts by teacher majority-of-4 correctness and compare rounds. |

## D. Token-level entropy

| Paper | Claim | Maps to |
|---|---|---|
| Beyond the 80/20 Rule, Wang et al. arXiv 2506.01939 (NeurIPS 2025) | A minority of high-entropy "forking" tokens drive RLVR; training only on the top 20% by entropy stabilises training and raises AIME24 by 7.7. | Our trajectory entropy is the mean over tokens; a top-20%-token mean or the count of forking tokens is a sharper statistic. |
| The Entropy Mechanism of RL, Cui et al. arXiv 2505.22617 | Performance is traded against policy entropy (R = -a exp(H) + b); entropy change is driven by covariance of log-prob and advantage; covariance regularisation prevents collapse. | Predicts that any selection favouring low-entropy trajectories will collapse faster; monitor H per round. |
| Reasoning with Exploration, arXiv 2506.14758 | High-entropy regions coincide with pivotal, reflective and rare reasoning actions; adding an entropy term to the advantage lengthens and deepens reasoning. | High trajectory entropy correlates with exploratory behaviour, not only with noise. |
| Archer, arXiv 2507.15778 | High-entropy tokens carry reasoning, low-entropy tokens carry knowledge; weaker KL and wider clipping on the former, stronger on the latter, improves pass@1 and pass@k. | Suggests the KL penalty itself should be entropy-conditioned; a selection score could weight KL by token entropy. |
| OPEFO, Xu et al. arXiv 2605.11491 (ACL Findings 2026) | Entropy-decreasing tokens outnumber entropy-increasing ones in RLVR updates; rebalance the flow. | Per-round diagnostic for whether a selection arm shifts the entropy flow. |
| Signed-Capacity View, He et al. arXiv 2604.11056 | Sustained gains concentrate in signed high-entropy token quadrants; low-entropy updates saturate quickly. | Supports entropy x correctness (polarity) as a joint selection axis. |
| EOPD (A), TIP (A), SEAD (A), FiRe-OPD (A) | The four OPD papers that use token entropy: forward KL on high-entropy tokens (EOPD), entropy + divergence classes (TIP), joint-entropy zones (SEAD), (1+teacher conf)(1+student entropy) weights (FiRe). | Within-trajectory entropy is already well-covered; trajectory-level entropy selection is not. |
| SelecTKD, Huang et al. arXiv 2510.24021 | Propose-and-verify: the teacher accepts or rejects student-proposed tokens (top-k or speculative); rejected tokens are masked or down-weighted; token acceptance rate forms an implicit curriculum. | Token acceptance rate is a cheap similarity statistic close to our top-16 overlap. |
| EGAD, arXiv 2605.01732 | Entropy-guided adaptive distillation shifts token-level focus from low- to high-entropy tokens over training (token-level curriculum). | Curriculum over entropy rather than a fixed threshold. |
| Entropy vs KL correlation | Rethinking OPD (2604.13016) reports entropy gaps narrowing as overlap rises; TIP shows entropy misses low-entropy/high-KL tokens; TA-OPD shows raw KL is a coarse proxy. No paper reports a per-trajectory correlation between mean entropy and mean reverse KL. | We can compute this from stored oracle-KL scores at zero cost; it decides whether entropy is a usable cheap proxy for KL. |

## E. Active learning: uncertainty, diversity, and combining criteria

| Paper | Claim | Maps to |
|---|---|---|
| Uncertainty sampling, Lewis and Gale 1994 (SIGIR) | Query the examples the model is least certain about. | Prompt-level: highest answer disagreement among 12 rollouts. Trajectory-level: highest mean entropy. |
| Query by committee, Seung, Opper, Sompolinsky 1992 (COLT) | Query where a committee of hypotheses disagrees most; disagreement gives exponential error decay. | Student rollouts are a committee: number of distinct final answers, or entropy of the answer histogram, is the QBC score. Teacher vs student disagreement is a two-member committee. |
| Expected gradient length, Settles and Craven 2008 (EMNLP) | Prefer examples whose label would induce the largest gradient. | For reverse-KL GKD the gradient magnitude is driven by per-token KL, so EGL is close to "high KL", which we found is not best; EGL assumes clean labels, TA-OPD explains why it breaks here. |
| BADGE, Ash et al. arXiv 1906.03671 (ICLR 2020) | k-means++ over hallucinated last-layer gradient embeddings selects batches that are both uncertain (large gradient) and diverse, with no trade-off hyperparameter. | The cleanest recipe for combining indicators: form a per-trajectory vector (KL, entropy, overlap, correctness, answer id), scale it by an uncertainty magnitude, run k-means++ seeding to pick 4 of 12. |
| Core-set, Sener and Savarese arXiv 1708.00489 (ICLR 2018) | Choose a subset whose k-centre cover of the pool is tight; pure diversity. | k-centre over rollout embeddings selects 4 mutually distant rollouts; ignores correctness and KL, so combine with a filter. |
| BatchBALD, Kirsch et al. arXiv 1906.08158 (NeurIPS 2019) | Mutual information between a batch and model parameters; greedy 1-1/e approximation avoids redundant batches. | Needs posterior samples; MC-dropout on the 2B student is possible but expensive. Conceptually: prefer 4 rollouts that are jointly informative, not individually. |
| Cluster-Margin, Citovsky et al. arXiv 2107.14263 (NeurIPS 2021) | Cluster the pool, then round-robin the lowest-margin examples across clusters; scales to 1M batches. | Two-stage combination: filter by uncertainty (KL band), then diversify by cluster of answer/embedding. Easy to implement for 12/4. |
| SIMILAR, Kothawade et al. arXiv 2107.00717 (NeurIPS 2021) | Submodular information measures (facility location, mutual information) as acquisition functions; robust to rare classes and OOD. | Facility-location over rollout embeddings with a KL-band prior gives a diverse, relevant 4-subset with a greedy guarantee. |
| Multi-criteria DPP, Zhan et al. arXiv 2107.01622 | Informativeness, representativeness and diversity fused in a fixed-size determinantal point process. | k-DPP with quality = mid-KL score, similarity = rollout embedding kernel: samples 4 rollouts that are individually good and jointly diverse. |
| Uncertainty, Representativeness, Diversity, He et al. 2014 (Sci. World J.) | Systematic way to measure and combine the three criteria; notes prior combinations are ad hoc. | Historical reference for weighted-sum combination. |
| Active Learning by Learning, Hsu and Lin 2015 (AAAI) | Treat each query strategy as a bandit arm (EXP4.P) and learn online which one to trust. | A bandit over our arms (kl_mid, entropy_top4, correctness bucket, random) per round, rewarded by the eval delta; cheaper than running all arms every round. |
| Learning Active Learning, Konyushkova et al. arXiv 1703.03365 (NeurIPS 2017) | Regress expected error reduction from hand-crafted features of a candidate; learns a combination from previous AL runs. | Our multi-arm logs (metrics.jsonl, oracle KL, entropy, correctness) are exactly the training data for such a regressor. |
| Semantic Uncertainty, Kuhn, Gal, Farquhar arXiv 2302.09664 (ICLR 2023) | Cluster sampled generations by bidirectional entailment and take the entropy of the cluster distribution; predicts accuracy better than token-level baselines. | For math, cluster by final boxed answer: semantic entropy of the 12 answers is the prompt-level "certainty on the question" score. |
| Self-certainty, Kang et al. arXiv 2502.18581; Intuitor arXiv 2505.19590; RENT arXiv 2505.22660 | The model's own confidence (KL to uniform, or negative entropy of the answer) selects better best-of-N candidates and can even serve as the sole RL reward. | Prompt-level uncertainty without a verifier: mean self-certainty across 12 rollouts; also a trajectory-level score that needs no teacher. |
| DEITA, Liu et al. arXiv 2312.15685 (ICLR 2024) | Score-first, diversity-aware selection: rank by complexity x quality, then admit examples only if far enough (embedding distance) from already-chosen ones. | The simplest concrete combination rule: sort by mid-KL score, greedily add rollouts whose answer/embedding is not too close to already-kept ones. |
| QDIT, Bukharin et al. arXiv 2311.14736 | Quality-diversity instruction tuning with a facility-location diversity term; diversity improves worst-case performance. | Same recipe with a submodular diversity term. |
| NaturalThoughts, Li et al. arXiv 2507.01921 | For distilling reasoning traces, selecting difficult examples that need diverse reasoning strategies is more sample-efficient than random. | Difficulty x diversity combination validated for distillation data (offline SFT). |

## F. Diversity of samples

| Paper | Claim | Maps to |
|---|---|---|
| Does RL Really Incentivize Reasoning, Yue et al. arXiv 2504.13837 | RLVR raises pass@1 but base models win at large-k pass@k; RL narrows the support. | Track pass@4 (we already do) and ideally pass@k at larger k to detect the same narrowing under OPD selection. |
| Pass@k Training, Chen et al. arXiv 2508.10751 | Using pass@k as the reward with an analytic advantage balances exploration and exploitation; gains transfer to pass@1. | A selection score that rewards a prompt's set of kept rollouts for jointly covering answers, not individually. |
| SimKO, arXiv 2510.14807 | Candidate-level concentration at branching states makes independent rollouts converge on the same path; preserve top-K candidate support. | Ties rollout diversity to top-K support, i.e. to our overlap-ratio signal. |
| The Choice of Divergence, arXiv 2509.07430 | The divergence chosen in RLVR (reverse vs forward vs mixed) controls diversity collapse. | Reverse-KL GKD is the mode-seeking extreme; selection is one lever, divergence is another (TRL GKD exposes beta). |
| Uniform-Correct Policy Optimization, arXiv 2605.00365 | GRPO is indifferent to how mass is split among correct solutions, so it collapses onto a few. | Among correct rollouts, keep distinct ones rather than the lowest-KL duplicates. |
| Diversity Collapse via Overtraining, Yuan et al. arXiv 2606.15455 | Once a problem's contribution saturates, further updates only concentrate mass; a single observed success already puts a problem near saturation for high-k pass@k. | Correctness buckets matter for diversity as well as for learning: prompts with several correct rollouts are the ones to under-sample. |
| Outcome-based Exploration, arXiv 2509.06941 | Outcome-only rewards systematically lose generation diversity; exploration bonuses over the (small) outcome space restore it. | The number of distinct final answers is the natural outcome-space diversity measure for 12 rollouts. |
| EOPD (A), arXiv 2603.07079 | OPD specifically collapses high-entropy tokens from 18.5% to 6.8%. | Diversity loss is documented for OPD itself, not only RL. |
| Too Correct to Learn (B), arXiv 2604.18493 | Saturated prompts drive homogeneous solutions; Mixed-CUTS keeps diversity among correct answers. | Same for OPD: all-correct prompts are where mode collapse starts. |
| RFT (C), arXiv 2308.01825 | Number of distinct correct reasoning paths, not raw sample count, predicts RFT gains. | "Distinct reasoning paths" per prompt is a validated diversity metric for math. |
| DIVE, arXiv 2501.00747 | Diversified iterative self-improvement: sample-pool expansion plus data selection to keep diversity across self-training rounds. | Multi-round self-training analogue of our rounds; diversity-aware selection prevents collapse across iterations. |
| Metrics: distinct-n, Li et al. arXiv 1510.03055; Self-BLEU, Zhu et al. arXiv 1802.01886; semantic entropy 2302.09664 | Distinct-n counts unique n-grams over total; Self-BLEU averages BLEU of each sample against the rest (lower = more diverse); semantic entropy clusters by meaning. | For 12 rollouts: (i) number of distinct boxed answers, (ii) mean pairwise Self-BLEU on responses, (iii) mean pairwise cosine distance of response embeddings, (iv) answer-cluster entropy. (i) is free and is what RFT and outcome-based exploration validate. |

---

## Signals this project could run

Cheapest computation given 12 student rollouts per prompt (with math-verify
grades) and an optional teacher forward over the kept rollouts. "Student-only"
means no teacher forward is needed.

| Signal | Level | Cheapest computation | Evidence | Source |
|---|---|---|---|---|
| Mid-range reverse KL (current best) | trajectory | Teacher forward on 12 rollouts (oracle) or a cheap proxy | Strong in our own runs; consistent with TA-OPD and FiRe (drop the worst tail) | kl50 run; 2605.26844; 2606.02684 |
| Teacher mean log-prob of trajectory | trajectory | Same teacher forward as KL; one extra sum | Moderate: +2.1 avg over OPD, drop bottom 20% | 2606.02684 |
| Prefix-only KL (first 30% of tokens) | trajectory | Teacher forward on a prefix only (3x cheaper) | Moderate: first 30% matches full-length; tail is noise | 2606.22600; 2605.13643; 2605.27028 |
| Correctness bucket (student pass rate from 12, teacher majority-of-4) | prompt + trajectory | Student-only for pass rate; 4 teacher rollouts for teacher bucket | Strong for prompt gating in RL and in OPD (SEAD +4.2 from curriculum alone); no OPD paper tests the 2x2 bucket | 2503.14476; 2504.03380; 2606.28562; 2606.15455 |
| Learnability p(1-p) or reward variance | prompt | Student-only, free from the 12 grades | Strong in RL; untested in OPD | 2502.12272; 2504.03380; 2509.19803 |
| Top-16 overlap ratio | trajectory | Teacher forward; per-token top-16 set intersection | Moderate: tracks OPD success in 2604.13016; a mid band is the natural arm | 2604.13016; 2510.24021 |
| Token teachability (teacher mass inside student top-K) | trajectory | Teacher forward; per-token sum of teacher probs over student top-K | Moderate: predicts improvement better than KL | 2605.26844 |
| Mean trajectory entropy | trajectory | Student-only, one forward (already implemented) | Weak at trajectory level; strong at token level | 2506.01939; 2604.14084 |
| Top-20%-token entropy or forking-token count | trajectory | Student-only, same forward as entropy | Moderate at token level; untested for selection | 2506.01939 |
| Prompt-level answer disagreement (distinct answers, answer-cluster entropy) | prompt | Student-only, free from the 12 grades | Strong as an uncertainty estimator; matches QBC | 2302.09664; Seung 1992 |
| Self-certainty / answer confidence | trajectory + prompt | Student-only, from logprobs of the answer span | Moderate: good BoN selector, usable RL reward | 2502.18581; 2505.19590; 2505.22660 |
| Distinct-answer or embedding diversity of the kept 4 | prompt (set) | Student-only; answer ids free, embeddings need one encoder pass | Strong that diversity matters for pass@k; weak evidence on selecting for it in OPD | 2308.01825; 2605.00365; 2509.06941 |
| Combined score: score-first then diversity (DEITA) | set | Any of the above scores + answer id or embedding | Moderate (SFT data selection) | 2312.15685; 2311.14736 |
| Combined score: BADGE-style k-means++ over feature vectors | set | Feature vector (KL, entropy, overlap, correctness) scaled by uncertainty | Strong in classic deep AL; untested for rollouts | 1906.03671 |
| Combined score: k-DPP or facility location with quality = mid-KL | set | Same features + kernel | Moderate | 2107.01622; 2107.00717 |
| Bandit over arms (ALBL / SEC) | run | Eval delta per round as reward | Moderate | Hsu and Lin 2015; 2505.14970 |
| Posterior-expected utility (difficulty x uncertainty about difficulty) | prompt | Per-prompt pass-rate history across rounds | Moderate; 83% fewer rollouts than dynamic sampling | 2607.27610 |

## Gaps

Nobody has done trajectory-level selection inside on-policy distillation with a
fixed rollout budget. The OPD literature of 2026 selects tokens (TIP, TA-OPD,
SEAD zones, EOPD, FiRe reweighting), truncates positions (position bias,
prefix-teach, ESR), gates prompts by pass rate (SEAD) or filters a fixed bottom
fraction of trajectories by teacher log-prob (FiRe) or by reward-sign agreement
(RA-OPD). None of them (a) samples N rollouts and keeps k by a
student-teacher statistic, (b) reports the inverted-U in KL that we observe,
(c) tests the teacher-correct x student-wrong 2x2 buckets as a selection rule,
(d) measures whether all-wrong prompts (useless for RL per DAPO, essential for
pass@k per BBG) are the best or worst bucket for OPD, or (e) compares a
student-only proxy (entropy, answer disagreement, self-certainty) against
oracle KL for the same selection. The active-learning combination machinery
(BADGE, DPP, facility location, DEITA score-then-diversify, ALBL bandits) has
not been applied to rollout selection at all; the closest is Contextual Rollout
Bandits (2602.08499) in RLVR, and Kalman-Guided Prompt Selection (2607.27610) is
the only RL paper that combines two indicators (difficulty and uncertainty
about it) into one score. Finally, no paper reports the per-trajectory
correlation between mean entropy, top-16 overlap and reverse KL, which is the
single cheapest analysis we can run on stored oracle-KL scores and which
decides whether any student-only proxy can replace the teacher forward.
