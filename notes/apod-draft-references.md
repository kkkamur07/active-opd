# References for papers/active-on-policy-distillation.md (verified against the arXiv API on 2026-11-19 unless noted)

## User's reading list (all IDs, titles, first authors verified)
- Agarwal et al. 2024, On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (ICLR 2024). https://arxiv.org/abs/2306.13649
- Song et al. 2026, A Survey of On-Policy Distillation for Large Language Models (work in progress). https://arxiv.org/abs/2604.00626
- Lu (Thinking Machines Lab) 2025, On-Policy Distillation (blog post, not arXiv). https://thinkingmachines.ai/blog/on-policy-distillation/
- Li et al. 2026, Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe. https://arxiv.org/abs/2604.13016
- Fu et al. 2026, Rethinking On-Policy Distillation of Large Language Models II: One Training Example. https://arxiv.org/abs/2609.04172
- Zhang et al. 2026, On-Policy Distillation with Best-of-N Teacher Rollout Selection. https://arxiv.org/abs/2605.09725
- Ding et al. 2026, Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement. https://arxiv.org/abs/2608.31046
- Ma et al. 2026, MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training. https://arxiv.org/abs/2606.30406
- Heo et al. 2026, On-Policy Delta Distillation. https://arxiv.org/abs/2607.15161
- Yang et al. 2026, OPRD: On-Policy Representation Distillation. https://arxiv.org/abs/2606.06021

## Data selection in on-policy distillation (abstracts read 2026-11-15)
- Hou et al. 2026, What Matters in On-Policy Distillation? A Perspective on Data Efficiency and Data Selection. https://arxiv.org/abs/2609.05198
  (hard examples help most; even ones beyond the teacher; 8 hard examples ~ 17K set)
- Wang et al. 2026, Beyond Prompt Count: How Data Shapes Transfer in On-Policy Distillation. https://arxiv.org/abs/2609.37377
  (targeted selection does not consistently beat uniform random sampling)
- Li et al. 2026, Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of LLMs. https://arxiv.org/abs/2608.16647
  (training difficulty barely matters; problems the teacher never solves are still useful)
- Yang et al. 2026, Correcting or Continuing? What On-Policy Distillation Learns from Student Errors. https://arxiv.org/abs/2610.18011
  (training only on problems the student fails lowers accuracy in all 8 configurations)
- Zhu 2026, A Shared Learning Rate Is Not a Neutral Control in Selective On-Policy Distillation. https://arxiv.org/abs/2609.22109

## Background (IDs from papers/apod-acl/refs.bib; verified in the earlier paper scan)
- Gu et al. 2024, MiniLLM. https://arxiv.org/abs/2306.08543
- Settles 2009, Active Learning Literature Survey (UW-Madison TR 1648). https://burrsettles.com/pub/settles.activelearning.pdf
- Kung et al. 2023, Active Instruction Tuning (EMNLP 2023). https://arxiv.org/abs/2311.00288
- Mindermann et al. 2022, RHO-LOSS (ICML 2022). https://arxiv.org/abs/2206.07137
- Xia et al. 2024, Rethinking Data Selection at Scale: Random Selection is Almost All You Need. https://arxiv.org/abs/2410.09335
- Yang et al. 2025, Qwen3 Technical Report. https://arxiv.org/abs/2505.09388
- Hendrycks et al. 2021, MATH. https://arxiv.org/abs/2103.03874 ; Lightman et al. 2023, Let's Verify Step by Step (MATH-500 split). https://arxiv.org/abs/2305.20050
- Guha et al. 2025, OpenThoughts. https://arxiv.org/abs/2506.04178 ; dataset: https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd
- Qwen3.5-2B / Qwen3.5-9B model cards: https://huggingface.co/Qwen/Qwen3.5-2B , https://huggingface.co/Qwen/Qwen3.5-9B
- von Werra et al., TRL. https://github.com/huggingface/trl ; Kwon et al. 2023, vLLM. https://arxiv.org/abs/2309.06180 ; Math-Verify: https://github.com/huggingface/Math-Verify
