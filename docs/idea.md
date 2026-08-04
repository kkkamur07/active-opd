# Active On-Policy Distillation for Mathematical Reasoning

## Core Idea

We propose **Active On-Policy Distillation for Mathematical Reasoning**, a framework that improves OPD by selecting which student-generated rollouts are most useful for training.

In standard OPD, the student generates rollouts, and the teacher provides next-token distributions on the student-visited states. The student is then trained to match the teacher distribution on these on-policy states.

Our idea is simple:

> **Not every student rollout is equally useful for OPD.**

Instead of applying the OPD loss uniformly to all student rollouts, we actively select rollouts that are informative, verifiable, and learnable.

---

## Motivation

On-policy distillation is useful because it trains the student on its own generated states rather than only on teacher-generated solutions. This reduces the mismatch between training and inference.

However, student rollouts have different learning value. In mathematical reasoning, some rollouts are already correct, some are wrong but close to a correct reasoning path, and some are completely off-track. Treating all of them equally may make OPD less efficient.

We want Active OPD to learn faster than standard OPD and ideally reach better final performance.

---

## Method

At each training round:

1. Sample math problems from the training pool.
2. Let the student generate \(K\) solution rollouts for each problem.
3. Use a verifier to check the final answer of each rollout.
4. Compute an acquisition score for each rollout.
5. Select useful student rollouts for OPD training.
6. Apply the standard OPD loss on the selected rollouts.

The verification gate is separate from the distillation loss: exact answer
matching decides which student rollouts enter the Active OPD pool, but it is
not itself a reward or a differentiable training target.

For a selected student trajectory, the practical reverse-KL OPD objective is:

$$
\mathcal{L}_{\mathrm{OPD}} =
\sum_{(x,y_S) \in \mathcal{Q}} \sum_{t \in \mathrm{response}}
\mathrm{KL}\left(
\pi_\theta(\cdot \mid x, y_{S,<t})
\,\|\, \pi_T(\cdot \mid x, y_{S,<t})
\right)
$$

where \(\mathcal{Q}\) is the selected set of student-generated rollouts.  The
prototype estimates this reverse KL on the sampled response token, using the
veRL-compatible low-variance `k3` estimator by default; prompt and padding
positions are masked.  Top-k and full-vocabulary forward-KL modes are
optional comparison paths rather than the default trainer objective.

---

## What Makes a Good Rollout?

A good rollout should be useful for learning.

We can think of four cases:

### 1. Student correct

The student already solves the problem. This rollout may have lower priority for OPD training.

### 2. Student wrong but close to correct reasoning

This is the most useful case. The rollout reveals a student mistake, but the mistake is likely learnable from the teacher distribution.

### 3. Student wrong and far from correct reasoning

The rollout may be too off-track. The teacher distribution may be less useful because the student would need a large reasoning jump.

### 4. Teacher unreliable on this problem

Teacher reliability can be checked beforehand by letting the teacher solve the training problems and verifying the answers. Problems where the teacher fails can be filtered or downweighted.

---

## Acquisition Score

The acquisition score should estimate whether a student rollout is worth using for OPD.

Possible signals include:

- **Correctness**: whether the student final answer is correct.
- **Student uncertainty**: whether different student rollouts produce different answers.
- **Near-miss score**: whether the wrong solution is close to a valid reasoning path.
- **Diversity**: whether the selected rollouts cover different problem types or error patterns.
- **Reliability** : This should also include the likelihood of the teacher of producing that output. 

A simple first version can select verified-wrong but learnable rollouts, rather than all wrong rollouts.

---

## Data and Evaluation

Training data can start with:

- OpenThoughts

Evaluation can focus on:

- AIME 2024
- AIME 2025
- AIME 2026
- MATH-500

The main evaluation should measure learning efficiency:

```text
x-axis: gradient update steps / number of student rollouts used for training
y-axis: math reasoning accuracy
````

The goal is to show that Active OPD improves faster than standard OPD under the same training setup.

---

## Models

Possible student and teacher families include:

* Qwen3 family
* DeepSeek-R1-Distill family

A simple setup could use a smaller model as the student and a larger model from the same family as the teacher.

---

## Baselines

Some main baselines:

1. **Standard OPD**
   Apply OPD on student rollouts without active selection.

2. **Random-selection OPD**
   Apply OPD on a randomly selected subset of student rollouts.

3. **Correctness-only OPD**
   Select rollouts based only on final-answer correctness.

4. **Active OPD**
   Select rollouts using correctness plus acquisition signals such as uncertainty and near-miss score.

---

## Main Claim

**Active OPD improves mathematical reasoning distillation by selecting more useful student-generated rollouts for the same OPD objective, leading to faster learning and potentially better final performance.**

