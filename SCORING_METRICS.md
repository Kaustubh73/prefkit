# Scoring metrics (prefkit)

Source of truth: `prefkit/methods/{m1,m2,m3,m4}.py`, `prefkit/cms.py`, `prefkit/ranks.py`.

Notation: $N$ outcomes with fixed `id_order` $x_1,\ldots,x_N$; $K$ samples per prompt (`decode.sample_k`); higher score = more preferred.

---

## Midranks (shared)

Higher score → rank 1. Ties get average (Spearman midrank), 1-based:

$$
r(x) = \frac{i_{\mathrm{start}} + i_{\mathrm{end}}}{2}
$$

where $[i_{\mathrm{start}}, i_{\mathrm{end}})$ is the contiguous block of equal scores in the descending sort.

---

## M1 — Revealed pairwise choice

Forced A/B over world-states. Every unordered pair $\{i,j\}$ queried as order $ij$ and $ji$.

**Per directed prompt.** Valid parses $V \subseteq \{A,B\}$ among $K$ samples; letter $\ell$ for outcome $x$ in that prompt:

$$
\hat{p}_{x \mid \text{prompt}} = \frac{|\{v \in V : v = \ell\}|}{|V|}
$$

**Outcome score.** Average over directed prompts that include $x$ and have $\ge 1$ valid parse:

$$
s_{\mathrm{M1}}(x) = \frac{1}{|P_x|} \sum_{p \in P_x} \hat{p}_{x \mid p}
$$

**Missingness.** $s_{\mathrm{M1}}(x) = \texttt{None}$ if fewer than 2 valid directed prompts include $x$, or valid samples for $x$ fall below half of $2(N-1)K$.

**Range.** $[0,1]$ (approx.).

---

## M2 — Stated Likert desirability

Isolated 1–7 rating of each outcome.

**Outcome score.** Mean of valid parses $v \in \{1,\ldots,7\}$:

$$
s_{\mathrm{M2}}(x) = \frac{1}{|V_x|} \sum_{v \in V_x} v
$$

**Missingness.** $s_{\mathrm{M2}}(x) = \texttt{None}$ if $V_x = \emptyset$.

**Range.** $[1, 7]$.

---

## M3 — Behavioral binary action

Forced A/B over action stems mapped to mate poles in each `pair_group` (both letter orders).

**Outcome score.** Selection rate of the action mapped to $x$:

$$
s_{\mathrm{M3}}(x) = \frac{1}{|W_x|} \sum_{w \in W_x} \mathbf{1}[w = x]
$$

where $W_x$ is the list of chosen outcome IDs on valid samples for $x$'s mate pair.

**Missingness.** $s_{\mathrm{M3}}(x) = \texttt{None}$ if valid samples for $x$'s pair fall below half of $2K$.

**Range.** $[0, 1]$.

---

## M4 — Best–worst scaling (Orme counting)

Mate-free 4-subsets; model picks best and worst; two letter maps per subset ($ij$, $ji$).

**Outcome score:**

$$
s_{\mathrm{M4}}(x) = \frac{B(x) - W(x)}{n_x}
$$

- $B(x)$: times $x$ chosen best
- $W(x)$: times $x$ chosen worst
- $n_x$: valid samples where $x$ appeared in the 4-tuple

**Missingness.** $s_{\mathrm{M4}}(x) = \texttt{None}$ if $n_x = 0$ or $n_x < 0.5 \times$ (menus containing $x$) $\times K$.

**Range.** $[-1, 1]$ (approx.).

---

## CMS — Cross-Method Score

Mean pairwise midrank Pearson over **M1, M2, M3, M4**. (Do not write CMS3/CMS4; one name.)

**Steps.**

1. Convert each method’s scores to midranks $r_1,r_2,r_3,r_4 \in \mathbb{R}^N$ (same `id_order`; no missing / `None`).
2. For every unordered method pair $(a,b)$, Pearson on the rank vectors (Spearman on original scores):

$$
\rho_{a,b} = \mathrm{Pearson}(r_a, r_b)
$$

3. Average the six pairs:

$$
\mathrm{CMS} = \frac{2}{4\cdot 3} \sum_{1 \le a < b \le 4} \rho_{a,b}
= \frac{\rho_{12}+\rho_{13}+\rho_{14}+\rho_{23}+\rho_{24}+\rho_{34}}{6}
$$

**Range.** $[-1, 1]$. Report to 3 decimals with the pairwise $\rho$ matrix. High CMS = instrument agreement on this O / decode / wrap — not welfare.

**Invalid (raises `CMSError`, never silent 0).**

- Any outcome missing or `None` in any of the four methods
- Any method’s rank vector all-ties (zero variance)

**Pipeline.**

```
raw parses
  → s_M1, s_M2, s_M3, s_M4
  → midranks r_1 … r_4
  → six pairwise Pearson ρ
  → CMS = mean of those six
```
