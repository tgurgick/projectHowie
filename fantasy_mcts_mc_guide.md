
# Fantasy Draft Planner — Monte Carlo + MCTS Implementation Guide

**Goal:** Build a local Python agent that recommends draft picks round-by-round using **Monte Carlo Tree Search (MCTS)** and **Monte Carlo (MC)** season simulations. This guide gives you a practical, implementation-ready blueprint with data structures, math choices for player distributions, and performance tips.

---

## 0) Repo Structure (suggested)

```
fantasy-draft-bot/
├─ data/
│  ├─ projections.csv           # player_id, name, pos, team, mean_proj, age, adp, tier, role flags...
│  ├─ buckets.yaml              # position×age/tenure priors (means, CVs, injury probs)
│  └─ league.yaml               # roster rules, scoring, league size, etc.
├─ src/
│  ├─ distributions.py          # build player distributions (Normal/Lognormal/Mixture), shrinkage vs bucket
│  ├─ presample.py              # pre-sample outcomes matrix [players x samples]
│  ├─ opponents.py              # ADP+noise opponent pick model
│  ├─ scoring.py                # season scoring; optional weekly lineup model
│  ├─ mcts.py                   # Node, PUCT, progressive widening, rollout policy, backprop
│  ├─ simulate.py               # run searches for each pick, write reports
│  └─ utils.py                  # hashing, caching, config parsing, RNG seeding
└─ README.md
```

---

## 1) Inputs & Config

- **Projections**: player_id, name, pos, team, mean_proj, age, adp, tier, role.  
- **League rules**: roster slots (QB/RB/WR/TE/FLEX), league size, snake, scoring.  
- **Bucket priors**: position × age/tenure priors for variance and injury.  
- **Seeds**: set RNG seeds for reproducibility.

---

## 2) Player Outcome Distributions

Turn each projection into a distribution:

### A) Truncated Normal  
\( X \sim \max(0, \mathcal{N}(\mu, \sigma^2)) \), with \( \sigma = \text{CV}\cdot\mu \).

### B) Lognormal  
Parameters:  
\(\sigma_{\ln}^2 = \ln(1 + \text{CV}^2)\)  
\(\mu_{\ln} = \ln(\mu) - \sigma_{\ln}^2/2\)  
Draw from LogNormal.

### C) Injury/Availability Mixture (overlay)  
Sample missed games \(M \in \{0,3,6\}\) with probs \(p_0, p_3, p_6\).  
Scale: \(X_{final} = X \times (17 - M)/17\).

> Recommended: Normal or Lognormal + injury overlay.

---

## 3) Variance via Buckets & Shrinkage

Define variance & injury probs by **position × age/tenure** buckets (rookie, peak, decline). Example YAML:

```yaml
rb:
  rookie:
    cv: 0.32
    injury_probs: {p0: 0.72, p3: 0.18, p6: 0.10}
  peak_24_27:
    cv: 0.20
    injury_probs: {p0: 0.82, p3: 0.12, p6: 0.06}
  decline_28_plus:
    cv: 0.28
    injury_probs: {p0: 0.77, p3: 0.15, p6: 0.08}
wr:
  rookie:
    cv: 0.28
    injury_probs: {p0: 0.78, p3: 0.14, p6: 0.08}
  peak_24_28:
    cv: 0.18
    injury_probs: {p0: 0.85, p3: 0.10, p6: 0.05}
  decline_31_plus:
    cv: 0.22
    injury_probs: {p0: 0.82, p3: 0.12, p6: 0.06}
```

### Shrinkage (empirical Bayes)  
\(\mu^* = w\mu_{player} + (1-w)\mu_{bucket}\)  
\(\sigma^{2*} = w\sigma_{player}^2 + (1-w)\sigma_{bucket}^2\)

### Z-adaptive variance  
\(z = (\mu_{player}-\mu_{bucket})/\sigma_{bucket}\)  
\(\sigma^* = \sigma_{bucket}(1+\alpha|z|)\)

---

## 4) Pre-sampling Outcomes

- Pre-sample matrix \(D \in \mathbb{R}^{P \times S}\) where `P=#players`, `S=5000–20000`.  
- Each row = player outcomes, each col = one “season world”.  
- Rollouts pick a **column index** to score entire rosters → faster & fairer.

---

## 5) Opponent Draft Model

- Rank by ADP + Normal(0,σ) noise (σ≈6–12).  
- Enforce roster needs (bias picks toward needed positions).  
- Precompute noisy orders per rollout seed for vectorization.

---

## 6) State Representation

- round_idx, pick_idx  
- roster signature (counts + player_ids)  
- available players bitset (top 150–200)  
- RNG column index  

---

## 7) MCTS Algorithm

### Selection (PUCT)  
\[
Q(s,a) + c \cdot P(a|s) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}
\]

### Expansion  
- Expand top-k (start 4–6, widen up to 10–12).

### Rollout  
- Simulate rest of draft (opponent model).  
- Your later picks by greedy VORP/needs heuristic.

### Value  
- Use team season points (sum from D[:,col]).  
- Alternative: playoff prob, expected wins.

### Backpropagation  
- Update visits and averages back to root.

---

## 8) Season Scoring

- **Baseline:** sum starter totals, bench = replacement-level EV.  
- **Optional weekly model:** scale season total into weekly profile + lineup selection.

---

## 9) Performance Tips

- Top-k only (≤10 candidates).  
- Pre-sample outcomes.  
- Approximate late rounds.  
- Cache node values.  
- Use multiprocessing/Numba if slow.

---

## 10) Calibration

- Sanity: RB-heavy should show boom/bust, WR-heavy stable.  
- Backtest: run on old seasons.  
- Sensitivity: ±20% CV, check stability.  
- Compare to greedy baseline.

---

## 11) Parameter Defaults

- Pre-samples per player: 10k  
- MCTS sims per pick: 1k–3k  
- Top-k expansion: start 6, widen to 10  
- ADP noise σ: 8–10  
- RB CV: rookie 0.30–0.40, peak 0.18–0.22, 28+ 0.25–0.35  
- WR CV: rookie 0.25–0.30, peak 0.16–0.20, 31+ 0.20–0.25  

---

## 12) Workflow Checklist

- [ ] Define distributions per player with buckets + injury overlay.  
- [ ] Pre-sample player×season matrix.  
- [ ] Opponent picks = ADP+noise + needs.  
- [ ] MCTS with PUCT, progressive widening.  
- [ ] Value = season points.  
- [ ] Cache states.  
- [ ] Validate with history.  
