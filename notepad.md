# Mental Model Theory — Math Notes (working notepad)

Author: Codex assistant
Date: 2025-10-19
Scope: Notes distilled from Mental Model Math.md to guide implementation of a symbolic core and a differentiable ML model (DMMC).

---

## 1) Quick Recap (my understanding)
- Reasoning state is a finite set of partial models (“worldlets”) Σ = {m_k}. Each worldlet stores explicit truths (and optionally negative literals) while leaving others unknown (⊥).
- Two evaluation regimes:
  - Inspection (fast): read off monotone consequences from τ(m) (truth-only projection) via lower-bound valuation v_m.
  - Deliberation (search): expand missing possibilities (E) and test necessity by supervaluation over completions Comp(E(Σ)).
- Premises are constructed via minimal generator G(·) and merged with ⊗ (consistent union product). Expansion E(·) makes meaning fully explicit (e.g., conditional adds {¬χ}).
- ML angle (DMMC): latent K worldlets in {1,0,⊥}, with an encoder E_θ(x) → Z, an inspection head for monotone queries, an expansion/search module (symbolic or learned), and losses that encourage MMT priors (few models, truth-only sparsity, shallow expansion when possible).

---

## 2) Core Objects & Semantics (to encode)
- Literals/atoms: 𝒜 = {p₁…p_n}, Lit = {p, ¬p}.
- Worldlet m ⊆ Lit (consistent); projection τ(m) = {p ∈ 𝒜 | p ∈ m}.
- Completions Comp(m): classical total assignments consistent with all signed facts in m; Comp(Σ) = ⋃_m Comp(m).
- Supervaluation:
  - Necessary: Σ ⊨ φ ⇔ ∀m∈Σ ∀s∈Comp(m): s ⊨ φ.
  - Possible: Σ ⊨◇ φ ⇔ ∃m∈Σ ∃s∈Comp(m): s ⊨ φ.
- Inspection (monotone ψ): Σ ⊢_ins ψ iff ∀m∈Σ: v_m ⊨ ψ, where v_m(p)=1 if p∈τ(m) else 0. Lemma: inspection ⇒ necessity for monotone ψ.

---

## 3) Construction Operators (to implement)
- Merge (⊗): pairwise unions of worldlets, filter inconsistent.
- Generator G(·) (minimal, truth-only):
  - G(p) = {{p}}; G(¬χ) = {{¬χ}}.
  - G(χ∧ψ) = {m_χ ∪ m_ψ}.
  - G(χ∨ψ) = G(χ) ∪ G(ψ) ∪ optional {m_χ ∪ m_ψ}.
  - G(χ→ψ) = {{χ, ψ}} (not-χ left implicit initially).
  - G(χ↔ψ) = {{χ, ψ}, {¬χ, ¬ψ}}.
- Expansion E(·) (explicitate omitted possibilities):
  - E respects connectives; crucially E(G(χ→ψ)) = {{χ, ψ}, {¬χ}}.
  - Thm: Comp(E(Σ_P)) equals the set of classical models of premises P.

---

## 4) Quantifiers/Counts/Modality (options layer)
- Set-worldlets for FOL: store positive witnesses W⁺ and structural set constraints C (e.g., S_A ⊆ S_C). Domain D unknown but non-empty.
- G(∃x P(x)) → witness P(a). G(∀x (A→C)) → subset constraint only. “Most(A,C)” → |A∩C| > |A\C|.
- Modal: □φ iff Σ ⊨ φ; ◇φ iff Σ ⊨◇ φ. Causality via coupled completions (actual vs. counterfactual) with sufficiency + tendency constraints.

---

## 5) Complexity & Cost (priors to encode)
- Cost(Σ) = α·|Σ| + β·avg unk(m). Pressure toward “one model is better than many” and minimal explicit info unless needed.
- Case-split bound: |E(G(φ))| ≤ 2^(#∨) · 2^(#→); |G(φ)| ≤ 2^(#∨).

---

## 6) Probabilities from Possibilities
- π(m) over worldlets; within each m, distribute over Comp(m) (assume near-uniform on unknowns).
- P(φ | Σ, π) = ∑_m π(m) · P_{s∼μ_m}[s ⊨ φ].
- Rough-average property for disjunction when joint worldlet omitted: P(A∨B) ≈ ½(P(A)+P(B)).

---

## 7) DMMC (Differentiable Mental-Model Calculus) — Implementation Sketch
- Latent state: Z ∈ {1,0,⊥}^{K×n} via logits ℓ and Gumbel-Softmax/straight-through.
- Modules:
  - Encoder E_θ(x) → Z (text/vision to worldlets).
  - Inspector: evaluates monotone queries with v_m (or relaxed \hat v_m) and t-norm/conorm.
  - Expander/Searcher: apply E-templates on demand; SAT/SMT or MILP for counterexample search over Comp(E(Σ)).
  - Probability head: softmax over worldlet scores → π.
- Losses:
  - Task CE/Brier on necessary/possible or QA labels via supervaluation on expanded Σ.
  - MMT prior: λ₁·|Σ| + λ₂·avg unk(m) + λ₃·ExpCost.
  - Optional ST estimator for non-differentiable expansion decisions.

---

## 8) Design Choices To Lock Down (please confirm)
1) Disjunction in G by default:
- Option A (MMT-minimal): G(A∨B) = {{A},{B}} only; joint added only by expansion or encoder.
- Option B (richer explicit): include joint {{A,B}} at construction. Default? (Leaning A for stronger MMT prior.)

2) Negatives inside worldlets:
- Keep ¬p in m but ignore at inspection (via τ). Activates during deliberation/search. Confirm yes.

3) Expansion budget/strategy:
- Depth-1 for conditionals first; selective expansion guided by learned heuristics and Cost(Σ). Define a scheduler.

4) Monotone-formula detector:
- Implement syntactic checker to route queries to inspector vs. searcher. Edge: literals vs. mixed forms.

5) K (max worldlets):
- Start small (K=3–5) for toy tasks; scale as needed. Regularize to actually use fewer.

6) Quantifiers layer:
- Initial version: propositional only; add set-worldlets later. If needed early, we’ll encode ∃ (witness) and ∀ (subset constraint) first.

7) Probabilities:
- Use π over worldlets; within-worldlet completion uniformity is an approximation. Consider temperature to control spread.

8) Backend for search:
- Start with PySAT/Z3 via a thin encoding of Comp(E(Σ)). Provide fallbacks to pure Python bit-vector SAT for portability.

---

## 9) Edge Cases & Pitfalls (watchouts)
- Inconsistency during ⊗: need fast mask to drop pairs where p and ¬p collide.
- Double counting in probability if joint worldlet exists alongside singles; ensure worldlet priors π are normalized and completions within each m do not overlap across worldlets (they can, but π handles mixture correctly).
- Non-monotone queries at inspection lead to unsoundness; must gate via monotone check.
- Quantifiers: domain variability can make supervaluation subtle; fix D or constrain via K knowledge base to avoid degenerate completions.
- Straight-through estimators can destabilize; cap expansion steps and use auxiliary predictors of counterexamples.

---

## 10) Minimal v0 Plan (symbolic core)
- Types: Atom, Literal, Worldlet (bitset of {1,0,⊥}), ModelSet Σ (list of worldlets).
- Ops: `merge_otimes(Σ₁, Σ₂)`, `generator_G(formula)`, `expand_E(Σ)`, `tau(worldlet)`, `inspect_monotone(Σ, ψ)`, `supervaluate(Σ, φ)`, `counterexample(P, θ)`.
- SAT bridge: encode Comp(E(Σ)) as CNF constraints; ask SAT for s ⊭ θ.
- Tests: MP easy; MT requires expansion; AC/DN invalid; disjunction rough-average probability check.

---

## 11) Datasets & Evaluation Ideas
- scikit-learn classics: Iris, Wine, Breast Cancer (Wisconsin diagnostic). Primary target for first experiments.
- Synthetic propositional sets matching classic MMT tasks (MP/MT/AC/DN, biconditionals).
- Syllogisms/quantifiers (All/Some/Most) with set diagrams.
- Natural language conditionals from existing logic QA datasets (scaled-down first).
- Metrics: classification accuracy/F1/AUROC where applicable, calibration (ECE/Brier), plus MMT metrics: |Σ|, avg unk(m), expansion depth, inspection/share ratio, time per query.

---

## 12) Open Questions For You
- Target domain first (textual logic, visual reasoning, planning)?
- Preferred default for G(A∨B) (Option A vs. B)?
- Do we include causal layer in v1 or postpone?
- Any constraints on K or latency that shape expansion strategy?
- Should we support probabilities in v1 or ship deterministic first?

---

## 13) Next Steps (proposed)
1) Finalize the design choices above (especially disjunction default, K, and expansion budget).
2) Implement symbolic core (G, ⊗, E, τ, inspection, supervaluation) with unit tests.
3) Add SAT-backed counterexample search and necessity checker.
4) Expose a simple API: `judge(Premises, Query) → {necessary|possible|impossible}` plus traces.
5) If desired, scaffold DMMC encoder stubs and a tiny training loop to learn π and worldlet logits from labeled tasks.

---

## 14) Glossary (quick refs)
- τ(m): truth-only projection (drops negatives).
- v_m: lower-bound valuation used for inspection.
- G(·): generator (minimal explicit worldlets from a formula).
- E(·): expansion (adds omitted possibilities demanded by full meaning).
- ⊗: consistent union merge of premises.
- Comp(m): classical completions consistent with m.
- Cost(Σ): α·|Σ| + β·avg unk(m) (+ λ·ExpCost in learning).

---

## 15) scikit-learn Benchmarks (Iris, Wine, Breast Cancer)
Note: assuming “wise” refers to scikit-learn’s Wine dataset.

**Datasets**
- Iris: 150 samples, 4 numeric features, 3 classes.
- Wine: 178 samples, 13 numeric features, 3 classes.
- Breast Cancer (Wisconsin diagnostic): 569 samples, 30 numeric features, binary classes.

**Goals**
- Use these as sanity-check benchmarks to validate DMMC’s inspection vs. deliberation behavior and the effect of the MMT prior (few worldlets, low expansion).
- Compare to standard baselines and inspect model traces for interpretability.

**Feature → Atom mapping (propositionalization)**
- Discretize numeric features into monotone threshold atoms per feature j:
  - Quantile thresholds: t ∈ {q20, q40, q60, q80} ⇒ atoms p_{j,t}: x_j ≥ t.
  - Supervised thresholds: 1–3 splits from a univariate DecisionTree (stumps) per feature.
  - Optionally k-bin (k∈{3,5}); encode as cumulative atoms to preserve monotonicity.
- Each sample → truth assignment over atoms; unknowns (⊥) if a feature missing (not expected here).

**Premise/Query design for MMT evaluation**
- Generate simple rules from thresholds (e.g., x_j ≥ t → class=c) to test MP vs. MT dynamics.
- For multiclass, use one-vs-rest class atoms C_c and encode premises as implications; evaluate necessity vs. possibility of class labels.

**Baselines**
- LogisticRegression, LinearSVC/RBF-SVC, DecisionTree, RandomForest; measure accuracy/AUROC and calibration.
- Ensure DMMC is not disadvantaged: tune thresholds and K fairly via CV.

**Protocol**
- Stratified 5× (train/val/test = 70/15/15) with fixed seeds; early dev can use 80/20 split.
- Report mean±std across runs for: accuracy/F1 (macro), AUROC (binary), ECE/Brier (probabilistic), and MMT metrics (|Σ|, unk, expansions, inspection% ).

**Loading hooks (sklearn)**
- `from sklearn.datasets import load_iris, load_wine, load_breast_cancer`
- `X, y = load_iris(return_X_y=True)`, similarly for wine and breast_cancer.

**Integration sketch**
1) Build `discretize(X)` → boolean atom matrix A and threshold dictionary.
2) Encode premises from atoms to initiate Σ via G and ⊗; set class atoms as queries.
3) Run inspection for monotone queries; escalate to expansion/search when needed.
4) Compare predictions and compute metrics; log Σ size and expansion counts.

**Open knobs**
- K (worldlets): start at 3 for Iris/Wine, 5–7 for Breast Cancer.
- Threshold count per feature (1–4); regularize via λ terms to keep Σ small and expansions rare.

