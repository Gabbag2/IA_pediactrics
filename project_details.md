# xCEBRA for Explainable Seizure Prediction in Pediatric Epilepsy

## Non-Invasive Localization and Pre-Ictal Dynamics from Scalp EEG

---

## 1. Motivation and Clinical Context

Pediatric drug-resistant epilepsy often requires surgical intervention, which in turn demands precise localization of the epileptogenic zone. Current clinical practice relies on intracranial EEG (iEEG/SEEG) — an invasive procedure that carries significant morbidity risks, especially in children. A computational approach that can extract spatial and spectral biomarkers of the epileptogenic zone directly from routine scalp EEG would reduce the need for invasive monitoring and accelerate clinical decision-making in pediatric neurology.

This project applies xCEBRA (Schneider et al., AISTATS 2025), a regularized contrastive learning framework with identifiable attribution maps, to scalp EEG recordings from pediatric epilepsy patients. The goal is to build an explainable model that (1) predicts the transition from normal brain activity to seizure and (2) identifies which brain regions and frequency bands drive this transition — for each individual child — without opening the skull.

---

## 2. Core Hypotheses

### H1 — Universal Pre-Ictal Dynamics

The transition from interictal to ictal state follows a stereotyped trajectory in a learned latent space, characterized by signatures consistent with critical phase transitions (increasing variance, decreasing local dimensionality, accelerating drift). These signatures are **universal across patients**, regardless of seizure focus location.

### H2 — Patient-Specific Spatial Attribution

While the latent dynamics are shared, the **input features driving the transition are patient-specific** and reflect the individual epileptogenic zone. The xCEBRA Jacobian, evaluated locally on each patient's data, reveals which scalp channels and frequency bands carry the pre-ictal signature for that child.

### H3 — Non-Invasive Lateralization

The attribution maps derived from xCEBRA correctly lateralize (and to some extent regionalize) the epileptogenic zone when compared against clinical ground truth, providing a non-invasive surrogate for information typically obtained through iEEG.

### H4 — Spectral Specificity

The pre-ictal transition is not uniformly distributed across frequency bands. We hypothesize that high-frequency activity (beta, gamma) and connectivity changes in specific bands carry more predictive information than low-frequency power, and that xCEBRA's attribution maps will quantitatively confirm this.

---

## 3. Dataset

**CHB-MIT Scalp EEG Database** (PhysioNet, open access)

- 23 pediatric patients (ages 1.5–19 years) with drug-resistant focal epilepsy
- 844 hours of continuous scalp EEG
- 198 annotated seizures with onset/offset timestamps
- 18–23 channels, international 10-20 system, sampled at 256 Hz
- Clinical metadata including seizure focus lateralization

The standardized 10-20 montage is critical: channel names (F7, T4, C3, etc.) map to consistent anatomical regions across all patients, making channel-level attribution maps directly comparable.

---

## 4. Feature Engineering

Features are computed in sliding windows (5 seconds, 50% overlap). Each window produces a feature vector that is the input to xCEBRA. Features are organized in two groups serving complementary purposes.

### 4.1 Channel × Band Features (for localization)

For each of the 18 EEG channels and each of the 5 canonical frequency bands (delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, gamma 30–80 Hz):

- **Relative spectral power**: power in the band divided by total power in the window. Normalization makes features comparable across patients and sessions.
- **Sample entropy**: per-channel complexity measure. Drops before seizures as brain activity becomes more rhythmic and predictable.

This gives 18 channels × 6 features = 108 features with explicit spatial identity.

### 4.2 Network Features (for generalization)

Computed across channel pairs or globally:

- **Phase Locking Value (PLV)**: for each pair of homologous channels (F3/F4, C3/C4, T3/T4, etc.) in each frequency band. Captures synchronization dynamics independently of amplitude.
- **Inter-hemispheric power asymmetry**: ratio of spectral power between homologous channels. Directly informative for lateralization.
- **Graph-theoretic summaries**: mean connectivity strength, clustering coefficient, and modularity computed from the PLV adjacency matrix. These are topography-agnostic and capture global network state.

### 4.3 Why Both Groups Are Needed

Channel × band features carry spatial information necessary for localization but could in principle lead to patient-specific overfitting. Network features are inherently more generalizable (they describe network state, not specific electrode readings) but lose spatial specificity.

xCEBRA resolves this tension through its architecture: the model learns from both feature types simultaneously, and the Jacobian attribution map tells us *post hoc* which features actually matter. If the model relies on channel-specific features for the pre-ictal dimensions, the attribution reveals the focus. If it relies on network features, it reveals universal mechanisms. The data decide.

---

## 5. Model Architecture

### 5.1 xCEBRA Multi-Objective Training

Following the xCEBRA framework, we train a contrastive model with two simultaneous objectives, each controlling a subset of latent dimensions:

**Objective 1 — Behavior-contrastive (ictal state):**
Latent dimensions 0–3 are trained with a contrastive loss conditioned on the ictal state label (interictal / pre-ictal / ictal). The pre-ictal label is assigned to the 5-minute window preceding annotated seizure onset. This forces these dimensions to encode seizure-relevant information.

**Objective 2 — Time-contrastive (temporal structure):**
Latent dimensions 4–10 are trained with a time-contrastive loss (nearby time points should be close in latent space). This captures the general temporal dynamics of EEG without any seizure labels, serving as a reference baseline.

### 5.2 Jacobian Regularization

The `JacobianReg` regularizer is applied during training following a linear ramp-up schedule (off for the first 25% of training, linearly increasing until 50%, then held constant). This regularization is what gives xCEBRA its identifiability guarantees for the attribution maps — it encourages the learned mapping to have a well-conditioned Jacobian, making the gradient-based attributions reliable.

### 5.3 Attribution Computation

After training, the attribution map is computed via the Inverted Neuron Gradient method:

```python
method = cebra.attribution.init(
    name="jacobian-based",
    model=model,
    input_data=features,
    output_dimension=model.num_output
)
result = method.compute_attribution_map()
```

The resulting Jacobian matrix J has shape (latent dimensions × input features). Rows 0–3 (behavior-contrastive) reveal which features encode the ictal state. Rows 4–10 (time-contrastive) reveal which features encode general EEG dynamics.

---

## 6. The Generalization Problem and How xCEBRA Addresses It

references for xCEBRA: 
https://arxiv.org/abs/2502.12977
https://cebra.ai/docs/demo_notebooks/Demo_xCEBRA_RatInABox.html 
### 6.1 The Core Tension

Different children have seizure foci in different brain regions (temporal left, frontal right, occipital, etc.). A model trained on all patients could either:

- Learn patient identity instead of brain states (overfitting)
- Learn only global network features and lose spatial specificity (underfitting for localization)

### 6.2 Why xCEBRA Resolves This

The xCEBRA Jacobian is **evaluated locally** at each input data point. The model f(x) → z has fixed weights after training, but J(x) = ∂f/∂x depends on the specific input x.

When evaluated on data from Child A (temporal right focus), J(x) will show high sensitivity to T4_gamma, T6_beta. When evaluated on data from Child B (frontal left focus), J(x) will show high sensitivity to F3_gamma, F7_beta. Same model, same weights, different attribution maps — because the data sit in different regions of the input space where the nonlinear function has different local gradients.

This is not a workaround — it is the core mathematical property of xCEBRA. The model learns the universal grammar of pre-ictal transitions ("gamma power surges locally, entropy drops, synchronization increases"). The Jacobian tells you *where specifically* it happens in each child.

### 6.3 Analogy

An MRI machine uses the same physics for every patient. The image is unique to each patient. The radiologist interprets each image individually. Here, xCEBRA is the MRI machine (universal model), the attribution map is the image (patient-specific), and the clinician reads the map for localization.

---

## 7. Validation Strategy

### 7.1 Leave-One-Patient-Out (LOPO) Cross-Validation

Train xCEBRA on 22 patients, evaluate on the held-out 23rd. For each held-out patient:

- **Latent trajectory analysis**: verify that the pre-ictal trajectory in the held-out patient follows the same pattern (drift, variance increase) as seen in training patients.
- **Attribution map computation**: compute the Jacobian on the held-out patient's pre-ictal segments.
- **Lateralization accuracy**: compare the hemisphere with highest attribution to the clinically documented seizure focus. Report concordance rate across all 23 folds.

This is the most critical test. If lateralization accuracy is significantly above chance (50%) in LOPO, the model has learned genuinely universal pre-ictal features and the Jacobian successfully individualizes them.

### 7.2 Latent Space Quality Checks

**Patient vs. state encoding**: color latent space points by patient identity vs. by ictal state. The desired outcome is mixing by patient (no patient-specific clusters) and separation by state (interictal, pre-ictal, ictal form distinct regions). If patients cluster separately, the model has learned identity, not states.

**Consistency metric**: compute the CEBRA consistency score between models trained on different subsets of patients. High consistency indicates the latent geometry is robust to patient composition.

### 7.3 Phase Transition Metrics in Latent Space

For each seizure in the dataset, compute the following metrics on the latent trajectory aligned to seizure onset (t=0):

- **Drift velocity**: v(t) = ‖z(t+1) − z(t)‖ / Δt. Expected: ramp-up before t=0.
- **Local variance**: σ²(t) over a rolling window. Expected: increase before t=0 (critical slowing down).
- **Distance to interictal centroid**: d(t) = ‖z(t) − μ_interictal‖. Expected: progressive divergence.
- **Local dimensionality**: estimated via nearest-neighbor statistics. Expected: decrease before t=0 (system constrains to low-dimensional manifold before seizure).

Average these across all seizures. If the signatures are consistent (narrow confidence intervals), the model captures universal pre-ictal dynamics. Report effect sizes and p-values against shuffled controls.

### 7.4 Hypothesis-Guided Attribution Testing

Using xCEBRA's `compute_attribution_score`, formally test clinical hypotheses:

**Test 1 — Spatial hypothesis**: construct a ground truth matrix where behavior-contrastive latent dimensions are attributed to channels ipsilateral to the known focus. Compute AUC.

**Test 2 — Spectral hypothesis**: construct a ground truth matrix where behavior-contrastive dimensions are attributed to high-frequency features (beta + gamma) across all channels. Compare AUC against attribution to low-frequency features (delta + theta).

**Test 3 — Null hypothesis**: construct a random attribution matrix. Verify that AUC is at chance level, confirming the metric is informative.

### 7.5 Benchmark Against Classical Methods

Compare xCEBRA localization performance against:

- **Spectral power asymmetry**: simple inter-hemispheric power ratio (clinical baseline)
- **Standard CEBRA** (without Jacobian regularization) + post-hoc feature ablation
- **Supervised CNN classifier** (interictal vs. ictal) with GradCAM attribution

If xCEBRA outperforms these baselines on lateralization accuracy and/or provides more specific localization, the identifiability guarantees of the regularized contrastive approach are justified.

### 7.6 Permutation test

Permute labels and test training CEBRA to see if the classification is driven by noise or actual mapping between neural signatures and annotations.
---

## 8. Expected Deliverables

1. **Trained xCEBRA model** on CHB-MIT with multi-objective contrastive learning
2. **Per-patient attribution maps** showing channel × band contributions to pre-ictal dynamics
3. **Lateralization accuracy** across 23 LOPO folds with comparison to clinical ground truth
4. **Phase transition analysis** demonstrating universal pre-ictal signatures in latent space
5. **Hypothesis-guided attribution scores** quantifying spatial and spectral specificity
6. **Reproducible pipeline** (Python, open-source, documented)
---

## 9. References

- Schneider, S., González Laiz, R., Filippova, A., Frey, M., & Mathis, M. W. (2025). Time-series attribution maps with regularized contrastive learning. AISTATS 2025. arXiv:2502.12977.
- Schneider, S., Lee, J. H., & Mathis, M. W. (2023). Learnable latent embeddings for joint behavioural and neural analysis. Nature, 617, 360–368.
- Shoeb, A. H. (2009). Application of machine learning to epileptic seizure onset detection and treatment. PhD thesis, MIT.
- Goldberger, A. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215–e220.
- Boran, E. et al. (2019). High-frequency oscillations in scalp EEG mirror seizure frequency in pediatric focal epilepsy. Scientific Reports, 9, 16560.

---

## 10. Hackathon Scope (48 Hours)

**Core (must deliver):**
- Feature extraction pipeline on CHB-MIT (channel × band + PLV)
- xCEBRA training with multi-objective setup
- Attribution maps for 3–5 patients as proof of concept
- Visualization of pre-ictal latent trajectories

**Stretch goals:**
- Full LOPO cross-validation (23 folds)
- Formal hypothesis-guided testing
- Phase transition metrics
- Interactive clinician dashboard

**Presentation narrative:**
"We show that xCEBRA, applied to routine pediatric scalp EEG, can predict seizures and explain where they come from — without invasive monitoring. The same model works for every child; the explanation is unique to each."
