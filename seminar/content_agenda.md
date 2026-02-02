# Seminar agenda — UQSA

**Organiser:** NTNU and Politecnico di Milano (PoliMi)  
**Dates:** 3–4 February 2026  
**Instructor:** Leif Rune Hellevik  
**Location:** Politecnico di Milano, Milan, Italy (Natta-tav)

The seminar is structured over two half-days, combining conceptual framing with guided, notebook-based exploration. The timetable respects the confirmed room reservations.

---

## Day 1 — Tuesday 3 February 2026

**Time: 10:00–15:00**

| Time | Block | What you do / material |
|---|---|---|
| **10:00–10:40** | **GSA for cardiovascular models (Welcome + framing)** | **Motivation, concepts, and intuition** (Lecture slides) |
| **10:40–11:00** | Transition: from slides to statistics | Explain why variance is needed for Sobol indices |
| **11:00–11:30** | Foundations: variance and total variance | *preliminaries.ipynb* (selected cells) |
| **11:30–11:45** | Coffee break | — |
| **11:45–12:30** | What does sensitivity mean in practice? | *sensitivity_introduction.ipynb* (Part 1) |
| **12:30–13:15** | First-order Sobol indices | *sensitivity_introduction.ipynb* (Part 2) |
| **13:15–14:00** | Lunch break | — |
| **14:00–15:00** | What is missing in first-order indices? | *sensitivity_introduction.ipynb* (Part 3: total indices + interactions) |

### Day 1 — description of blocks

**GSA for cardiovascular models (Lecture slides)**  
Conceptual framing of uncertainty, credibility, and global sensitivity analysis in the context of complex models and digital twins. Introduces key ideas without formulas and sets expectations for the notebook-based format.

**Foundations: variance and total variance**  
Selected parts of *preliminaries.ipynb* to recall variance and the law of total variance as the statistical foundation for Sobol indices.

**Sensitivity in practice (Part 1)**  
Guided exploration of *sensitivity_introduction.ipynb*: scatterplots, conditional behaviour, and qualitative intuition about input–output relationships.

**First-order Sobol indices (Part 2)**  
Computation and interpretation of first-order Sobol indices within *sensitivity_introduction.ipynb*.

**What is missing in first-order indices? (Part 3)**  
Introduction of total Sobol indices and interaction effects within *sensitivity_introduction.ipynb*.

**Outcome of Day 1:**  
Participants understand what Sobol indices mean and why interactions matter.

---

## Day 2 — Wednesday 4 February 2026

**Time: 10:00–13:00**

| Time | Block | What you do / material |
|---|---|---|
| **10:00–10:15** | Recap and framing of Day 2 | From “what Sobol indices mean” → “how we compute them” |
| **10:15–10:45** | Monte Carlo in practice | *monte_carlo.ipynb* |
| **10:45–11:10** | Higher-order effects and interactions | *sensitivity_higher_order.ipynb* |
| **11:10–11:25** | Coffee break | — |
| **11:25–11:55** | Polynomial Chaos (PCE) approach | *introduction_gpc.ipynb* |
| **11:55–12:25** | Application I: arterial wall models | *wall_models.ipynb* |
| **12:25–13:00** | Application II: g\*-function + wrap-up | *gstar_function.ipynb* |

### Day 2 — description of blocks

**Recap and framing of Day 2**  
Short synthesis: Day 1 focused on meaning and intuition; Day 2 focuses on computation and applications.

**Monte Carlo in practice**  
Demonstration of Sobol estimation via Monte Carlo, including sampling cost and convergence.

**Higher-order effects and interactions**  
Exploration of interaction effects and interpretation of total vs. first-order indices.

**Polynomial Chaos approach**  
Comparison of Monte Carlo and PCE: same quantities (variance and Sobol indices), but different efficiency and assumptions.

**Application I — arterial wall models**  
Realistic case study showing which parameters dominate uncertainty in arterial wall predictions and why this matters for credibility.

**Application II — g\*-function + wrap-up**  
Benchmark nonlinear example illustrating nonlinearity and interactions, followed by key take-home messages and pointers to materials.

---

## Practical information

The seminar combines guided demonstration with interactive exploration. Participants who wish to run code will be provided with ready-to-use notebooks and Colab links.

Interactive notebooks and slides will be shared after the seminar.

Familiarity with basic probability and numerical modelling is helpful, but no prior experience with sensitivity analysis is assumed.
