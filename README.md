# IFCS 2026 Data Challenge
### Profiling and Predicting Financial Distress in Italian SMEs

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Competition](https://img.shields.io/badge/Event-IFCS%202026%20Data%20Challenge-blueviolet.svg)]()
[![Validation](https://img.shields.io/badge/Validation-5--Fold%20Stratified%20CV-green.svg)]()

> **Author**: **Alessandro Castellani**  
> *Undergraduate background in Mathematics (Università dell'Insubria) | Graduate coursework in Applied Statistics & Data Science (Università Cattolica del Sacro Cuore)*  
> 📬 [alecaste041202@gmail.com](mailto:alecaste041202@gmail.com) | 🔗 [LinkedIn Profile](https://www.linkedin.com/in/alessandro-castellani/) | 🐙 [GitHub Profile](https://github.com/apmcastellani-projects)

---

## 📌 Competition Overview

The **IFCS 2026 Data Challenge** is an international competitive data challenge organized by the **International Federation of Classification Societies (IFCS)**.

The task revolves around financial statement analysis for approximately **14,000 Italian Small and Medium Enterprises (SMEs)** for FY2023:
1. **Unsupervised Profiling (Task A)**: Clustering firms based on financial structures and identifying regional distribution patterns across Italian provinces.
2. **Supervised Classification (Task B)**: Building a high-recall, calibrated classification pipeline to detect early signs of **Financial Distress** before insolvency occurs.

---

## 🔬 Methodology & Engineering Highlights

```
[Raw SME Financial Data] 
       │ 
       ▼
[Feature Engineering: Signed Log Transform: sign(x) · ln(1 + |x|)]
       │
       ▼
[5-Fold Stratified Cross-Validation (Per-Fold Scaling to prevent Leakage)]
       │
       ▼
[Out-of-Fold (OOF) Prediction Generation & ROC-AUC Evaluation]
       │
       ▼
[Multi-Criteria Threshold Optimization: Youden's J, Max F1, Min Distance (0,1)]
       │
       ▼
[Pipeline Persistence (.pkl) & Out-of-Sample Batch Inference]
```

### 1. Robust Signed Log Transformation
Financial indicators (such as Operating Income and Total Financial Expenses) often take both positive and negative values while exhibiting extreme heavy tails. We formulated a symmetric signed log transform:
$$f(x) = \text{sign}(x) \cdot \ln(1 + |x|)$$
This preserves polarity (profit vs. loss) while dampening multi-order-of-magnitude scaling discrepancies.

### 2. Leakage-Free Stratified Cross-Validation
To guarantee strict independence and prevent data leakage:
- Evaluated over a **5-Fold Stratified Cross-Validation** split.
- Feature scalers (`StandardScaler`) are strictly fit on the training fold and applied to the validation fold.

### 3. Systematic Decision Threshold Optimization
In distressed-firm classification, false negatives (missing an insolvent firm) carry vastly different real-world costs than false alarms. Rather than defaulting to a naive $0.5$ threshold, three optimization criteria were benchmarked on Out-of-Fold (OOF) probabilities:
- **Youden's J Index**: Maximizes informedness $(\text{TPR} - \text{FPR})$.
- **Max F1-Score**: Balances precision and recall under class imbalance.
- **Euclidean Distance to Ideal**: Minimizes $\sqrt{(1 - \text{TPR})^2 + \text{FPR}^2}$ in ROC space.

---

## 📂 Repository Layout

```text
ifcs-data-challenge-2026/
├── docs/
│   └── 01_participant_brief.pdf     # Official IFCS competition task specification
├── src/
│   ├── train_model.py               # Feature pipeline, CV, threshold tuning, and model export
│   ├── predict.py                   # Out-of-sample batch inference script
│   └── validate_predictions.py     # Schema, distribution, and sanity checks on final submissions
└── README.md
```

---

## 🛠️ Reproduction & Inference

```bash
git clone https://github.com/apmcastellani-projects/ifcs-data-challenge-2026.git
cd ifcs-data-challenge-2026/src
pip install pandas numpy scikit-learn

# Train model and generate optimal threshold
python train_model.py

# Run batch predictions on test set
python predict.py
```

---

## 📬 Contact & Opportunities

I am actively seeking **internship and analytical collaboration opportunities** across quantitative modeling, sports analytics, and data science.

- **Email**: [alecaste041202@gmail.com](mailto:alecaste041202@gmail.com)
- **LinkedIn**: [linkedin.com/in/alessandro-castellani](https://www.linkedin.com/in/alessandro-castellani/)
- **GitHub**: [github.com/apmcastellani-projects](https://github.com/apmcastellani-projects)
