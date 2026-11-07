# Agribusiness Exploratory Data Analysis (EDA)

[![Current Milestone: Week 3 – Exploratory Data Analysis Report Design](https://img.shields.io/badge/Milestone-Week%203%20%7C%20EDA%20Report%20Design-blue.svg)](reports/week3_eda_report_design.md)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Project Phase**: **Week 3 – Exploratory Data Analysis Report Design**  
> **Role**: Junior Data Scientist Intern  
> **Repository**: `agribusiness-eda-analysis`  
> **Primary Deliverable**: [`reports/week3_eda_report_design.md`](reports/week3_eda_report_design.md)

---

## 1. Project Title & Overview

The **Agribusiness Exploratory Data Analysis (EDA)** project provides a structured analytical architecture to investigate multi-decadal crop production trends, spatial yield disparities, and high-frequency wholesale commodity market dynamics.

Building upon the data sourcing and cleaning foundations established in previous phases, Week 3 designs the complete exploratory framework that translates clean tabular data into actionable agribusiness intelligence before statistical forecasting and predictive modeling.

```
+--------------------------------------------------------------------------------------------------+
|                                    Agribusiness EDA Pipeline                                     |
+--------------------------+-----------------------------+-----------------------------------------+
| Descriptive Analytics    | Spatio-Temporal Analytics   | Correlation & Diagnostics               |
| - Univariate profiling   | - Regional yield indices    | - Pearson & Spearman matrices           |
| - Parametric & robust    | - Cropping seasonal cycles  | - Bivariate regression trendlines       |
| - Distribution shapes    | - Market price volatility   | - Anomaly & structural break audit      |
+--------------------------+-----------------------------+-----------------------------------------+
```

---

## 2. Business Problem & Analytical Objectives

Agribusiness participants—including commercial farm managers, food processors, commodity traders, and government planners—face structural uncertainties stemming from biological weather dependencies, fragmented market linkages, and harvest price gluts.

This project designs an exploratory framework to answer critical business questions:
1. **Regional Dominance**: Which agricultural regions contribute the largest volumes to national crop supply?
2. **Productivity Frontiers**: How do crop yields (kg/ha) vary across agro-climatic zones, and where do persistent yield gaps exist?
3. **Price Seasonality**: What recurring monthly cycles govern wholesale mandi prices between harvest surges and lean seasons?
4. **Price Volatility**: Which agricultural commodities exhibit the highest market risk and return dispersion?
5. **Acreage Elasticity**: Does expanding cultivated land area produce proportional increases in harvest output, or do diminishing returns dominate?
6. **Feature Discovery**: Which historical variables and rolling lag indicators demonstrate the strongest predictive potential for future forecasting?

> **Methodological Disclaimer**: In accordance with the ethical research guidelines of this internship assignment, **no actual empirical statistics or findings are claimed or fabricated**. All numerical examples and outputs are explicitly designated as **HYPOTHETICAL DEMONSTRATIONS**.

---

## 3. Key Agribusiness Metrics and KPIs

| KPI Name | Mathematical Formula | Business Significance |
| :--- | :--- | :--- |
| **Total Production ($P$)** | $\sum P_i \quad [\text{Metric Tonnes}]$ | Total physical volume available for domestic supply and processing |
| **Average Yield ($Y$)** | $\frac{\sum P \times 1000}{\sum A} \quad [\text{kg/ha}]$ | Fundamental measure of land productivity and agronomic technology |
| **Production YoY Growth** | $\left(\frac{P_t - P_{t-1}}{P_{t-1}}\right) \times 100 \quad [\%]$ | Annual supply expansion rate; indicates capacity changes |
| **Yield YoY Growth** | $\left(\frac{Y_t - Y_{t-1}}{Y_{t-1}}\right) \times 100 \quad [\%]$ | Intensive efficiency improvements independent of land clearing |
| **Wholesale Modal Price** | Representative clearing price $[\text{INR/q}]$ | Prevailing transaction rate for physical commercial procurement |
| **Price Volatility ($\sigma_{\text{ann}}$)**| $\text{std}(\ln(P_t / P_{t-1})) \times \sqrt{252}$ | Quantifies financial procurement risk and required hedging buffers |
| **Spatial Yield Index ($SYI$)**| $\left(\frac{Y_{\text{region}}}{Y_{\text{national}}}\right) \times 100 \quad [\%]$ | Measures relative competitive advantage against national benchmarks |

---

## 4. Visualisation Plan

The EDA report specifies seven core visual analytics components:
- **Horizontal Bar Charts**: Ranked comparisons of crop production and regional contributions.
- **Histograms with KDE**: Distributional skewness and kurtosis profiling for crop yields.
- **Box Plots**: Interquartile dispersion, median stability, and tail outlier visualization across states.
- **Bivariate Scatter Plots**: OLS regression trendlines examining Area vs. Production elasticity.
- **Time-Series Trajectory Lines**: 7-day and 30-day rolling moving average commodity price trends.
- **Calendar Heatmaps**: Monthly price seasonal patterns across consecutive agricultural years.
- **Correlation Heatmaps**: Annotated Pearson and Spearman association matrices.

---

## 5. Repository Structure

```
agribusiness-eda-analysis/
│
├── README.md                                  # Executive overview, methodology, setup & roadmap
├── requirements.txt                           # Pinned dependencies for reproducible execution
├── .gitignore                                 # Excludes virtualenvs, bytecode, and data files
│
├── data/
│   ├── README.md                              # Variable dictionary, schemas, and storage rules
│   ├── raw/                                   # Immutable raw survey & price snapshots (read-only)
│   │   └── .gitkeep
│   └── processed/                             # Cleaned, standardized analytical panels for EDA
│       └── .gitkeep
│
├── notebooks/
│   └── 01_agribusiness_eda.ipynb              # Interactive EDA walkthrough with hypothetical data
│
├── src/
│   ├── __init__.py                            # Package initialization
│   ├── utils.py                               # Standardized logging, directory management, timers
│   ├── eda.py                                 # Statistical profiling, KPI calculators, anomaly filters
│   └── visualisation.py                       # Publication-grade Matplotlib & Seaborn plotting suite
│
└── reports/
    └── week3_eda_report_design.md             # Complete 23-section comprehensive EDA report design
```

---

## 6. Technology Stack

- **Core Programming**: [Python 3.10+](https://www.python.org/)
- **Data Wrangling & Profiling**: [Pandas](https://pandas.pydata.org/) (v2.0+), [NumPy](https://numpy.org/) (v1.24+), [SciPy](https://scipy.org/) (v1.11+)
- **Visual Analytics**: [Matplotlib](https://matplotlib.org/) (v3.7+), [Seaborn](https://seaborn.pydata.org/) (v0.12+)
- **Interactive Development**: [Jupyter Notebook](https://jupyter.org/) (v7.0+)

---

## 7. How to Reproduce the Work

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/agribusiness-eda-analysis.git
cd agribusiness-eda-analysis
```

### 2. Create and Activate Virtual Environment
```bash
# On Linux/macOS
python -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Execute Demonstration Notebook
```bash
jupyter notebook notebooks/01_agribusiness_eda.ipynb
```

---

## 8. Limitations

1. **Reporting Lags**: Official sub-national agricultural production statistics carry an 18-to-24-month administrative compilation lag.
2. **Boundary Discontinuities**: Historical district splits create longitudinal panel imbalances.
3. **Correlation vs. Causation**: Strong statistical associations between inputs and yields do not establish direct agronomic causation without controlled field trials.
4. **Mandi Data Gaps**: Rural assembly markets with intermittent reporting require temporal interpolation that may smooth localized volatility.

---

## 9. Current Status & Future Work (Week 4)

- [x] **Week 3 Milestone Completed**:
  - Publication of the 23-section comprehensive technical report ([`reports/week3_eda_report_design.md`](reports/week3_eda_report_design.md)).
  - Implementation of reusable statistical profiling functions in `src/eda.py`.
  - Development of publication-ready visual analytics routines in `src/visualisation.py`.
  - Interactive demonstration notebook with hypothetical multi-region panel (`notebooks/01_agribusiness_eda.ipynb`).
- [ ] **Week 4 Upcoming**: Econometric Time-Series Modeling (ARIMA / SARIMA) and Machine Learning Price Forecasting.
