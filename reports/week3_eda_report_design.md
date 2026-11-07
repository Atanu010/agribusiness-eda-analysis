# Week 3: Exploratory Data Analysis Report Design for Agribusiness Trends

**Project Title**: Agribusiness Exploratory Data Analysis (EDA)  
**Internship Role**: Junior Data Scientist Intern  
**Repository**: `agribusiness-eda-analysis`  
**Current Phase**: Week 3 – Exploratory Data Analysis (EDA) Report Design  
**Date**: November 2026  
**Document Type**: Technical EDA Report Design and Analytical Blueprint (Word-Ready Format)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Business Context](#2-business-context)
3. [EDA Objectives](#3-eda-objectives)
4. [Dataset Overview](#4-dataset-overview)
5. [Univariate Analysis](#5-univariate-analysis)
6. [Production Analysis](#6-production-analysis)
7. [Yield Analysis](#7-yield-analysis)
8. [Regional Analysis](#8-regional-analysis)
9. [Seasonal Analysis](#9-seasonal-analysis)
10. [Market Price Analysis](#10-market-price-analysis)
11. [Distribution Analysis](#11-distribution-analysis)
12. [Bivariate Analysis](#12-bivariate-analysis)
13. [Multivariate Analysis](#13-multivariate-analysis)
14. [Correlation Analysis](#14-correlation-analysis)
15. [Statistical Anomalies](#15-statistical-anomalies)
16. [Visualisation Plan](#16-visualisation-plan)
17. [KPI Framework](#17-kpi-framework)
18. [Insight Generation Framework](#18-insight-generation-framework)
19. [Expected Business Insights](#19-expected-business-insights)
20. [Final EDA Report Structure](#20-final-eda-report-structure)
21. [Limitations](#21-limitations)
22. [Week 4 Handoff](#22-week-4-handoff)
23. [Python Implementation Plan](#23-python-implementation-plan)

---

## 1. Executive Summary

### 1.1 Purpose of the EDA
The primary purpose of Week 3 in the Junior Data Scientist internship is to design a comprehensive, research-backed **Exploratory Data Analysis (EDA) Report Blueprint** focused on agribusiness market trends, spatial crop productivity, and wholesale commodity price dynamics. Exploratory Data Analysis is the critical bridge between raw data engineering (established in Weeks 1 and 2) and predictive econometric/machine learning forecasting (scheduled for Weeks 4 and 5).

### 1.2 Business Context & Value
Agricultural markets operate under complex biological and market forces: inelastic short-run demand, seasonal harvesting gluts, spatial yield polarization across diverse agro-ecological zones, and multi-layered wholesale price spreads. Before deploying predictive algorithms (e.g., ARIMA/SARIMA, Random Forest, or XGBoost), exploratory analysis is essential to:
- Diagnose structural skewness and non-normality in crop output and prices.
- Identify the spatial clustering of agricultural productivity.
- Quantify recurring seasonal cycles (Kharif harvest dips vs. lean-season peaks).
- Formulate high-leverage predictive features and isolate true structural breaks from recording artifacts.

### 1.3 Expected Decision-Support Value
By translating multi-decadal agricultural panel data into descriptive and visual insights, this EDA blueprint empowers:
- **Agribusiness Enterprises & Processors**: Optimizing physical procurement timing, geographic sourcing allocations, and inventory carryover strategies.
- **Farmer Producer Organizations (FPOs)**: Identifying regional productivity gaps and mitigating post-harvest distress sales.
- **Financial & Policy Institutions**: Monitoring price stability, designing targeted buffer-stock interventions, and evaluating crop insurance benchmarks.

> **Methodological Disclaimer**: In accordance with the ethical research guidelines of this internship assignment, **no actual empirical findings or statistics are claimed or fabricated**. All numerical examples, statistical outputs, and expected insights presented herein are explicitly designated as **HYPOTHETICAL** analytical illustrations.

---

## 2. Business Context

Agribusiness decision-making requires transforming retrospective spatio-temporal data into forward-looking operational intelligence:

```
+--------------------------------------------------------------------------------------------------+
|                                Business Decision Support via EDA                                 |
+--------------------------+-----------------------------+-----------------------------------------+
| Production & Sourcing    | Trading & Risk Management   | Spatial & Policy Planning               |
| - Crop acreage response  | - Wholesale price spreads   | - Agro-climatic yield benchmarks        |
| - Intensive yield trends | - Rolling price volatility  | - Regional infrastructure deficits      |
| - Double-cropping index  | - Seasonal arrival troughs  | - Minimum Support Price (MSP) gaps      |
+--------------------------+-----------------------------+-----------------------------------------+
```

1. **Agricultural Production Planning**: Evaluating whether national output growth is driven by land expansion (extensive margin) or technological yield improvements (intensive margin).
2. **Crop Selection & Portfolio Allocation**: Assisting regional agricultural extension officers in recommending crop portfolios that balance high average yields with low inter-annual yield variance.
3. **Market Analysis & Price Monitoring**: Tracking price dispersion across regulated APMC mandis to detect spatial market integration or fragmentation.
4. **Seasonal Procurement Planning**: Anticipating harvest arrival peaks to schedule freight logistics, warehouse space, and working capital loans.
5. **Resource Allocation**: Identifying districts with lagging yields despite high irrigation access to target soil health interventions.
6. **Forecasting Preparation**: Determining the required order of seasonal differencing ($D=1, s=12$) and autoregressive lag structures before training predictive models.

---

## 3. EDA Objectives

The EDA framework systematically investigates eight core analytical questions:

```
+--------------------------------------------------------------------------------------------------+
| Core Analytical Questions                                                                        |
+--------------------------------------------------------------------------------------------------+
| 1. Regional Production Dominance: Which states and districts contribute the largest shares to    |
|    national production across staple foodgrains, oilseeds, and pulses?                           |
| 2. Spatial Yield Disparities: How significantly does crop yield (kg/ha) vary across agro-climatic|
|    zones, and which regions exhibit persistent productivity deficits?                            |
| 3. Seasonality Patterns: What regular seasonal cycles characterize monthly wholesale prices and  |
|    physical market arrivals between Kharif, Rabi, and Summer seasons?                            |
| 4. Longitudinal Price Trajectories: How have nominal and inflation-adjusted wholesale commodity |
|    prices trended over time across major regulated APMC mandis?                                  |
| 5. Commodity Price Volatility: Which agricultural commodities display the highest historical     |
|    price volatility and return dispersion?                                                       |
| 6. Acreage-Production Elasticity: What is the empirical relationship between cultivated area     |
|    and harvest volume? Does acreage response exhibit diminishing returns to scale?              |
| 7. Arrival-Price Transmission: To what degree do physical arrival gluts depress modal prices?   |
| 8. Feature Selection for Forecasting: Which lag variables, rolling moving averages, and seasonal |
|    dummies demonstrate the strongest statistical association with future prices?                |
+--------------------------------------------------------------------------------------------------+
```

---

## 4. Dataset Overview

The exploratory analysis integrates longitudinal crop panels and high-frequency wholesale market records:

| Variable Name | Description | Data Type | Standard Unit | Analytical Purpose in EDA |
| :--- | :--- | :--- | :--- | :--- |
| `date` | Calendar observation or transaction date | `datetime64[ns]` | ISO 8601 (`YYYY-MM-DD`) | Longitudinal sorting, time-series indexing |
| `year` | Agricultural or calendar year | `int64` | Dimensionless | Multi-decadal trend analysis, YoY growth rates |
| `month` | Calendar month (1 to 12) | `int64` | Dimensionless | Monthly seasonality and cycle decomposition |
| `state_name` | Administrative state / province | `category` | Normalized Text | Macro-level regional benchmarking |
| `district_name` | Sub-national administrative district | `category` | Normalized Text | Micro-regional yield and production clustering |
| `market_name` | Regulated APMC wholesale market (mandi) | `category` | Normalized Text | Market-level price dispersion and liquidity |
| `crop` | Canonical crop name (e.g., Rice, Wheat) | `category` | Canonical Ontology | Commodity-specific filtering and comparison |
| `crop_category` | Taxonomic category (Cereals, Pulses, etc.) | `category` | Broad Classification | Portfolio aggregation and sector allocation |
| `season` | Cropping season (Kharif, Rabi, Zaid, Annual)| `category` | Standardized Season | Seasonal output and cropping cycle analysis |
| `area` | Harvested / cultivated land surface | `float64` | Hectares (ha) | Land utilization, input elasticity modeling |
| `production` | Aggregate physical output volume | `float64` | Metric Tonnes (t) | Total supply volume, national contribution |
| `yield_kg_per_ha` | Land productivity ratio ($\frac{P \times 1000}{A}$) | `float64` | kg / Hectare (kg/ha) | Land efficiency, intensive technology adoption |
| `modal_price` | Most frequent wholesale trading price | `float64` | INR / Quintal (100 kg) | Representative market clearing benchmark |
| `price_change_pct` | Relative price return ($\frac{P_t - P_{t-1}}{P_{t-1}} \times 100$) | `float64` | Percentage (%) | Price momentum, return distribution modeling |
| `rainfall_mm` | Cumulative seasonal precipitation | `float64` | Millimeters (mm) | Weather sensitivity and yield correlation |

---

## 5. Univariate Analysis

Univariate profiling examines the statistical properties of individual variables in isolation before testing multivariate relationships.

### 5.1 Numerical Variables Profiling
For continuous variables (`area`, `production`, `yield_kg_per_ha`, `modal_price`), both parametric and non-parametric metrics are computed:
- **Measures of Central Tendency**:
  - *Mean ($\mu$)*: Sensitive to extreme values; useful for aggregate balance sheets.
  - *Median ($Q_2$)*: Robust measure of typical performance in skewed distributions.
- **Measures of Dispersion**:
  - *Standard Deviation ($\sigma$)* & *Variance ($\sigma^2$)*: Quantifies absolute volatility.
  - *Interquartile Range ($IQR = Q_3 - Q_1$)*: Non-parametric spread unaffected by tail outliers.
  - *Minimum & Maximum*: Bounding extreme observed operational ranges.
- **Distributional Shape**:
  - *Fisher-Pearson Skewness ($S$)*: Identifies right-skewed output distributions ($S > 0$) where a few mega-districts produce the bulk of output.
  - *Kurtosis ($K$)*: Evaluates fat-tailed risks (leptokurtosis, $K > 3$), common in perishable crop prices.

### 5.2 Categorical Variables Profiling
For discrete variables (`state_name`, `district_name`, `crop`, `season`):
- **Cardinality**: Total count of unique categories.
- **Frequency Distribution**: Value counts and percentage shares of total observations.
- **Top Category Dominance**: Identifies whether reporting is dominated by specific states or crops.

---

## 6. Production Analysis

Production analysis evaluates the volume of agricultural supply across dimensions:

```
+--------------------------------------------------------------------------------------------------+
|                                    Production Analysis Plan                                      |
+--------------------------+-----------------------------+-----------------------------------------+
| Aggregation Level        | Analytical Technique        | Planned Visualisation                   |
+--------------------------+-----------------------------+-----------------------------------------+
| National Output by Crop  | Multi-decadal volume sum    | Stacked area chart / horizontal bar     |
| Spatial State Share      | Percentage share of total   | Treemap / Pareto bar chart              |
| District Concentration   | Lorenz curve & Gini index   | District choropleth heatmap             |
| Longitudinal Trajectory  | Compound Annual Growth Rate | Multi-line time series with trend line  |
+--------------------------+-----------------------------+-----------------------------------------+
```

### Key Analytical Metrics:
1. **Year-over-Year Production Growth Rate ($YoY_P$)**:
   $$\text{YoY Growth } (\%) = \left( \frac{P_{c, t} - P_{c, t-1}}{P_{c, t-1}} \right) \times 100$$
2. **Regional Production Share ($Share_r$)**:
   $$\text{Production Share } (\%) = \left( \frac{\sum P_{r, c}}{\sum P_{\text{total}, c}} \right) \times 100$$

---

## 7. Yield Analysis

### 7.1 Yield as a Critical Business Metric
While total production dictates total market supply, **Crop Yield ($Y = \frac{P \times 1000}{A}$)** reflects underlying agronomic efficiency, soil health, irrigation access, and technology adoption. Production can expand simply by clearing more land (extensive growth), but sustainable agribusiness competitiveness depends entirely on intensive yield growth.

```
+--------------------------------------------------------------------------------------------------+
| [HYPOTHETICAL COMPARATIVE EXAMPLE]                                                               |
| Region A: Area = 10,000 ha, Production = 20,000 tonnes  -> Yield = 2,000 kg/ha                   |
| Region B: Area =  2,500 ha, Production = 10,000 tonnes  -> Yield = 4,000 kg/ha                   |
|                                                                                                  |
| Business Implication:                                                                            |
| Region A produces double the volume of Region B, but operates at HALF the productivity.          |
| Sourcing from Region B yields higher logistics efficiency and lower land degradation risk.       |
+--------------------------------------------------------------------------------------------------+
```

### 7.2 Analytical Steps for Yield
1. **Crop Productivity Benchmarking**: Rank crops by national average yield and coefficient of variation ($CV = \frac{\sigma}{\mu} \times 100$).
2. **Spatial Yield Index ($SYI$)**:
   $$SYI_{r, c} = \left( \frac{\bar{Y}_{r, c}}{\bar{Y}_{\text{national}, c}} \right) \times 100$$
3. **Productivity Frontier Analysis**: Plotting the 90th percentile district yields against the median to identify the regional technology gap.

---

## 8. Regional Analysis

Agricultural performance in sub-continental economies is characterized by sharp agro-climatic disparities. The regional analysis segments data across:
1. **State-Level Comparisons**: Evaluating macro-policy impacts (e.g., procurement infrastructure in Punjab/Haryana vs. eastern states).
2. **District-Level Clustering**: Utilizing unsupervised grouping (k-means) based on Area, Production, and Yield to identify high-productivity belts versus low-yield vulnerability zones.
3. **Mandi-Level Comparisons**: Comparing market arrival volumes and price dispersion across major terminal markets vs. primary rural assembly mandis.

---

## 9. Seasonal Analysis

### 9.1 Cropping Seasons and Price Cyclicality
Agricultural markets follow biological seasons that induce predictable economic cycles:
- **Kharif Season (Monsoon Sowing: June–July; Harvest: Oct–Nov)**: Characterized by massive supply surges in paddy, maize, soybean, and pulses, frequently depressing wholesale prices.
- **Rabi Season (Winter Sowing: Oct–Nov; Harvest: March–April)**: Wheat, mustard, and gram harvests clearing through APMC markets.
- **Zaid / Summer (March–June)**: Short-duration pulses and horticultural crops.

### 9.2 Monthly Seasonal Price Index ($SPI_m$)
To extract seasonal cycles from underlying secular trends:
$$SPI_m = \left( \frac{\bar{Price}_m}{\bar{Price}_{\text{annual}}} \right) \times 100$$

Planned visualisations include **Seasonal Sub-series Plots** (showing monthly price distributions across 10 years) and **Calendar Heatmaps** (Months on x-axis, Years on y-axis, colored by modal price).

---

## 10. Market Price Analysis

### 10.1 Key Price Metrics
Wholesale market prices are evaluated across five dimensions:
1. **Average & Median Price**: Central tendency benchmarks.
2. **Price Spread**: Difference between daily maximum and minimum bids ($\Delta = P_{\max} - P_{\min}$), reflecting intraday volatility and quality variation.
3. **Percentage Price Return**:
   $$R_t = \left( \frac{P_t - P_{t-1}}{P_{t-1}} \right) \times 100$$
4. **Annualized Historical Volatility ($\sigma_{\text{ann}}$)**:
   $$\sigma_{\text{ann}} = \text{std}\left(\ln\left(\frac{P_t}{P_{t-1}}\right)\right) \times \sqrt{252}$$
5. **Crop-wise & Market-wise Dispersion**: Comparing staple cereal price stability against high-beta horticultural crops (onions, potatoes).

---

## 11. Distribution Analysis

To determine appropriate econometric modeling specifications, continuous variables are profiled for distributional characteristics:

```
+--------------------------------------------------------------------------------------------------+
| Visualisation Type | Target Variables                   | Analytical Purpose                     |
+--------------------+------------------------------------+----------------------------------------+
| Histogram + KDE    | Crop Yield (kg/ha), Production (t) | Evaluate skewness, bimodal clustering  |
| Box Plot           | Modal Price (INR/q) by Season      | Identify interquartile spread & tails  |
| Log-Density Plot   | Mandi Arrival Volumes              | Normalize extreme positive skewness    |
| Q-Q Plot           | Residual price returns             | Test Gaussian normality assumptions    |
+--------------------------------------------------------------------------------------------------+
```

---

## 12. Bivariate Analysis

Bivariate analysis evaluates statistical associations between paired agricultural variables:

```
+--------------------------------------------------------------------------------------------------+
| Variable Pair           | Expected Relationship          | Visualisation Method                  |
+-------------------------+--------------------------------+---------------------------------------+
| Area vs. Production     | Strong positive linear         | Scatter plot with OLS trendline       |
| Area vs. Yield          | Weak / Non-linear (diminishing)| Scatter plot with Lowess curve        |
| Production vs. Yield    | Moderate positive              | Scatter plot grouped by crop          |
| Arrivals vs. Price      | Negative (supply depression)   | Scatter plot with hyperbolic fit      |
| Rainfall vs. Yield      | Positive up to optimal ceiling | Scatter plot with polynomial fit      |
+--------------------------------------------------------------------------------------------------+
```

> **Core Econometric Principle**: **Correlation does not imply causation**. A strong statistical correlation between cultivated area and aggregate production does not prove that allocating more land automatically causes proportional output increases, especially if marginal lands suffer from soil salinity or lack irrigation.

---

## 13. Multivariate Analysis

To assess interactions across multiple dimensions simultaneously:
1. **Pivot Tables & Multi-Index Summaries**: Slicing mean yields across State $\times$ Season $\times$ Crop.
2. **Faceted Visualisations**: Small-multiple scatter plots comparing Area vs. Production faceted by Crop Category (`Cereals`, `Pulses`, `Oilseeds`).
3. **Three-Way Interaction Plots**: Plotting Production vs. Year, colored by State, with bubble size proportional to Cultivated Area.

---

## 14. Correlation Analysis

### 14.1 Pearson vs. Spearman Formulation
- **Pearson Linear Correlation ($r$)**: Measures linear relationships between continuous, normally distributed variables:
  $$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
- **Spearman Rank Correlation ($\rho$)**: Measures monotonic relationships based on ranked values; robust to non-normal distributions and extreme outliers:
  $$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

### 14.2 Proposed Correlation Matrix Blueprint
The matrix below outlines the proposed pairwise structure evaluated during full empirical execution:

| Variable | Area (ha) | Production (t) | Yield (kg/ha) | Modal Price | Price Volatility | Rainfall (mm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Area (ha)** | 1.00 | Proposed | Proposed | Proposed | Proposed | Proposed |
| **Production (t)** | Proposed | 1.00 | Proposed | Proposed | Proposed | Proposed |
| **Yield (kg/ha)** | Proposed | Proposed | 1.00 | Proposed | Proposed | Proposed |
| **Modal Price** | Proposed | Proposed | Proposed | 1.00 | Proposed | Proposed |
| **Price Volatility**| Proposed | Proposed | Proposed | Proposed | 1.00 | Proposed |
| **Rainfall (mm)** | Proposed | Proposed | Proposed | Proposed | Proposed | 1.00 |

*(Actual coefficients will be calculated on clean empirical panels in Week 4; no synthetic correlation values are asserted)*

---

## 15. Statistical Anomalies

Building upon the Week 2 cleaning architecture, anomalies identified during exploratory analysis will be audited rather than discarded:

```
+--------------------------------------------------------------------------------------------------+
| Anomaly Type         | Detection Technique             | Diagnostic Decision Action              |
+----------------------+---------------------------------+-----------------------------------------+
| Extreme High Yield   | Grouped IQR ($> Q_3 + 3.0 IQR$) | Check biological ceiling; verify if     |
|                      |                                 | high-yield variety or unit mismatch     |
+----------------------+---------------------------------+-----------------------------------------+
| Near-Zero Yield      | Yield $< 50\text{ kg/ha}$       | Check historical drought records; retain|
|                      |                                 | as genuine climatic shock if confirmed  |
+----------------------+---------------------------------+-----------------------------------------+
| Price Spike          | Rolling Z-Score ($|z| > 3.5$)   | Check trade policy / export bans; retain|
|                      |                                 | with structural break indicator dummy   |
+----------------------+---------------------------------+-----------------------------------------+
| Arrival Gluts        | Daily arrival $> 5 \times$ mean | Check peak post-harvest harvest window; |
|                      |                                 | validate against physical mandi capacity|
+--------------------------------------------------------------------------------------------------+
```

---

## 16. Visualisation Plan

The detailed visualization blueprint below defines the core charts produced during the analysis:

| Analytical Question | Chart Type | Dimensions & Variables | Purpose |
| :--- | :--- | :--- | :--- |
| **Which crops dominate aggregate volume?** | Horizontal Bar Chart | `crop` vs. `production` | Rank commodity volumes with labeled data values |
| **How are crop yields distributed?** | Histogram + KDE | `yield_kg_per_ha` | Diagnose skewness, kurtosis, and modal efficiency |
| **How severe is regional yield dispersion?** | Box Plot | `state_name` vs. `yield_kg_per_ha` | Contrast median yields, IQR spread, and outliers |
| **What is the acreage-production elasticity?**| Scatter Plot + OLS Fit | `area` vs. `production` | Evaluate returns to scale and linear regression slope |
| **How do wholesale prices trend over time?** | Multi-Line Chart | `date` vs. `modal_price` (7-day MA) | Track secular trends and volatility cycles |
| **What is the seasonal price pattern?** | Calendar Heatmap | `month` $\times$ `year` colored by Price | Pinpoint recurring seasonal harvest price depressions |
| **How do continuous features associate?** | Correlation Heatmap | Pairwise Pearson/Spearman matrix | Identify multicollinearity and predictive features |

---

## 17. KPI Framework

The table below formalizes the core Agribusiness Key Performance Indicators:

| KPI Name | Mathematical Formula | Business Meaning & Significance |
| :--- | :--- | :--- |
| **Total Production ($P$)** | $\sum P_i \quad [\text{Metric Tonnes}]$ | Aggregate commodity supply available for consumption and trade |
| **Average Yield ($Y$)** | $\frac{\sum P \times 1000}{\sum A} \quad [\text{kg/ha}]$ | Fundamental benchmark of land productivity and agronomic efficiency |
| **Production Growth Rate** | $\left(\frac{P_t - P_{t-1}}{P_{t-1}}\right) \times 100 \quad [\%]$ | Annual supply expansion rate; monitors national food security |
| **Yield Growth Rate** | $\left(\frac{Y_t - Y_{t-1}}{Y_{t-1}}\right) \times 100 \quad [\%]$ | Intensive technological adoption rate independent of land expansion |
| **Average Modal Price** | $\frac{\sum P_{\text{modal}}}{N} \quad [\text{INR/q}]$ | Baseline clearing price benchmark across wholesale markets |
| **Price Volatility ($\sigma_{\text{ann}}$)**| $\text{std}(\ln(P_t / P_{t-1})) \times \sqrt{252}$ | Quantifies financial procurement risk and required hedging buffers |
| **Cultivated Area ($A$)** | $\sum A_i \quad [\text{Hectares}]$ | Land resource allocation reflecting farmer planting intentions |
| **Spatial Yield Index ($SYI$)**| $\left(\frac{Y_{\text{region}}}{Y_{\text{national}}}\right) \times 100 \quad [\%]$ | Relative regional competitive advantage in commodity production |

---

## 18. Insight Generation Framework

To ensure that exploratory findings translate into operational value, the framework applies a structured four-stage reasoning model:

```
Observation (Empirical Metric)
       ↓
Pattern (Contextual Trend)
       ↓
Business Interpretation (Economic Impact)
       ↓
Potential Action (Decision Recommendation)
```

```
+--------------------------------------------------------------------------------------------------+
| [HYPOTHETICAL DEMONSTRATION OF INSIGHT GENERATION]                                               |
|                                                                                                  |
| 1. Observation:                                                                                  |
|    District X exhibits a Spatial Yield Index (SYI) of 145% for Wheat, but wholesale modal prices |
|    consistently trade 18% below the state average during April-May.                             |
|                                                                                                  |
| 2. Pattern:                                                                                      |
|    High physical productivity coincides with acute post-harvest arrival gluts and severe local   |
|    market price depression.                                                                      |
|                                                                                                  |
| 3. Business Interpretation:                                                                      |
|    Local APMC mandis lack adequate warehouse storage and processing infrastructure, forcing      |
|    farmers into distress selling upon harvest.                                                   |
|                                                                                                  |
| 4. Potential Action:                                                                             |
|    Agribusiness procurement teams should establish direct farmgate procurement hubs in District  |
|    X, securing high-quality grain at competitive costs while offering farmers transparent        |
|    premiums over local distressed mandi rates.                                                   |
+--------------------------------------------------------------------------------------------------+
```

---

## 19. Expected Business Insights

Based on agribusiness economic principles, the following patterns are hypothesized for empirical validation in subsequent project phases:
1. **Spatial Productivity Clusters**: High cereal yields are anticipated to cluster heavily in northwestern irrigated belts (Punjab, Haryana, Western UP), whereas rainfed central regions will exhibit lower mean yields with higher coefficient of variation.
2. **Harvest Price Depression**: Wholesale prices for major Kharif pulses (chickpea, pigeonpea) are expected to display systematic price troughs in October–December, lagging peak market arrival surges by 1 to 3 weeks.
3. **Acreage Stagnation with Yield Growth**: Cereal production over the last 15 years is hypothesized to reflect flat cultivated land allocation offset by positive intensive yield growth.
4. **Asymmetric Horticultural Volatility**: Perishable commodities (onion, tomato, potato) will exhibit annualized historical volatilities exceeding $40\%$, characterized by sharp positive skewness during monsoon supply disruptions.

---

## 20. Final EDA Report Structure

The final comprehensive EDA deliverable will be compiled according to the following 17-part structure:

1. **Executive Summary**: Overview of study scope, data sources, and top strategic insights.
2. **Business Context & Problem Statement**: Stakeholder value mapping and decision objectives.
3. **Dataset Overview & Metadata**: Complete catalog of variables, time horizons, and coverage.
4. **Data Cleaning & Quality Audit**: Summary of missingness treatments, deduplication, and scorecards.
5. **National Agricultural Production Profiling**: Aggregate volume trends and crop diversification.
6. **Crop Yield Dynamics & Spatial Disparities**: Intensive productivity benchmarking and SYI rankings.
7. **Regional & District-Level Performance**: Spatial clustering and leading vs. lagging zones.
8. **Seasonal Patterns & Cropping Cycles**: Monthly indices, calendar heatmaps, and arrival timing.
9. **Market Price Dynamics & Volatility**: Modal price spreads, inflation trends, and volatility rankings.
10. **Distributional Analysis & Normality Tests**: Skewness, kurtosis, and transformations.
11. **Bivariate Relationships & Elasticities**: Scatter plots, regression fits, and scale returns.
12. **Multivariate Interactions**: Grouped cross-tabulations and faceted visual analytics.
13. **Correlation Analysis**: Pearson linear vs. Spearman rank correlation matrices.
14. **Anomaly Diagnosis & Structural Breaks**: Categorization of extreme climatic and trade shocks.
15. **Synthesis of Strategic Business Insights**: Actionable takeaways for agribusiness stakeholders.
16. **Limitations & Methodological Constraints**: Data gaps, causality warnings, and assumptions.
17. **Appendix & Technical Documentation**: Python environment specifications and full code snippets.

---

## 21. Limitations

1. **Data Availability & Publication Lags**: Official district-level production statistics often experience an 18-to-24-month administrative reporting lag.
2. **Historical Administrative Boundary Shifts**: Bifurcation of districts over time creates panel discontinuities requiring spatial concordance crosswalks.
3. **Correlation vs. Causation**: Statistical association between weather variables, area, and yields does not establish direct agronomic causation without controlled agronomic trial data.
4. **Reporting Heterogeneity in Mandis**: Smaller rural APMC mandis report prices intermittently, requiring temporal interpolation that may smooth true localized volatility.
5. **Sampling Bias in Crop-Cutting Experiments**: Historical yield figures are based on statistical sampling plots that may under-represent marginal farm holdings.

---

## 22. Week 4 Handoff

The outputs generated in Week 3 provide the foundation for Week 4 Predictive Modeling and Forecasting:

```
+--------------------------------------------------------------------------------------------------+
|                                    Week 4 Handoff Deliverables                                   |
+--------------------------+-----------------------------------------------------------------------+
| 1. EDA Demonstration     | Tested `notebooks/01_agribusiness_eda.ipynb` showcasing full workflow |
+--------------------------+-----------------------------------------------------------------------+
| 2. Visualisation Library | Standardized plotting routines in `src/visualisation.py`              |
+--------------------------+-----------------------------------------------------------------------+
| 3. Feature Candidate Set | Engineered candidate predictors (lags, moving averages, SYI)         |
+--------------------------+-----------------------------------------------------------------------+
| 4. Stationarity Clues    | Identified seasonal periods ($s=12$) and secular trend trajectories   |
+--------------------------+-----------------------------------------------------------------------+
| 5. Anomaly Audit Log     | Catalog of validated extreme events for dummy variable inclusion      |
+--------------------------+-----------------------------------------------------------------------+
| 6. Modeling Strategy     | Recommendation of ARIMA/SARIMA for prices and Panel ML for production |
+--------------------------+-----------------------------------------------------------------------+
```

---

## 23. Python Implementation Plan

The representative script below illustrates the standardized Pandas, NumPy, Matplotlib, and Seaborn workflow executed in `src/eda.py` and `src/visualisation.py`:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load clean analytical dataset
df = pd.read_parquet("data/processed/crop_apy_clean.parquet")

# 2. Descriptive statistics profiling
stats_summary = df[['area', 'production', 'yield_kg_per_ha', 'modal_price']].describe().T
stats_summary['skewness'] = df[['area', 'production', 'yield_kg_per_ha', 'modal_price']].skew()
stats_summary['iqr'] = stats_summary['75%'] - stats_summary['25%']

# 3. Grouped aggregations across crops and regions
crop_performance = df.groupby('crop').agg({
    'production': 'sum',
    'yield_kg_per_ha': 'mean',
    'modal_price': 'mean'
}).sort_values('production', ascending=False)

# 4. Correlation matrix computation
corr_pearson = df[['area', 'production', 'yield_kg_per_ha', 'modal_price']].corr(method='pearson')
corr_spearman = df[['area', 'production', 'yield_kg_per_ha', 'modal_price']].corr(method='spearman')

# 5. Visualizing yield distributions and boxplots
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='crop', y='yield_kg_per_ha', palette='Set2')
plt.title("Crop Productivity Dispersion Across Major Agricultural Commodities")
plt.xlabel("Crop")
plt.ylabel("Yield (kg/ha)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Bivariate scatter plot with regression fit
plt.figure(figsize=(9, 6))
sns.regplot(data=df, x='area', y='production', scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'})
plt.title("Cultivated Area vs. Aggregate Production Volume")
plt.xlabel("Area (Hectares)")
plt.ylabel("Production (Metric Tonnes)")
plt.tight_layout()
plt.show()
```
