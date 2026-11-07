# Agribusiness EDA Data Directory & Variable Dictionary

## 1. Directory Purpose and Structure

This directory manages datasets for the **Agribusiness Exploratory Data Analysis (EDA)** project (Week 3).

```
data/
├── README.md                 # Variable dictionary, schemas, and storage rules
├── raw/                      # Unaltered source data snapshots (read-only)
│   └── .gitkeep
└── processed/                # Standardized, cleaned panels ready for EDA
    └── .gitkeep
```

---

## 2. Expected Dataset Variables & Schema Specification

The table below defines the target variables analyzed across the exploratory workflows:

| Variable Name | Description | Target Data Type | Standard Unit | Analytical Role in EDA |
| :--- | :--- | :--- | :--- | :--- |
| `date` | Calendar transaction or observation date | `datetime64[ns]` | ISO 8601 (`YYYY-MM-DD`) | Time-series indexing, temporal ordering |
| `year` | Calendar year of harvest or market trade | `int64` | Dimensionless | Annual trend analysis, YoY growth rates |
| `month` | Month of observation (1 to 12) | `int64` | Calendar month | Monthly seasonality, cyclical pattern detection |
| `state_name` | State or provincial administrative region | `category` | Normalized text | Spatial comparison, macro-regional profiling |
| `district_name` | Sub-national district or county | `category` | Normalized text | Micro-regional yield and production clustering |
| `market_name` | Regulated wholesale market / APMC mandi | `category` | Normalized text | Market-level price dispersion, arrival tracking |
| `crop` | Canonical crop name (e.g. Rice, Wheat) | `category` | Canonical ontology | Commodity comparisons, production ranking |
| `crop_category` | Macro grouping (Cereals, Pulses, Oilseeds) | `category` | Broad classification | Portfolio allocation and diversification analysis |
| `season` | Cropping season (Kharif, Rabi, Zaid, Annual)| `category` | Standard season | Seasonal production and price response |
| `area` | Harvested / cultivated land surface | `float64` | Hectares (ha) | Land utilization, input elasticity |
| `production` | Aggregate physical output volume | `float64` | Metric Tonnes (t) | Supply volume, regional output share |
| `yield_kg_per_ha` | Land productivity efficiency ($\frac{P \times 1000}{A}$) | `float64` | kg / Hectare (kg/ha) | Technological efficiency, intensive margin |
| `modal_price` | Most frequent wholesale trading price | `float64` | INR / Quintal (100 kg) | Clearing price, market valuation |
| `price_change_pct` | Relative price return: $\frac{P_t - P_{t-1}}{P_{t-1}} \times 100$ | `float64` | Percentage (%) | Price momentum, volatility quantification |
| `rainfall_mm` | Cumulative precipitation (where available) | `float64` | Millimeters (mm) | Weather sensitivity and yield correlation |

---

## 3. Data Integrity & Immutability Rules

1. **Read-Only Raw Data**: Files placed in `data/raw/` are treated as immutable baseline snapshots.
2. **Analysis-Ready Processed Zone**: Tables in `data/processed/` must be generated programmatically from validated scripts.
3. **Hypothetical Flagging**: Any demonstration data generated within notebooks for pedagogical purposes is explicitly flagged as synthetic/hypothetical.
