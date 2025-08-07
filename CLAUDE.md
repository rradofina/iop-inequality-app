# IOP Inequality Analysis Tool - Project Documentation

## Project Overview
This Streamlit web application was developed for the **Asian Development Bank (ADB) Economic Research Division** to provide National Statistical Offices (NSOs) across the Asia-Pacific region with an accessible tool for conducting Inequality of Opportunity (IOP) analysis without requiring R programming expertise.

## GitHub Repository
🔗 **Repository**: https://github.com/rradofina/iop-inequality-app.git

## Purpose & Motivation
- **Target Users**: National Statistical Offices (NSOs) in Asia-Pacific region
- **Problem Solved**: NSOs often lack R programming expertise, making IOP analysis challenging
- **Solution**: User-friendly web interface that replicates R methodology while allowing users to focus on results rather than coding
- **Organization**: Asian Development Bank - Economic Research Division

## What is Inequality of Opportunity (IOP)?
IOP analysis distinguishes between two sources of inequality:
1. **Circumstances**: Factors beyond individual control (birthplace, parental education, gender, ethnicity)
2. **Effort**: Individual choices, talent, and hard work

The key insight: While some inequality may be acceptable (due to different effort levels), inequality due to circumstances is considered unfair and should be addressed through policy.

### Why This Matters for Policy
- **Fairness Principle**: Everyone should have equal opportunities regardless of circumstances they cannot control
- **Policy Targeting**: Identifies which circumstances contribute most to inequality
- **Resource Allocation**: Helps prioritize interventions (e.g., rural education if Birth Area is the main driver)
- **Development Goals**: Aligns with SDGs and inclusive growth objectives

## Key Features Implemented

### 1. Core IOP Analysis Methods
- **C-Tree (Conditional Inference Trees)**: Partitions population into types based on circumstances
- **C-Forest (Random Forest)**: Ensemble method for robust IOP estimation
- **Shapley Decomposition**: Decomposes inequality contributions by each circumstance variable

### 2. Data Processing Pipeline
- **Data Upload**: Supports CSV and Excel files
- **Age Adjustment**: Regression-based age adjustment (matching R implementation)
- **Variable Transformation**: 
  - Sex: 2→0 (female), 1→1 (male)
  - Religion: Groups minorities (7,8,9→6)
  - Filters out invalid parental education values
- **Weighted Analysis**: Full support for sample weights

### 3. CPI-PPP Adjustment (New Feature)
- **Purpose**: Converts local currency to international dollars for cross-country comparisons
- **Implementation**: 
  - Reads CPI-PPP reference file (Excel with CPI values 1960-2022, base year 2017=100)
  - Auto-detects country code and year from data
  - Applies formula: `factor = ppp × cpi / 100`
  - Adjusts income: `income_adjusted = income / factor`
- **UI Elements**:
  - Separate file uploader for CPI-PPP reference
  - Toggle to enable/disable adjustment
  - Displays adjustment metrics (country, year, CPI, PPP factor)

### 4. AI-Powered Insights (Groq Integration)
- **Models Available**:
  - GPT-OSS-120B (Primary - Best reasoning for detailed policy analysis)
  - GPT-OSS-20B (Secondary - Faster with excellent quality-speed balance)
  - Llama 3.3 70B (Alternative - Quick balanced analysis)
  - Mixtral 8x7B (Lightweight - Fast basic insights)
- **Connection Flow**:
  - Enter Groq API key in sidebar
  - Click "Connect to AI Model" button (directly below API key input)
  - Once connected, AI analysis runs automatically when results are generated
  - No manual buttons needed in results section
- **Analysis Output** (Designed for NSOs and policymakers):
  - **Simple Explanation of IOP Percentages**: What it means in layman terms (e.g., "41% of inequality is due to circumstances beyond individual control")
  - **Most Important Circumstances**: Detailed explanation of why each matters (e.g., Birth Area reflecting urban-rural divides, Father's Education affecting human capital transmission)
  - **Policy Implications**: Specific, actionable recommendations for each circumstance
  - **Notable Patterns**: Comparison with international benchmarks, surprising findings
- **Tailored for High-Level Audiences**: 
  - Written for government officials and development partners
  - Balances technical accuracy with accessibility
  - Focuses on actionable insights over methodology

### 5. Visualization & Export
- **Tree Visualizations**: 
  - Static tree diagrams
  - Interactive sunburst charts
  - Text-based rules export
- **Charts**: 
  - Type distributions
  - Feature importance plots
  - Shapley value contributions
- **Export Options**: JSON, CSV, TXT formats

## Technical Implementation

### Python vs R Methodology Comparison
- **R Implementation**: Uses `partykit::ctree` (Conditional Inference Trees with statistical testing)
- **Python Implementation**: Uses `sklearn.DecisionTreeRegressor` (CART algorithm)
- **Differences**: 
  - Different tree algorithms but same core IOP methodology
  - Results typically differ by 2-5 percentage points
  - Both approaches are academically accepted

### Inequality Measures
- **Gini Coefficient**: Weighted implementation matching R's `dineq::gini.wtd`
- **Mean Log Deviation (MLD)**: Weighted implementation matching R's `dineq::mld.wtd`
- **Relative IOP**: `IOP = I(y_tilde) / I(y)` where y_tilde is smoothed income

### Advanced Parameters (Matching R)
- **C-Tree Parameters**:
  - `mincriterion`: Significance level for splits (default 0.99)
  - `minbucket`: Minimum observations in terminal nodes
  - `maxdepth`: Maximum tree depth
- **C-Forest Parameters**:
  - `n_estimators`: Number of trees
  - `max_features`: Features per split
  - `min_samples_leaf`: Minimum samples in leaf nodes

## Data Requirements
- **Required Columns**:
  - `income`: Income values (positive)
  - `cntry`: 3-letter country code (e.g., "NPL")
  - `year`: Year of survey
- **Circumstance Variables** (optional but recommended):
  - `Sex`: Gender
  - `Father_Edu`: Father's education level
  - `Mother_Edu`: Mother's education level
  - `Birth_Area`: Birth region
  - `Religion`: Religious affiliation
- **Optional**:
  - `weights`: Sample weights
  - `age`: For age adjustment

## Deployment & Usage

### Local Installation
```bash
# Clone repository
git clone https://github.com/rradofina/iop-inequality-app.git
cd iop-inequality-app

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Cloud Deployment
- Can be deployed on Streamlit Cloud (free)
- Docker-ready for containerized deployment
- Suitable for institutional servers

## Benefits for NSOs
1. **No Programming Required**: Point-and-click interface
2. **Immediate Results**: Real-time analysis and visualization
3. **Standardized Methodology**: Ensures consistency across countries
4. **International Comparability**: Built-in PPP adjustment
5. **Policy-Ready Output**: AI-generated insights and recommendations
6. **Training-Friendly**: Intuitive interface reduces training needs

## Project Impact
This tool democratizes IOP analysis for NSOs across Asia-Pacific, enabling evidence-based policy making on inequality without technical barriers. By removing the R programming requirement, NSOs can focus on interpreting results and developing policies rather than struggling with code.

## Support & Maintenance
- **Organization**: Asian Development Bank - Economic Research Division
- **Repository**: https://github.com/rradofina/iop-inequality-app
- **Documentation**: Available in repository README and TECHNICAL_COMPARISON.md

## Recent Updates (Latest Session)
- Added CPI-PPP adjustment functionality for international comparisons
- Integrated GPT-OSS-20B as secondary AI model option
- Added Test AI Connection button for immediate API validation
- Improved model hierarchy labeling (Primary/Secondary/Alternative)
- Enhanced PPP adjustment UI with detailed metrics display
- Increased AI token limits to 2500 for complete responses
- Fixed HTML tags appearing in AI outputs with markdown-only instructions
- Updated all prompts to use circumstance-effort framework terminology
- Added sample CPI-PPP file option for easy testing
- Improved policy-friendly language throughout AI insights
- Fixed sample file loading issues by consolidating in data directory
- **Added Shapley decomposition to Summary tab** for better policy insights
- Shapley values now sorted by importance with visual bar chart

---
*This tool represents a significant step forward in making advanced inequality analysis accessible to statistical offices throughout the Asia-Pacific region, supporting ADB's mission of promoting inclusive economic growth.*