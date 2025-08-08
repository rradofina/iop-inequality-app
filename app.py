"""
IOP Inequality Analysis Tool - Streamlit App
Based on ADB Workshop methodology for ex-ante inequality of opportunity
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
import warnings
from itertools import combinations
from math import factorial
import time
import io
import json
import os

warnings.filterwarnings('ignore')

# Try to import Groq for AI insights
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="IOP Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Increase sidebar width only when expanded */
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 400px !important;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] > div {
        width: 400px !important;
    }
    
    /* Reduce spacing after sliders */
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Larger tab pills */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 30px;
        padding: 15px 25px;
        font-weight: 700;
        font-size: 1.1rem !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Larger metric labels */
    [data-testid="metric-container"] {
        margin: 0.5rem 0;
    }
    [data-testid="metric-container"] label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #333 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Larger headers and subheaders */
    .stMarkdown h2 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stMarkdown h3 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .stMarkdown h4 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    
    /* Larger multiselect and selectbox labels */
    .stMultiSelect label, .stSelectbox label {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #333 !important;
    }
    
    /* Better table spacing */
    .dataframe {
        font-size: 1.05rem !important;
    }
    .dataframe th {
        font-weight: 700 !important;
        padding: 10px !important;
    }
    .dataframe td {
        padding: 8px !important;
    }
    
    /* Tree rule styling */
    .tree-rule {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 12px 18px;
        margin: 12px 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 1rem;
    }
    
    /* Larger button text */
    .stButton button {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def weighted_gini(values, weights=None):
    """Calculate weighted Gini coefficient."""
    if weights is None:
        weights = np.ones(len(values))
    
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    sorted_weights = sorted_weights / sorted_weights.sum()
    
    cum_weights = np.cumsum(sorted_weights)
    cum_weighted_values = np.cumsum(sorted_values * sorted_weights)
    
    if len(values) == 0 or cum_weighted_values[-1] == 0:
        return 0
    
    gini = 1 - 2 * np.sum(cum_weighted_values[:-1] * sorted_weights[1:]) / cum_weighted_values[-1]
    gini = gini - sorted_weights[-1]
    
    return max(0, min(1, gini))

def weighted_mld(values, weights=None):
    """Calculate weighted Mean Log Deviation."""
    if weights is None:
        weights = np.ones(len(values))
    
    mask = values > 0
    values = values[mask]
    weights = weights[mask]
    
    if len(values) == 0:
        return 0
    
    weights = weights / weights.sum()
    mean_value = np.average(values, weights=weights)
    
    if mean_value == 0:
        return 0
    
    log_ratios = np.log(mean_value / values)
    mld = np.average(log_ratios, weights=weights)
    
    return mld

def weighted_mean(values, weights=None):
    """Calculate weighted mean."""
    if weights is None:
        return np.mean(values)
    return np.average(values, weights=weights)

# ============================================================================
# CPI-PPP ADJUSTMENT FUNCTIONS
# ============================================================================

@st.cache_data
def load_and_apply_ppp_adjustment(df, cpi_ppp_file):
    """
    Apply CPI-PPP adjustment to income values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with 'income', 'cntry', and 'year' columns
    cpi_ppp_file : UploadedFile
        Excel file with CPI values and PPP rates
    
    Returns:
    --------
    df : pd.DataFrame
        Data with adjusted income
    ppp_info : dict or None
        Information about the adjustment applied
    """
    try:
        # Load CPI-PPP reference file
        cpi_ppp_df = pd.read_excel(cpi_ppp_file)
        
        # Check if data has required columns
        if 'cntry' not in df.columns or 'year' not in df.columns:
            st.warning("⚠️ Data missing 'cntry' or 'year' columns for PPP adjustment")
            return df, None
        
        # Extract country and year from data
        country_code = df['cntry'].iloc[0] if isinstance(df['cntry'].iloc[0], str) else str(df['cntry'].iloc[0])
        year = int(df['year'].iloc[0])
        
        # Handle multiple countries warning
        if df['cntry'].nunique() > 1:
            st.warning(f"⚠️ Multiple countries detected. Using PPP adjustment for: {country_code}")
        
        # Find country row (column B is index 1)
        country_mask = cpi_ppp_df.iloc[:, 1] == country_code
        if not country_mask.any():
            st.error(f"❌ Country code '{country_code}' not found in CPI-PPP file")
            return df, None
        
        country_row = cpi_ppp_df[country_mask].iloc[0]
        
        # Get CPI value for the year
        # Years start from column D (index 3) for 1960
        # So year column index = year - 1960 + 3
        year_col_idx = year - 1960 + 3
        
        # Check if year is in valid range
        if year < 1960 or year > 2022:
            st.error(f"❌ Year {year} is outside the CPI data range (1960-2022)")
            return df, None
        
        try:
            cpi_value = float(country_row.iloc[year_col_idx])
        except:
            st.error(f"❌ Could not find CPI value for year {year}")
            return df, None
        
        # Get PPP value (column BO = index 66)
        try:
            ppp_value = float(country_row.iloc[66])
        except:
            st.error(f"❌ Could not find PPP value in column BO")
            return df, None
        
        # Calculate adjustment factor (same formula as R)
        cpi_ppp_factor = ppp_value * cpi_value / 100
        
        # Store original income and apply adjustment
        df['income_local'] = df['income'].copy()
        df['income'] = df['income'] / cpi_ppp_factor
        
        # Update log income if it exists
        if 'loginc' in df.columns:
            df['loginc'] = np.log(df['income'] + 1)
        
        # Get country name from column C (index 2)
        country_name = country_row.iloc[2]
        
        ppp_info = {
            'country_code': country_code,
            'country_name': country_name,
            'year': year,
            'cpi': cpi_value,
            'ppp': ppp_value,
            'factor': cpi_ppp_factor
        }
        
        return df, ppp_info
        
    except Exception as e:
        st.error(f"❌ Error processing CPI-PPP file: {str(e)}")
        return df, None

# ============================================================================
# DATA PREPARATION
# ============================================================================

@st.cache_data
def prepare_data(df, apply_age_adjustment=True, cpi_ppp_file=None, apply_ppp=False):
    """
    Prepare and clean data for IOP analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data
    apply_age_adjustment : bool
        Whether to apply age adjustment
    cpi_ppp_file : UploadedFile or None
        CPI-PPP reference file for PPP adjustment
    apply_ppp : bool
        Whether to apply PPP adjustment
    """
    
    # Apply PPP adjustment FIRST (before other transformations)
    ppp_info = None
    if apply_ppp and cpi_ppp_file is not None:
        df, ppp_info = load_and_apply_ppp_adjustment(df, cpi_ppp_file)
    
    # Handle essential column naming for backward compatibility
    # Only map the absolutely essential columns
    if 'hh_income_pc' in df.columns and 'income' not in df.columns:
        df = df.rename(columns={'hh_income_pc': 'income'})
    if 'hh_weight' in df.columns and 'weights' not in df.columns:
        df = df.rename(columns={'hh_weight': 'weights'})
    
    # Add weights if missing
    if 'weights' not in df.columns:
        df['weights'] = 1.0
    
    # Add derived columns
    df['loginc'] = np.log(df['income'] + 1)
    
    # Age adjustment
    if apply_age_adjustment and 'age' in df.columns:
        df['age2'] = df['age'] ** 2
        from sklearn.linear_model import LinearRegression
        
        X = df[['age', 'age2']].values
        y = df['loginc'].values
        weights = df['weights'].values
        
        model = LinearRegression()
        model.fit(X, y, sample_weight=weights)
        
        fitted_values = model.predict(X)
        adjusted_loginc = y - fitted_values + model.intercept_
        df['income_adjusted'] = np.exp(adjusted_loginc)
        
        original_mean = weighted_mean(df['income'].values, weights)
        adjusted_mean = weighted_mean(df['income_adjusted'].values, weights)
        scale_factor = original_mean / adjusted_mean
        df['income'] = df['income_adjusted'] * scale_factor
        df['loginc'] = np.log(df['income'] + 1)
    
    # Drop missing values
    df = df.dropna()
    
    # Identify circumstance variables dynamically
    # Define columns that are NOT circumstances (core/system columns)
    core_columns = {
        'id', 'year', 'cntry', 'income', 'loginc', 'weights',
        'age', 'age2', 'income_adjusted', 'income_local', 'y_tilde',
        'hh_income_pc', 'hh_weight', 'y_tilde_rf', 'types'
    }
    
    # Convert to lowercase for case-insensitive comparison
    core_columns_lower = {col.lower() for col in core_columns}
    
    # Auto-detect all other columns as potential circumstances
    potential_circumstances = []
    for col in df.columns:
        if col.lower() not in core_columns_lower:
            potential_circumstances.append(col)
            # Convert to categorical for tree-based methods
            df[col] = pd.Categorical(df[col])
    
    # Sort circumstances alphabetically for consistent ordering
    potential_circumstances.sort()
    
    return df, potential_circumstances, ppp_info

# ============================================================================
# IOP ANALYSIS FUNCTIONS
# ============================================================================

class ConditionalTree:
    """Conditional Inference Tree for IOP analysis."""
    
    def __init__(self, min_samples_leaf=50, max_depth=None, mincriterion=0.99, random_state=42):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.mincriterion = mincriterion  # Added for R compatibility
        self.random_state = random_state
        self.tree = None
        
    def fit(self, X, y, sample_weight=None):
        self.tree = DecisionTreeRegressor(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        X_encoded = X.copy()
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'category':
                X_encoded[col] = X_encoded[col].cat.codes
        
        self.tree.fit(X_encoded.values, y, sample_weight=sample_weight)
        return self
    
    def predict_types(self, X):
        X_encoded = X.copy()
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'category':
                X_encoded[col] = X_encoded[col].cat.codes
        return self.tree.apply(X_encoded.values)
    
    def get_tree_plot(self, feature_names, max_depth_display=None):
        """Generate tree visualization with safety checks."""
        # Get tree complexity metrics
        n_nodes = self.tree.tree_.node_count
        depth = self.tree.get_depth()
        
        # Safety check: prevent visualization of extremely large trees
        MAX_NODES = 500  # Increased limit
        MAX_DEPTH = 25   # Increased limit
        
        # If tree is too large and no simplification requested, return None
        if (n_nodes > MAX_NODES or depth > MAX_DEPTH) and max_depth_display is None:
            # Return None to indicate tree is too large to visualize
            return None
        
        # If max_depth_display is set, we can try to show a simplified version
        if max_depth_display and depth > max_depth_display:
            actual_depth = max_depth_display
        else:
            actual_depth = None
        
        # Improved size calculation based on tree structure
        # Use logarithmic scaling for very large trees
        import math
        if n_nodes > 50:
            width = min(20, 8 + math.log(n_nodes) * 2)
            height = min(15, 5 + depth * 0.8)
        else:
            width = min(15, max(8, n_nodes * 0.2))
            height = min(10, max(5, depth * 1.2))
        
        dpi = 50  # Keep low DPI for safety
        
        # Additional safety check for total pixel count
        total_pixels = (width * dpi) * (height * dpi)
        if total_pixels > 3_000_000:  # Slightly increased limit
            # Scale down if still too large
            scale_factor = (3_000_000 / total_pixels) ** 0.5
            width *= scale_factor
            height *= scale_factor
        
        try:
            fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
            
            # Apply max_depth_display if specified
            if max_depth_display is not None:
                # Note: sklearn's plot_tree doesn't support max_depth directly
                # We'll use it but note this limitation
                plot_tree(self.tree, 
                         feature_names=feature_names, 
                         filled=True, 
                         rounded=True, 
                         ax=ax, 
                         fontsize=9,
                         proportion=True,
                         precision=2,
                         impurity=False,
                         max_depth=max_depth_display)  # This may not work with sklearn
            else:
                plot_tree(self.tree, 
                         feature_names=feature_names, 
                         filled=True, 
                         rounded=True, 
                         ax=ax, 
                         fontsize=9,
                         proportion=True,
                         precision=2,
                         impurity=False)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            # If any error occurs during plotting, return None
            return None
    
    def get_interactive_tree_plot(self, feature_names, data=None):
        """Generate interactive Plotly sunburst visualization."""
        from sklearn.tree import _tree
        import plotly.graph_objects as go
        
        tree = self.tree.tree_
        
        # Build hierarchical data for sunburst
        ids = []
        labels = []
        parents = []
        values = []
        colors = []
        
        def add_node(node_id, parent_id=None):
            if tree.feature[node_id] != _tree.TREE_UNDEFINED:
                # Internal node
                feature = feature_names[tree.feature[node_id]]
                threshold = tree.threshold[node_id]
                n_samples = tree.n_node_samples[node_id]
                
                # Create unique ID and label for this node
                node_str_id = f"node_{node_id}"
                label = f"{feature} ≤ {threshold:.1f}"
                
                ids.append(node_str_id)
                labels.append(label)
                parents.append(parent_id if parent_id else "")
                values.append(n_samples)
                colors.append(n_samples)
                
                # Process children
                left_child = tree.children_left[node_id]
                right_child = tree.children_right[node_id]
                
                add_node(left_child, node_str_id)
                add_node(right_child, node_str_id)
            else:
                # Leaf node
                value = tree.value[node_id][0, 0]
                n_samples = tree.n_node_samples[node_id]
                
                node_str_id = f"leaf_{node_id}"
                label = f"${value:.0f}"
                
                ids.append(node_str_id)
                labels.append(label)
                parents.append(parent_id if parent_id else "")
                values.append(n_samples)
                colors.append(value)
        
        # Build the tree data starting from root
        add_node(0)
        
        # Create sunburst chart
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colorscale='Viridis',
                colorbar=dict(title="Value"),
                line=dict(width=1)
            ),
            hovertemplate='<b>%{label}</b><br>Samples: %{value}<extra></extra>',
            textinfo="label"
        ))
        
        fig.update_layout(
            title="Interactive Decision Tree (Click to explore)",
            height=700,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        return fig
    
    
    def get_tree_rules(self, feature_names):
        """Extract tree rules as text."""
        from sklearn.tree import _tree
        
        tree = self.tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree.feature
        ]
        
        def recurse(node, depth, parent_rule="Root"):
            indent = "  " * depth
            rules = []
            
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree.threshold[node]
                n_samples = tree.n_node_samples[node]
                
                rules.append({
                    'depth': depth,
                    'rule': f"{parent_rule}",
                    'samples': n_samples,
                    'split': f"{name} <= {threshold:.2f}"
                })
                
                rules.extend(recurse(tree.children_left[node], depth + 1, 
                                   f"{parent_rule} AND {name} <= {threshold:.2f}"))
                rules.extend(recurse(tree.children_right[node], depth + 1,
                                   f"{parent_rule} AND {name} > {threshold:.2f}"))
            else:
                # Leaf node
                value = tree.value[node][0, 0]
                n_samples = tree.n_node_samples[node]
                rules.append({
                    'depth': depth,
                    'rule': f"{parent_rule}",
                    'samples': n_samples,
                    'value': value,
                    'is_leaf': True
                })
            return rules
        
        rules = recurse(0, 0)
        return rules

class ConditionalForest:
    """Conditional Random Forest for IOP analysis."""
    
    def __init__(self, n_estimators=200, max_features='sqrt', min_samples_leaf=10, random_state=42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.forest = None
        
    def fit(self, X, y, sample_weight=None):
        self.forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        X_encoded = X.copy()
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'category':
                X_encoded[col] = X_encoded[col].cat.codes
        
        self.forest.fit(X_encoded.values, y, sample_weight=sample_weight)
        return self
    
    def predict(self, X):
        X_encoded = X.copy()
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'category':
                X_encoded[col] = X_encoded[col].cat.codes
        return self.forest.predict(X_encoded.values)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance scores."""
        importance = self.forest.feature_importances_
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

def run_ctree_analysis(data, circumstances, outcome='income', weights='weights', params=None):
    """Run C-Tree analysis."""
    X = data[circumstances]
    y = data[outcome].values
    w = data[weights].values if weights in data else None
    
    # Get parameters from session state or use defaults
    if params is None:
        params = st.session_state.get('ctree_params', {})
    
    min_samples_leaf = params.get('min_samples_leaf', max(50, int(len(data) * 0.01)))
    max_depth = params.get('max_depth', None)
    mincriterion = params.get('mincriterion', 0.99)
    
    # Fit tree
    ctree = ConditionalTree(
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        mincriterion=mincriterion
    )
    ctree.fit(X, y, w)
    
    # Get types
    types = ctree.predict_types(X)
    data['types'] = types
    
    # Calculate type means
    type_means = data.groupby('types').apply(
        lambda g: weighted_mean(g[outcome].values, g[weights].values if weights in g else None)
    )
    
    # Create smoothed income
    data['y_tilde'] = data['types'].map(type_means)
    
    # Calculate inequality measures
    gini_total = weighted_gini(data[outcome].values, w)
    mld_total = weighted_mld(data[outcome].values, w)
    gini_smoothed = weighted_gini(data['y_tilde'].values, w)
    mld_smoothed = weighted_mld(data['y_tilde'].values, w)
    
    # Get tree statistics for UI display
    tree_stats = {
        'n_nodes': ctree.tree.tree_.node_count,
        'depth': ctree.tree.get_depth(),
        'n_features': len(circumstances)
    }
    
    return {
        'n_types': len(type_means),
        'gini_total': gini_total,
        'mld_total': mld_total,
        'gini_smoothed': gini_smoothed,
        'mld_smoothed': mld_smoothed,
        'iop_gini_relative': gini_smoothed / gini_total if gini_total > 0 else 0,
        'iop_mld_relative': mld_smoothed / mld_total if mld_total > 0 else 0,
        'type_means': type_means.to_dict(),
        'model': ctree,
        'data_with_types': data,
        'tree_stats': tree_stats
    }

def run_cforest_analysis(data, circumstances, outcome='income', weights='weights', params=None):
    """Run C-Forest analysis."""
    X = data[circumstances]
    y = data[outcome].values
    w = data[weights].values if weights in data else None
    
    # Get parameters from session state or use defaults
    if params is None:
        params = st.session_state.get('cforest_params', {})
    
    n_estimators = params.get('n_estimators', 100)
    max_features = params.get('max_features', 'sqrt')
    min_samples_leaf = params.get('min_samples_leaf', max(10, int(len(data) * 0.001)))
    
    # Fit forest
    cforest = ConditionalForest(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf
    )
    cforest.fit(X, y, w)
    
    # Predict smoothed income
    y_tilde_rf = cforest.predict(X)
    data['y_tilde_rf'] = y_tilde_rf
    
    # Calculate inequality measures
    gini_total = weighted_gini(data[outcome].values, w)
    mld_total = weighted_mld(data[outcome].values, w)
    gini_smoothed = weighted_gini(y_tilde_rf, w)
    mld_smoothed = weighted_mld(y_tilde_rf, w)
    
    return {
        'gini_total': gini_total,
        'mld_total': mld_total,
        'gini_smoothed': gini_smoothed,
        'mld_smoothed': mld_smoothed,
        'iop_gini_relative': gini_smoothed / gini_total if gini_total > 0 else 0,
        'iop_mld_relative': mld_smoothed / mld_total if mld_total > 0 else 0,
        'model': cforest,
        'feature_importance': cforest.get_feature_importance(circumstances)
    }

# ============================================================================
# AI INSIGHTS FUNCTIONS
# ============================================================================

def test_ai_connection(api_key, model_id="openai/gpt-oss-120b"):
    """Test AI connection with a simple query."""
    if not GROQ_AVAILABLE or not api_key:
        return False
    
    try:
        client = Groq(api_key=api_key)
        
        # Simple test query
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Connection successful' in 3 words or less."
                }
            ],
            temperature=0.5,
            max_tokens=10
        )
        
        return True
    except Exception as e:
        return False

def generate_ai_insights(results, api_key, model_id="openai/gpt-oss-120b"):
    """Generate AI-powered interpretation of IOP results."""
    if not GROQ_AVAILABLE or not api_key:
        return None
    
    try:
        # Prepare context with results
        ctree_results = results.get('ctree', {})
        cforest_results = results.get('cforest', {})
        shapley_results = results.get('shapley', {})
        
        # Format Shapley results for better readability
        shapley_text = ""
        shapley_list = []
        if shapley_results and shapley_results.get('relative'):
            sorted_shapley = sorted(shapley_results['relative'].items(), key=lambda x: x[1], reverse=True)
            for circ, val in sorted_shapley:
                shapley_text += f"- {circ}: {val:.1f}%\n"
                shapley_list.append(f"{circ} ({val:.1f}%)")
        
        context = f"""
        You are an expert economist preparing a report for National Statistical Offices, ministries, and development partners.
        
        IMPORTANT: Use markdown formatting ONLY. Never use HTML tags like <br>. For line breaks, use double newlines.
        
        TASK: Generate a "Key Findings" summary using the circumstance-effort framework.
        
        DATA FROM ANALYSIS:
        - IOP (Gini) from C-Tree: {ctree_results.get('iop_gini_relative', 0):.1%}
        - IOP (MLD) from C-Tree: {ctree_results.get('iop_mld_relative', 0):.1%}
        - Total Inequality (Gini): {ctree_results.get('gini_total', 0):.4f}
        - Population Types Identified: {ctree_results.get('n_types', 'N/A')}
        - IOP (Gini) from C-Forest: {cforest_results.get('iop_gini_relative', 0):.1%}
        - IOP (MLD) from C-Forest: {cforest_results.get('iop_mld_relative', 0):.1%}
        
        Shapley Decomposition (Circumstance Contributions):
        {shapley_text if shapley_text else 'Not performed'}
        
        INSTRUCTIONS:
        1. Use the circumstance-effort framework terminology throughout
        2. Explain that circumstances are factors beyond individual control (birthplace, parental background, gender)
        3. Explain that effort represents individual choices and actions
        4. Create a markdown table with three columns: Metric | Value | Plain-Language Explanation
        5. For each metric, provide intuitive explanations a policymaker would immediately understand
        6. After the table, provide:
           - A simple explanation: "{ctree_results.get('iop_gini_relative', 0):.0%} of inequality stems from circumstances beyond individual control, while the remaining comes from differences in effort and choices"
           - Comparative context with similar countries
           - Which circumstances are the main drivers and their practical implications
        
        FORMAT YOUR RESPONSE AS:
        
        ## Key Findings Table
        
        | Metric | Value | Plain-Language Explanation |
        |--------|-------|---------------------------|
        | [metric rows] | ... | ... |
        
        ### What This Means
        [2-3 sentence simple explanation]
        
        ### Comparative Context
        [How this compares to similar countries]
        
        ### Most Important Finding
        [Which circumstances dominate and their practical implications]
        
        Be precise with numbers but explain them in accessible terms.
        """
        
        # Use the selected model or default to GPT-OSS-120B
        if not model_id:
            model_id = "openai/gpt-oss-120b"
            
        client = Groq(api_key=api_key)
        
        completion = client.chat.completions.create(
            model=model_id,  # Use selected model
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert economist specializing in inequality and social policy. Provide clear, actionable insights."
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            temperature=0.7,
            max_tokens=2500  # Increased to ensure complete tables
        )
        
        return completion.choices[0].message.content
            
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def generate_policy_recommendations(results, api_key, model_id="openai/gpt-oss-120b"):
    """Generate specific policy recommendations based on results."""
    if not GROQ_AVAILABLE or not api_key:
        return None
    
    try:
        ctree_results = results.get('ctree', {})
        shapley_results = results.get('shapley', {})
        relative_contributions = shapley_results.get('relative', {})
        
        # Find top contributors with detailed formatting
        top_circumstances_text = ""
        if relative_contributions:
            sorted_circumstances = sorted(relative_contributions.items(), key=lambda x: x[1], reverse=True)
            for i, (circ, val) in enumerate(sorted_circumstances[:3], 1):
                top_circumstances_text += f"{i}. {circ}: {val:.1f}% of inequality\n"
        
        context = f"""
        You are a policy advisor preparing recommendations for government ministries and development partners.
        
        IMPORTANT: Use markdown formatting ONLY. Never use HTML tags. For line breaks, use double newlines.
        
        TASK: Generate actionable policy recommendations using the circumstance-effort framework.
        
        KEY FINDINGS FROM ANALYSIS:
        - Circumstance-based inequality: {ctree_results.get('iop_gini_relative', 0):.1%} of total inequality
        - Effort-based inequality: {100 - ctree_results.get('iop_gini_relative', 0):.1%} of total inequality
        - Number of population types identified: {ctree_results.get('n_types', 'N/A')}
        
        TOP CIRCUMSTANCES DRIVING INEQUALITY:
        {top_circumstances_text if top_circumstances_text else 'No Shapley decomposition available'}
        
        FRAMEWORK CONTEXT:
        Under the circumstance-effort framework, policy should focus on equalizing circumstances (factors beyond individual control) while preserving incentives for effort. The goal is equal opportunity, not equal outcomes.
        
        INSTRUCTIONS:
        Based on these circumstances, generate concrete policy recommendations that level the playing field without undermining effort-based incentives.
        
        Structure your response as follows:
        
        1. Start with a brief introductory paragraph (2-3 sentences) summarizing why these interventions are needed
        
        2. Then provide 3-4 policy recommendations in this table format:
        
        | Intervention | Why (Rationale) | How (Implementation Steps) | Expected Impact | Time Horizon | Cost Level |
        |--------------|-----------------|---------------------------|-----------------|--------------|------------|
        | [Name] | [Link to data] | [Step-by-step guide] | [Outcomes] | [Short/Medium/Long] | [Low/Medium/High] |
        
        3. End with a "Priority Sequencing" section recommending which intervention to implement first and why
        
        GUIDELINES FOR RECOMMENDATIONS:
        - Make interventions specific and actionable (not generic)
        - Link each recommendation directly to the data findings
        - Consider scalability for different resource settings
        - Include both immediate and long-term interventions
        - Focus on evidence-based approaches that have worked in similar contexts
        - For Birth Area: Consider infrastructure, connectivity, service delivery
        - For Parental Education: Consider adult education, conditional transfers, early childhood programs
        - For Gender: Consider targeted programs, legal reforms, awareness campaigns
        
        Be practical and consider implementation feasibility in developing country contexts.
        """
        
        # Use same model for consistency
        if not model_id:
            model_id = "openai/gpt-oss-120b"
            
        client = Groq(api_key=api_key)
        
        completion = client.chat.completions.create(
            model=model_id,  # Use selected model
            messages=[
                {
                    "role": "system",
                    "content": "You are a policy advisor specializing in reducing inequality. Provide specific, implementable recommendations."
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            temperature=0.6,
            max_tokens=2500  # Increased to ensure complete policy tables
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

def run_shapley_decomposition(data, circumstances, outcome='income', weights='weights', n_permutations=100):
    """Run fast Shapley decomposition."""
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def calculate_inequality(data, active_circumstances):
        if not active_circumstances:
            return 0.0
        
        X = data[list(active_circumstances)]
        y = data[outcome].values
        w = data[weights].values if weights in data else None
        
        ctree = ConditionalTree(min_samples_leaf=max(50, int(len(data) * 0.01)))
        ctree.fit(X, y, w)
        
        types = ctree.predict_types(X)
        data_temp = data.copy()
        data_temp['types'] = types
        
        type_means = data_temp.groupby('types').apply(
            lambda g: weighted_mean(g[outcome].values, g[weights].values if weights in g else None)
        )
        
        y_tilde = data_temp['types'].map(type_means).values
        return weighted_gini(y_tilde, w)
    
    # Initialize
    marginal_contributions = {circ: [] for circ in circumstances}
    
    # Sample permutations
    for perm_idx in range(n_permutations):
        progress_bar.progress((perm_idx + 1) / n_permutations)
        status_text.text(f'Computing Shapley values... {perm_idx + 1}/{n_permutations}')
        
        perm = np.random.permutation(circumstances)
        coalition = []
        prev_inequality = 0.0
        
        for circ in perm:
            coalition.append(circ)
            curr_inequality = calculate_inequality(data, coalition)
            marginal = curr_inequality - prev_inequality
            marginal_contributions[circ].append(marginal)
            prev_inequality = curr_inequality
    
    progress_bar.empty()
    status_text.empty()
    
    # Average contributions
    shapley_values = {circ: np.mean(marginal_contributions[circ]) for circ in circumstances}
    total_shapley = sum(shapley_values.values())
    
    # Relative contributions
    relative_shapley = {}
    if total_shapley > 0:
        for circ in circumstances:
            relative_shapley[circ] = (shapley_values[circ] / total_shapley) * 100
    else:
        relative_shapley = {circ: 0.0 for circ in circumstances}
    
    return shapley_values, relative_shapley

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 IOP Inequality Analysis Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ex-ante Inequality of Opportunity using C-Trees, C-Forests, and Shapley Decomposition</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain income and circumstance variables"
        )
        
        # Sample data option
        use_sample = st.checkbox("Use sample data (Nepal 2011)", value=False)
        
        st.divider()
        
        # Configuration
        st.header("⚙️ Configuration")
        
        apply_age_adjustment = st.checkbox("Apply age adjustment", value=True)
        
        st.divider()
        
        # CPI-PPP Adjustment Section
        st.header("💱 PPP Adjustment")
        
        # Option to use sample CPI-PPP file
        use_sample_ppp = st.checkbox(
            "Use sample CPI-PPP file", 
            value=False,
            help="Use the pre-loaded CPI-PPP reference file with data for all countries"
        )
        
        if use_sample_ppp:
            # Load sample CPI-PPP file from data directory
            try:
                app_dir = os.path.dirname(os.path.abspath(__file__))
                sample_ppp_path = os.path.join(app_dir, 'data', 'cpi_ppp.xlsx')
                
                # Check if file exists
                if os.path.exists(sample_ppp_path):
                    # Create a file-like object for consistency
                    with open(sample_ppp_path, 'rb') as f:
                        cpi_ppp_file = f.read()
                    st.success("✅ Sample CPI-PPP file loaded")
                    cpi_ppp_loaded = True
                else:
                    st.error("❌ Sample CPI-PPP file not found in data directory")
                    cpi_ppp_file = None
                    cpi_ppp_loaded = False
            except Exception as e:
                st.error(f"❌ Error loading sample file: {str(e)}")
                cpi_ppp_file = None
                cpi_ppp_loaded = False
        else:
            # Regular file upload
            cpi_ppp_file = st.file_uploader(
                "Upload CPI-PPP Reference File",
                type=['xlsx', 'xls'],
                key="cpi_ppp_upload",
                help="Excel file with CPI values (1960-2022, base year 2017=100) and PPP rates in column BO"
            )
            cpi_ppp_loaded = cpi_ppp_file is not None
        
        if cpi_ppp_loaded:
            if not use_sample_ppp:
                st.success("✅ CPI-PPP file loaded")
            apply_ppp_adjustment = st.checkbox(
                "Apply PPP adjustment to income", 
                value=True,
                help="Converts local currency to international dollars (PPP-adjusted, 2017 base)"
            )
        else:
            st.info("📁 Optional: Upload CPI-PPP file or use sample file for international comparisons")
            apply_ppp_adjustment = False
        
        st.divider()
        
        # Analysis options
        st.header("📈 Analysis Options")
        
        run_ctree = st.checkbox("Run C-Tree Analysis", value=True)
        run_cforest = st.checkbox("Run C-Forest Analysis", value=True)
        run_shapley = st.checkbox("Run Shapley Decomposition", value=True)
        
        if run_shapley:
            n_permutations = st.slider("Shapley permutations", 10, 500, 100)
        else:
            n_permutations = 100
        
        # Advanced Settings (expanded by default)
        with st.expander("⚙️ Advanced Settings", expanded=True):
            st.markdown("### Parameter Configuration")
            st.info("These parameters match the R implementation and allow fine-tuning of the analysis.")
            
            # Reset button
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                if 'ctree_params' in st.session_state:
                    del st.session_state['ctree_params']
                if 'cforest_params' in st.session_state:
                    del st.session_state['cforest_params']
                st.rerun()
            
            # C-Tree Parameters
            if run_ctree:
                st.subheader("🌳 C-Tree Parameters")
                
                ctree_mincriterion = st.slider(
                    "Minimum Criterion (1-α)",
                    min_value=0.90,
                    max_value=0.999,
                    value=0.99,
                    step=0.001,
                    help="Significance level for node splits. Higher values create simpler trees."
                )
                
                ctree_minbucket = st.slider(
                    "Minimum Bucket Size",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    help="Minimum number of observations in terminal nodes."
                )
                
                ctree_maxdepth = st.slider(
                    "Maximum Tree Depth",
                    min_value=2,
                    max_value=20,
                    value=10,
                    help="Maximum depth of the tree. Set to 20 for unlimited depth."
                )
                if ctree_maxdepth == 20:
                    ctree_maxdepth = None  # None means unlimited
                
                # Store C-Tree params
                st.session_state['ctree_params'] = {
                    'mincriterion': ctree_mincriterion,
                    'min_samples_leaf': ctree_minbucket,
                    'max_depth': ctree_maxdepth
                }
            
            # C-Forest Parameters
            if run_cforest:
                st.subheader("🌲 C-Forest Parameters")
                
                cforest_n_estimators = st.slider(
                    "Number of Trees",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=50,
                    help="Number of trees in the forest. More trees = better accuracy but slower."
                )
                
                cforest_max_features = st.selectbox(
                    "Max Features per Split",
                    options=["sqrt", "log2", "0.5", "0.75"],
                    index=0,
                    help="Number of features to consider at each split."
                )
                
                cforest_min_samples_leaf = st.slider(
                    "Min Samples per Leaf",
                    min_value=5,
                    max_value=100,
                    value=10,
                    step=5,
                    help="Minimum samples required in leaf nodes."
                )
                
                # Store C-Forest params
                st.session_state['cforest_params'] = {
                    'n_estimators': cforest_n_estimators,
                    'max_features': cforest_max_features if cforest_max_features in ["sqrt", "log2"] else float(cforest_max_features),
                    'min_samples_leaf': cforest_min_samples_leaf
                }
            
            # Cross-Validation Settings
            st.subheader("📊 Cross-Validation")
            
            cv_folds = st.slider(
                "Number of CV Folds",
                min_value=3,
                max_value=20,
                value=10,
                help="Number of folds for cross-validation (if applicable)."
            )
            
            # Visualization Settings
            st.subheader("🎨 Visualization")
            
            normalize_output = st.checkbox(
                "Normalize Tree Output",
                value=True,
                help="Show relative values (normalized to population mean) vs absolute values."
            )
            
            tree_font_size = st.slider(
                "Tree Font Size",
                min_value=4,
                max_value=12,
                value=10,
                help="Font size for tree visualization."
            )
            
            # Store visualization params
            st.session_state['viz_params'] = {
                'normalize': normalize_output,
                'font_size': tree_font_size
            }
        
        st.divider()
        
        # AI Insights Configuration
        with st.expander("🤖 AI Insights Configuration", expanded=True):
            st.markdown("### AI-Powered Result Interpretation")
            
            if GROQ_AVAILABLE:
                st.info("Enter your Groq API key for AI-powered insights. Get a free key at [groq.com](https://console.groq.com/keys)")
                
                api_key = st.text_input(
                    "Groq API Key", 
                    type="password",
                    help="Your API key is never stored and only used for this session.",
                    value=st.session_state.get('ai_api_key', '')
                )
                
                # Initialize connection state
                if 'ai_connected' not in st.session_state:
                    st.session_state['ai_connected'] = False
                
                # Connect to AI Model button - always visible below API key input
                if not st.session_state['ai_connected']:
                    # Show connect button (disabled if no API key)
                    if st.button("🔌 Connect to AI Model", 
                                use_container_width=True, 
                                key="connect_ai_button",
                                disabled=(not api_key),
                                help="Enter API key above to enable connection"):
                        if api_key:
                            with st.spinner("Connecting to AI..."):
                                # Save API key
                                st.session_state['ai_api_key'] = api_key
                                st.session_state['ai_provider'] = "Groq"
                                
                                # Test connection with default model
                                if test_ai_connection(api_key, "openai/gpt-oss-120b"):
                                    st.session_state['ai_connected'] = True
                                    st.rerun()
                                else:
                                    st.error("❌ Connection failed. Please check your API key.")
                    
                    if not api_key:
                        st.info("👆 Enter your API key above to connect")
                else:
                    st.success("✅ Connected to AI Model")
                    
                    # Model selection (only show after connected)
                    st.subheader("🤖 Select AI Model")
                    
                    model_options = {
                        "GPT-OSS-120B (Primary - Best)": "openai/gpt-oss-120b",
                        "GPT-OSS-20B (Secondary - Faster)": "openai/gpt-oss-20b",
                        "Llama 3.3 70B (Alternative)": "llama-3.3-70b-versatile",
                        "Mixtral 8x7B (Lightweight)": "mixtral-8x7b-32768"
                    }
                    
                    selected_model_name = st.selectbox(
                        "Choose Model",
                        options=list(model_options.keys()),
                        index=0,
                        help="GPT-OSS-120B: Primary model with best reasoning\nGPT-OSS-20B: Secondary model, faster but still excellent\nLlama 3.3: Alternative fast model\nMixtral: Lightweight and quick"
                    )
                    
                    st.session_state['ai_model'] = model_options[selected_model_name]
                    st.session_state['ai_model_name'] = selected_model_name
                    
                    # Show model info
                    if "GPT-OSS-120B" in selected_model_name:
                        st.info("🎆 Primary Model: Best for detailed policy analysis")
                    elif "GPT-OSS-20B" in selected_model_name:
                        st.info("⚡ Secondary Model: Faster with excellent insights")
                    elif "Llama" in selected_model_name:
                        st.info("🦙 Alternative: Quick balanced analysis")
                    else:
                        st.info("🚀 Lightweight: Fast basic insights")
                    
                    # Disconnect button
                    if st.button("🔌 Disconnect", use_container_width=True):
                        st.session_state['ai_connected'] = False
                        if 'ai_api_key' in st.session_state:
                            del st.session_state['ai_api_key']
                        st.rerun()
                
                st.markdown("""
                **AI Insights include:**
                - Plain language explanation of results
                - Policy implications
                - Comparison between methods
                - Key findings summary
                """)
            else:
                st.error("❌ Groq package not installed. Run: `pip install groq`")
                st.session_state['ai_api_key'] = None
                st.session_state['ai_provider'] = None
    
    # Main content
    if uploaded_file is not None or use_sample:
        # Load data
        if use_sample:
            # Load sample data with proper path handling
            try:
                # Get the directory where the app.py file is located
                app_dir = os.path.dirname(os.path.abspath(__file__))
                sample_data_path = os.path.join(app_dir, 'data', '2011_NPL.csv')
                df = pd.read_csv(sample_data_path)
                st.success("Sample data loaded successfully!")
            except Exception as e:
                st.error(f"Sample data file not found. Please upload your own data. Error: {str(e)}")
                return
        else:
            # Load uploaded file
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"File '{uploaded_file.name}' loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        
        # Prepare data
        with st.spinner("Preparing data..."):
            df_clean, potential_circumstances, ppp_info = prepare_data(
                df, 
                apply_age_adjustment,
                cpi_ppp_file=cpi_ppp_file if apply_ppp_adjustment else None,
                apply_ppp=apply_ppp_adjustment
            )
        
        # Display PPP adjustment info if applied
        if ppp_info:
            st.success(f"✅ PPP Adjustment Applied: {ppp_info['country_name']} ({ppp_info['country_code']}) - Year {ppp_info['year']}")
            
            # Show PPP adjustment details
            with st.expander("💱 PPP Adjustment Details", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Country", f"{ppp_info['country_code']}")
                with col2:
                    st.metric("Year", ppp_info['year'])
                with col3:
                    st.metric("CPI (2017=100)", f"{ppp_info['cpi']:.1f}")
                with col4:
                    st.metric("PPP Factor", f"{ppp_info['factor']:.3f}")
                
                st.info(f"💡 Income values have been converted to international dollars (PPP-adjusted, 2017 base year). "
                       f"Original local currency values were divided by {ppp_info['factor']:.3f}")
        
        # Data overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Observations", f"{len(df_clean):,}")
        with col2:
            st.metric("Variables", len(df_clean.columns))
        with col3:
            st.metric("Circumstances Available", len(potential_circumstances))
        
        # Select circumstances
        st.subheader("Select Circumstance Variables")
        selected_circumstances = st.multiselect(
            "Choose circumstances for analysis:",
            potential_circumstances,
            default=potential_circumstances,
            help="These variables determine the 'types' in the population"
        )
        
        if len(selected_circumstances) == 0:
            st.warning("Please select at least one circumstance variable.")
            return
        
        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            
            # Clear previous results from session state
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']
            
            # Results container
            results = {}
            
            # C-Tree Analysis
            if run_ctree:
                with st.spinner("Running C-Tree analysis..."):
                    ctree_params = st.session_state.get('ctree_params', None)
                    ctree_results = run_ctree_analysis(df_clean.copy(), selected_circumstances, params=ctree_params)
                    results['ctree'] = ctree_results
            
            # C-Forest Analysis
            if run_cforest:
                with st.spinner("Running C-Forest analysis..."):
                    cforest_params = st.session_state.get('cforest_params', None)
                    cforest_results = run_cforest_analysis(df_clean.copy(), selected_circumstances, params=cforest_params)
                    results['cforest'] = cforest_results
            
            # Shapley Decomposition
            if run_shapley:
                st.subheader("Shapley Decomposition")
                shapley_values, relative_shapley = run_shapley_decomposition(
                    df_clean.copy(), selected_circumstances, n_permutations=n_permutations
                )
                results['shapley'] = {
                    'values': shapley_values,
                    'relative': relative_shapley
                }
            
            # Store results in session state
            st.session_state['analysis_results'] = results
            st.session_state['selected_circumstances'] = selected_circumstances
        
        # Display results if they exist in session state
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            selected_circumstances = st.session_state.get('selected_circumstances', selected_circumstances)
            
            st.header("📊 Results")
            
            # Create tabs for different results
            tabs = []
            if 'ctree' in results:
                tabs.append("🌳 C-Tree")
            if 'cforest' in results:
                tabs.append("🌲 C-Forest")
            if 'shapley' in results:
                tabs.append("🎲 Shapley")
            tabs.append("📊 Summary")
            
            tab_objects = st.tabs(tabs)
            tab_idx = 0
            
            # C-Tree Results
            if 'ctree' in results:
                with tab_objects[tab_idx]:
                    ctree_results = results['ctree']  # Get ctree results
                    st.markdown("### 🌳 Conditional Inference Tree Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Number of Types", ctree_results['n_types'])
                    with col2:
                        iop_val = ctree_results['iop_gini_relative']
                        delta_color = "normal" if iop_val < 0.3 else "inverse"
                        st.metric("IOP (Gini)", f"{iop_val:.1%}", 
                                delta=f"{'Low' if iop_val < 0.2 else 'Med' if iop_val < 0.4 else 'High'}",
                                delta_color=delta_color)
                    with col3:
                        st.metric("Total Gini", f"{ctree_results['gini_total']:.4f}")
                    with col4:
                        st.metric("IOP Gini", f"{ctree_results['gini_smoothed']:.4f}")
                    
                    # Tree Visualization - Better Options
                    st.subheader("🌳 Decision Tree Structure")
                    
                    # Display tree statistics if available
                    if 'tree_stats' in ctree_results:
                        stats = ctree_results['tree_stats']
                        st.info(f"📊 Tree Statistics: {stats['n_nodes']} nodes, depth={stats['depth']}, features={stats['n_features']}")
                    
                    # Add visualization type selector
                    viz_type = st.radio(
                        "Select visualization type:",
                        ["Tree Diagram", "Interactive Tree", "Text Rules"],
                        horizontal=True,
                        key="viz_type_selector"  # Add key for state management
                    )
                    
                    if viz_type == "Tree Diagram":
                        # Add option to simplify tree
                        if stats['n_nodes'] > 100:
                            st.info("💡 Large tree detected. Consider using 'Interactive Tree' view for better navigation.")
                            simplify = st.checkbox("Simplify tree visualization (show first 5 levels only)", value=False)
                            max_depth_display = 5 if simplify else None
                        else:
                            max_depth_display = None
                        
                        # Try to create tree plot
                        tree_fig = ctree_results['model'].get_tree_plot(selected_circumstances, max_depth_display=max_depth_display)
                        
                        if tree_fig is None:
                            # Tree is too large or error occurred
                            st.warning("⚠️ Tree is too large to visualize as a static diagram.")
                            st.info("Please use 'Interactive Tree' or 'Text Rules' view to explore the decision paths.")
                            
                            # Automatically show text rules as fallback
                            st.write("**Showing text rules instead:**")
                            rules = ctree_results['model'].get_tree_rules(selected_circumstances)
                            leaf_rules = [r for r in rules if r.get('is_leaf', False)]
                            
                            for i, rule in enumerate(leaf_rules[:10]):
                                clean_rule = rule['rule'].replace("Root AND ", "")
                                st.markdown(f"""
                                <div class="tree-rule">
                                <strong>Type {i+1}</strong><br>
                                Rule: {clean_rule}<br>
                                Mean Income: ${rule['value']:.2f}<br>
                                Samples: {rule['samples']}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            if len(leaf_rules) > 10:
                                st.info(f"Showing first 10 types out of {len(leaf_rules)} total")
                        else:
                            # Successfully created tree plot
                            st.info("💡 Tip: The tree shows how the population is split into types based on circumstances.")
                            try:
                                st.pyplot(tree_fig, use_container_width=False)
                                plt.close(tree_fig)  # Close figure to free memory
                            except Exception as e:
                                st.error("Failed to display tree diagram. Switching to text view.")
                                # Fallback to text rules
                                rules = ctree_results['model'].get_tree_rules(selected_circumstances)
                                leaf_rules = [r for r in rules if r.get('is_leaf', False)]
                                
                                for i, rule in enumerate(leaf_rules[:10]):
                                    clean_rule = rule['rule'].replace("Root AND ", "")
                                    st.markdown(f"""
                                    <div class="tree-rule">
                                    <strong>Type {i+1}</strong><br>
                                    Rule: {clean_rule}<br>
                                    Mean Income: ${rule['value']:.2f}<br>
                                    Samples: {rule['samples']}
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                    elif viz_type == "Interactive Tree":
                        st.write("**Interactive Decision Tree Visualization**")
                        st.info("🔍 Click on any section to zoom in and explore branches. Click the center to zoom back out.")
                        
                        try:
                            # Generate interactive sunburst chart
                            interactive_fig = ctree_results['model'].get_interactive_tree_plot(selected_circumstances)
                            st.plotly_chart(interactive_fig, use_container_width=True)
                            
                            # Add explanation
                            st.markdown("""
                            **How to use the Sunburst Chart:**
                            - **Click** on any segment to zoom into that branch
                            - **Click** the center circle to zoom back out
                            - **Hover** over segments to see sample counts
                            - Segment **size** represents the number of samples
                            - **Colors** show different values/regions
                            - Inner rings are decision nodes, outer rings are predictions
                            """)
                        except Exception as e:
                            st.error(f"Failed to generate interactive tree: {str(e)}")
                            st.info("Showing text rules as fallback...")
                            # Fallback to text rules
                            rules = ctree_results['model'].get_tree_rules(selected_circumstances)
                            leaf_rules = [r for r in rules if r.get('is_leaf', False)]
                            
                            for i, rule in enumerate(leaf_rules[:10]):
                                clean_rule = rule['rule'].replace("Root AND ", "")
                                st.markdown(f"""
                                <div class="tree-rule">
                                <strong>Type {i+1}</strong><br>
                                Rule: {clean_rule}<br>
                                Mean Income: ${rule['value']:.2f}<br>
                                Samples: {rule['samples']}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    elif viz_type == "Text Rules":
                        st.write("**Decision rules for each type:**")
                        rules = ctree_results['model'].get_tree_rules(selected_circumstances)
                        
                        # Display only leaf nodes (types)
                        leaf_rules = [r for r in rules if r.get('is_leaf', False)]
                        
                        # Add download button for rules
                        rules_text = "\n\n".join([
                            f"Type {i+1}\nRule: {rule['rule'].replace('Root AND ', '')}\nMean Income: ${rule['value']:.2f}\nSamples: {rule['samples']}"
                            for i, rule in enumerate(leaf_rules)
                        ])
                        st.download_button(
                            label="📥 Download All Rules (TXT)",
                            data=rules_text,
                            file_name="tree_rules.txt",
                            mime="text/plain"
                        )
                        
                        for i, rule in enumerate(leaf_rules[:10]):  # Show max 10 types
                            clean_rule = rule['rule'].replace("Root AND ", "")
                            st.markdown(f"""
                            <div class="tree-rule">
                            <strong>Type {i+1}</strong><br>
                            Rule: {clean_rule}<br>
                            Mean Income: ${rule['value']:.2f}<br>
                            Samples: {rule['samples']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                        if len(leaf_rules) > 10:
                            st.info(f"Showing first 10 types out of {len(leaf_rules)} total. Download the file above to see all rules.")
                    
                    # Type distribution
                    if ctree_results['n_types'] <= 20:
                        st.subheader("Type Mean Incomes")
                        type_df = pd.DataFrame({
                            'Type': list(ctree_results['type_means'].keys()),
                            'Mean Income': list(ctree_results['type_means'].values())
                        })
                        fig = px.bar(type_df, x='Type', y='Mean Income', title="Mean Income by Type")
                        fig.update_layout(
                            font=dict(size=14),
                            title_font_size=18,
                            xaxis=dict(title_font_size=16, tickfont_size=14),
                            yaxis=dict(title_font_size=16, tickfont_size=14)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Income distribution by type
                        if 'data_with_types' in ctree_results:
                            st.subheader("Income Distribution by Type")
                            data_types = ctree_results['data_with_types']
                            fig_box = px.box(data_types, x='types', y='income', 
                                            title="Income Distribution Across Types",
                                            labels={'types': 'Type', 'income': 'Income'})
                            fig_box.update_layout(
                                font=dict(size=14),
                                title_font_size=18,
                                xaxis=dict(title_font_size=16, tickfont_size=14),
                                yaxis=dict(title_font_size=16, tickfont_size=14)
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
                
                tab_idx += 1
            
            # C-Forest Results
            if 'cforest' in results:
                with tab_objects[tab_idx]:
                    cforest_results = results['cforest']  # Get cforest results
                    st.markdown("### 🌲 Conditional Random Forest Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("IOP (Gini)", f"{cforest_results['iop_gini_relative']:.1%}")
                    with col2:
                        st.metric("IOP (MLD)", f"{cforest_results['iop_mld_relative']:.1%}")
                    with col3:
                        st.metric("Total Gini", f"{cforest_results['gini_total']:.4f}")
                    with col4:
                        st.metric("Total MLD", f"{cforest_results['mld_total']:.4f}")
                    
                    # Feature Importance
                    if 'feature_importance' in cforest_results:
                        st.subheader("📊 Feature Importance")
                        importance_df = cforest_results['feature_importance']
                        
                        # Bar chart
                        fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                                               orientation='h',
                                               title="Circumstance Importance in Random Forest",
                                               color='Importance',
                                               color_continuous_scale='Viridis')
                        fig_importance.update_layout(
                            height=400,
                            font=dict(size=14),
                            title_font_size=18,
                            xaxis=dict(title_font_size=16, tickfont_size=14),
                            yaxis=dict(title_font_size=16, tickfont_size=14)
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Table
                        st.dataframe(importance_df.style.format({'Importance': '{:.4f}'}), 
                                   use_container_width=True)
                
                tab_idx += 1
            
            # Shapley Results
            if 'shapley' in results:
                with tab_objects[tab_idx]:
                    st.markdown("### 🎲 Shapley Value Decomposition")
                    
                    # Get shapley values from results
                    shapley_values = results['shapley']['values']
                    relative_shapley = results['shapley']['relative']
                    
                    # Create visualizations
                    shapley_df = pd.DataFrame({
                        'Circumstance': list(relative_shapley.keys()),
                        'Contribution (%)': list(relative_shapley.values())
                    })
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart
                        fig_bar = px.bar(
                            shapley_df, 
                            x='Circumstance', 
                            y='Contribution (%)',
                            title="Shapley Value Contributions",
                            color='Contribution (%)',
                            color_continuous_scale='Blues'
                        )
                        fig_bar.update_layout(
                            font=dict(size=14),
                            title_font_size=18,
                            xaxis=dict(title_font_size=16, tickfont_size=14),
                            yaxis=dict(title_font_size=16, tickfont_size=14)
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with col2:
                        # Pie chart
                        fig_pie = px.pie(
                            shapley_df,
                            values='Contribution (%)',
                            names='Circumstance',
                            title="Relative Contributions"
                        )
                        fig_pie.update_layout(
                            font=dict(size=14),
                            title_font_size=18
                        )
                        fig_pie.update_traces(textfont_size=14)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Table
                    st.subheader("Detailed Shapley Values")
                    shapley_table = pd.DataFrame({
                        'Circumstance': list(shapley_values.keys()),
                        'Shapley Value': [f"{v:.6f}" for v in shapley_values.values()],
                        'Relative Contribution': [f"{v:.2f}%" for v in relative_shapley.values()]
                    })
                    st.dataframe(shapley_table, use_container_width=True)
                
                tab_idx += 1
            
            # Summary Tab
            with tab_objects[tab_idx]:
                st.markdown("### 📊 Summary of All Results")
                
                summary_data = []
                
                if 'ctree' in results:
                    ctree_results = results['ctree']
                    summary_data.append({
                        'Method': 'C-Tree',
                        'IOP (Gini)': f"{ctree_results['iop_gini_relative']:.1%}",
                        'IOP (MLD)': f"{ctree_results['iop_mld_relative']:.1%}",
                        'Types': ctree_results['n_types']
                    })
                
                if 'cforest' in results:
                    cforest_results = results['cforest']
                    summary_data.append({
                        'Method': 'C-Forest',
                        'IOP (Gini)': f"{cforest_results['iop_gini_relative']:.1%}",
                        'IOP (MLD)': f"{cforest_results['iop_mld_relative']:.1%}",
                        'Types': 'N/A'
                    })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                
                # Shapley Decomposition in Summary
                if 'shapley' in results:
                    st.divider()
                    st.markdown("### 🎲 Circumstance Contributions (Shapley Decomposition)")
                    st.markdown("*Shows how much each circumstance contributes to total inequality*")
                    
                    shapley_values = results['shapley']['values']
                    relative_shapley = results['shapley']['relative']
                    
                    # Sort by contribution (highest first)
                    sorted_shapley = dict(sorted(relative_shapley.items(), key=lambda x: x[1], reverse=True))
                    
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        # Table with sorted values
                        shapley_summary_df = pd.DataFrame({
                            'Circumstance': list(sorted_shapley.keys()),
                            'Contribution (%)': [f"{v:.1f}%" for v in sorted_shapley.values()]
                        })
                        st.dataframe(shapley_summary_df, use_container_width=True, hide_index=True)
                        
                        # Highlight the most important circumstance
                        top_circumstance = list(sorted_shapley.keys())[0]
                        top_value = list(sorted_shapley.values())[0]
                        st.info(f"📌 **{top_circumstance}** is the most important circumstance, contributing **{top_value:.1f}%** to inequality")
                    
                    with col2:
                        # Horizontal bar chart for better readability
                        import plotly.graph_objects as go
                        
                        fig_shapley = go.Figure(go.Bar(
                            x=list(sorted_shapley.values()),
                            y=list(sorted_shapley.keys()),
                            orientation='h',
                            marker_color='steelblue',
                            text=[f"{v:.1f}%" for v in sorted_shapley.values()],
                            textposition='auto',
                        ))
                        
                        fig_shapley.update_layout(
                            title="Contribution to Inequality (%)",
                            xaxis_title="Percentage Contribution",
                            yaxis_title="",
                            height=300,
                            margin=dict(l=0, r=0, t=40, b=40),
                            title_font_size=18,
                            xaxis=dict(title_font_size=14, tickfont_size=12),
                            yaxis=dict(tickfont_size=14)
                        )
                        
                        st.plotly_chart(fig_shapley, use_container_width=True)
                
                # AI Insights Section (Automatic if connected)
                st.divider()
                st.subheader("🤖 AI-Powered Insights")
                
                api_key = st.session_state.get('ai_api_key')
                ai_connected = st.session_state.get('ai_connected', False)
                model_id = st.session_state.get('ai_model', 'openai/gpt-oss-120b')
                model_name = st.session_state.get('ai_model_name', 'GPT-OSS-120B')
                
                if ai_connected and api_key:
                    # Check if we need to generate insights
                    results_hash = str(hash(str(results)))
                    
                    # Auto-generate insights if not already generated for these results
                    if 'last_results_hash' not in st.session_state or st.session_state.get('last_results_hash') != results_hash:
                        st.session_state['last_results_hash'] = results_hash
                        
                        with st.spinner(f"Analyzing results with {model_name}..."):
                            insights = generate_ai_insights(results, api_key, model_id)
                            
                            if insights and not insights.startswith("Error"):
                                st.session_state['ai_insights'] = insights
                                
                                # Generate policy recommendations
                                recommendations = generate_policy_recommendations(results, api_key, model_id)
                                if recommendations and not recommendations.startswith("Error"):
                                    st.session_state['ai_recommendations'] = recommendations
                    
                    # Display saved insights
                    if 'ai_insights' in st.session_state:
                        st.success(f"✅ Analysis Complete - Using {model_name}")
                        
                        # Main insights
                        with st.container():
                            st.markdown("#### 📊 Key Findings")
                            st.markdown(st.session_state['ai_insights'])
                        
                        # Policy recommendations
                        if 'ai_recommendations' in st.session_state:
                            with st.expander("🎯 Policy Recommendations", expanded=True):
                                st.markdown(st.session_state['ai_recommendations'])
                        
                        # Regenerate button
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            if st.button("🔄 Regenerate", use_container_width=True, key="regenerate_insights"):
                                if 'last_results_hash' in st.session_state:
                                    del st.session_state['last_results_hash']
                                if 'ai_insights' in st.session_state:
                                    del st.session_state['ai_insights']
                                if 'ai_recommendations' in st.session_state:
                                    del st.session_state['ai_recommendations']
                                # Preserve tab selection
                                st.session_state['selected_tab'] = len(tabs) - 1  # Summary is last tab
                                st.rerun()
                    else:
                        st.info("Generating analysis... Please wait.")
                
                else:
                    # Not connected to AI
                    st.warning("⚠️ AI not connected. Please connect to AI Model in the sidebar to enable insights.")
                    st.info("Steps: 1) Enter Groq API key in sidebar 2) Click 'Connect to AI Model' 3) Re-run analysis")
                    if not GROQ_AVAILABLE:
                        st.error("❌ Groq package not installed. Run: `pip install groq`")
                    with st.expander("Why use AI Insights?"):
                        st.markdown("""
                        AI Insights help you:
                        - Understand what the IOP percentages mean in plain language
                        - Identify which circumstances matter most for policy
                        - Get specific, actionable policy recommendations
                        - Communicate results to non-technical stakeholders
                        """)
                
                # Download results
                st.subheader("Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # JSON download
                    results_json = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=results_json,
                        file_name="iop_results.json",
                        mime="application/json"
                    )
                
                with col2:
                    # CSV download
                    if summary_data:
                        csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Summary (CSV)",
                            data=csv,
                            file_name="iop_summary.csv",
                            mime="text/csv"
                        )
    
    else:
        # Welcome message
        st.info("""
        👋 **Welcome to the IOP Analysis Tool!**
        
        To get started:
        1. Upload your data file using the sidebar
        2. Select circumstance variables
        3. Choose analysis methods
        4. Click 'Run Analysis' to see results
        
        **Data Requirements:**
        - CSV or Excel format
        - Must contain an 'income' column
        - Circumstance variables (e.g., Sex, Father_Edu, Mother_Edu, Birth_Area, Religion)
        - Optional: 'weights' column for weighted analysis
        """)
        
        # About section
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            This tool implements ex-ante Inequality of Opportunity (IOP) analysis using:
            
            - **C-Tree (Conditional Inference Trees)**: Partitions population into types based on circumstances
            - **C-Forest (Conditional Random Forest)**: Ensemble method for robust IOP estimation  
            - **Shapley Decomposition**: Decomposes inequality contributions by each circumstance
            
            Based on methodology from the ADB Workshop on Inequality of Opportunity.
            """)

if __name__ == "__main__":
    main()