# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:38:36 2026

@author: tmaro
"""

from collections import namedtuple
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from scipy import stats 

import yfinance as yf 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st
import pandas as pd

PROJECT_DIR = Path(__file__).parent.resolve()
STYLE_FILE = PROJECT_DIR/"style.css"

@st.cache_data
def data_download():
    # Compute annualized parameters (252 trading days) 
    # Download 20 years of historical daily adjusted prices
    """
   Download data.

   Args:
       None

   Returns:
       Dataframe
   """
    tickers = ["SPY", "BND"] 
    raw_data = yf.download(tickers, start="2004-01-01", auto_adjust=True)["Close"]
    return raw_data

# Get stylesheet.    
def local_css(file_name):
    """
       Load CSS stylesheet.

       Parameters:
           file_name (string): Name of file

       Returns:
           Nothing
       """
    with open(PROJECT_DIR / file_name, "r", encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        st.sidebar.title("Investment Analysis Legend")
        
# Hypothesis testing to test null hypothesis and get p-value
def compute_hypothesis_test(daily_returns, slider_vals):
    """
       Compute values for hypothesis testing of historic returns and actual portfolio.

       Parameters:
           daily_returns (DataFrame): Daily returns as floats.

       Returns:
           str: Tuple with t stat, p-value, net Series mean and net Series std. dev.
       """
    blended = (slider_vals.frac_equity * daily_returns["SPY"]) + ((1.0 - slider_vals.frac_equity) * daily_returns["BND"])
    fee_daily = (slider_vals.fee_bps / 10000) / 252 
    net_series = blended - fee_daily 
    # Get t-statistic and p-value now.
    # Null hypothesis states that the true mean excess return of the Monte Carlo simulated portfolio 
    # compared to the passive 73/27 index is less than or equal to zero. If p < 0.01 at the 99% CI
    # then we can reject the null hypothesis. This means that the market return alpha is greater than zero.
    t_stat, p_val = stats.ttest_1samp(net_series, 0.0) 
    return t_stat, p_val, float(net_series.mean() * 252), float(net_series.std() * np.sqrt(252))
       
def execute_pipeline():
    """
       Start entire application as entry point to create namedtuples and call functions.

       Parameters:
           None

       Returns:
           None
       """
    raw_data_df = data_download()
    daily_returns = raw_data_df.pct_change().dropna()
    local_css(STYLE_FILE)
    
    # Interactive Sliders 
    equity_pct = st.sidebar.slider("Equity Allocation (%)", min_value=10, max_value=100, value=73, step=1) 
    fee_bps = st.sidebar.slider("Management Fee (Basis Points)", min_value=0, max_value=200, value=60, step=5) 
    n_sims = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=25000, value=10000, step=1000) 
    
    # Calculate and return data
    # Create namedtuples to organize data, reduce code smells, maintain safety, avoid bugs.
    frac_equity = equity_pct/100
    SliderValues = namedtuple("SliderValues", ['frac_equity', 'fee_bps', 'n_sims'])
    slider_vals = SliderValues(frac_equity, fee_bps, n_sims)
    print(slider_vals.fee_bps)
    
    StatValuesBasic = namedtuple('StatValues', ['mu_gross','sigma', 'correlation'])
    StatValuesAnnual = namedtuple('StatValuesAnnual', ['ann_returns', 'ann_volatility'])
    MCPaths = namedtuple('MCPaths', ['years_arr', 'gross_paths', 'net_paths'])
        
    mu_gross, sigma, ann_returns, ann_volatility, correlation = calculate_data(daily_returns, slider_vals)
    stat_vals_basic = StatValuesBasic(mu_gross, sigma, correlation)
    stat_vals_annual = StatValuesAnnual(ann_returns, ann_volatility)
    # Perform a Monte Carlo simulation.
    years_arr, gross_paths, net_paths = monte_carlo_sim(stat_vals_basic, slider_vals)
    mc_paths = MCPaths(years_arr, gross_paths, net_paths)
    
    
    # Display data in tabs
    display_data(stat_vals_annual, mc_paths, correlation, daily_returns, slider_vals)
    
    # Analyze real portfolio data from cumulative returns.
    analyze_real_portfolio(daily_returns, frac_equity)

# Perform relevant calculations and display data in tabs.
def calculate_data(daily_returns, slider_vals):
    """
       Calculate relevant statistical data.

       Parameters:
           daily_returns (DataFrame): Daily returns as floats.
           slider_vals (tuple): Slider values of fraction portfolio equity percent, fee in basis points, number of Monte Carlo simulations.
       Returns:
           tuple: Gross mean return, standard deviation, annual returns, annual volatility, stock/bond correlation
       """
    # Compute the blended benchmark daily return series
    blended_series = (slider_vals.frac_equity * daily_returns["SPY"]) + ((1.0 - slider_vals.frac_equity) * daily_returns["BND"]) 
    mu_gross = float(blended_series.mean() * 252) 
    sigma = float(blended_series.std() * np.sqrt(252))
    daily_returns[f"Blended_{slider_vals.frac_equity}_{1 - slider_vals.frac_equity:.2f}"] = (slider_vals.frac_equity * daily_returns["SPY"]) \
        + ((1.0 - slider_vals.frac_equity) * daily_returns["BND"]) 
    ann_returns = daily_returns.mean() * 252 
    ann_volatility = daily_returns.std() * np.sqrt(252)
    correlation = daily_returns["SPY"].corr(daily_returns["BND"]) 
    print("\nAnnualized Volatility:\n", ann_volatility) 
    print(f"\nStock/Bond Correlation: {correlation:.3f}")
    return mu_gross, sigma, ann_returns, ann_volatility, correlation
    
def display_data(stat_values_annual, mc_paths, correlation, daily_returns, slider_vals):
    """
       Display data for relevant statistical performance values of annual returns/volatility and other analysis values.

       Parameters:
           stat_values_annual (tuple): Statistical values.
           mc_paths (tuple): Monte Carlo paths for gross and net paths.
           correlation (float): Stock/bond correlation.
           daily_returns (DataFrame): Daily returns as floats.
           slider_vals (tuple): Slider values of fraction portfolio equity percent, fee in basis points, number of Monte Carlo simulations.

       Returns:
           Nothing
       """
    tab1, tab2, tab3 = st.tabs(["Investment Tables", "Investment Graphs", "Application Notes"])
    
    with tab1:
        # Convert Ticker types which are like Dataframes to show as tables.
        # Convert each separate series object into its own DataFrame
        df_returns = stat_values_annual.ann_returns.to_frame(name="Annualized Returns")
        df_volatility = stat_values_annual.ann_volatility.to_frame(name="Annualized Volatility")
        
       # Display them as two completely separate HTML tables
        st.subheader("Annualized Returns")
        styled_html = (df_returns.style
                    .set_table_attributes('class="custom-table"') # Assigns base table class
                    .format("{:.2%}")
                    .to_html())
        st.markdown(styled_html, unsafe_allow_html=True)
        
        st.subheader("Annualized Volatility")
        styled_html = (df_volatility.style
                    .set_table_attributes('class="custom-table"') # Assigns base table class
                    .format("{:.2%}")
                    .to_html())
        st.markdown(styled_html, unsafe_allow_html=True)
        
        st.markdown(
                    f'<div class="calculation-highlight">MONTE CARLO SIMULATION DATA</div>'
                    f'<div class="calculation-highlight">Stock/Bond Correlation: {correlation:,.3f}</div>'
                    f'<div class="calculation-highlight">Median Gross Terminal Wealth (20 yrs): ${np.median(mc_paths.gross_paths[-1]):,.0f}</div>' 
                    f'<div class="calculation-highlight">Median Net Terminal Wealth (20 yrs): ${np.median(mc_paths.net_paths[-1]):,.0f}</div>'
                    f'''<div class="calculation-highlight">Median Fee Cost Friction: ${np.median(
                        mc_paths.gross_paths[-1]) - np.median(mc_paths.net_paths[-1]):,.0f}</div>''',
                    unsafe_allow_html=True
                   )
        # Get hypothesis testing information
        
        t_stat, p_val, mean_diff, std_diff = compute_hypothesis_test(daily_returns, slider_vals)
        st.markdown(
                    f'<div class="calculation-highlight">t-statistic: {t_stat:,.2f}</div>'
                    f'<div class="calculation-highlight">p-value: {p_val:.4f}</div>' 
                    f'<div class="calculation-highlight">mean difference: {mean_diff:.4f}</div>'
                    f'<div class="calculation-highlight">std. dev. difference: {std_diff:.3f}</div>',
                    unsafe_allow_html=True
                   )
        
    with tab2:
        # Plot fan chart in Streamlit.
        fig = plot_fandown_chart(mc_paths, slider_vals.fee_bps)
        st.pyplot(fig, width=1600)
        # Plot histogram in Streamlit.
        #fig = plot_histogram(mc_paths.gross_paths[-1], mc_paths.net_paths[-1], slider_vals.fee_bps)
        fig = plot_histogram(mc_paths, slider_vals.fee_bps)
        st.pyplot(fig)
    
    with tab3:
        try:
            with open(PROJECT_DIR / "README.md", "r", encoding="utf-8") as f: 
                readme_content = f.read() 
                st.markdown(readme_content, unsafe_allow_html=True) 
        except FileNotFoundError:
            st.error("README..md file not found.")


def monte_carlo_sim(stat_vals_basic, slider_vals):
    # Monte Carlo simulation parameters 
    """
       Simulate stock and bond returns for Monte Carlo simulation paths.

       Parameters:
           stat_val_basic (tuple): Statistical values for gross mean return, volatility/std. dev., correlation stocks/bonds.
           slider_vals (tuple): Slider values of fraction portfolio equity, fee in basis points, number of Monte Carlo simulations

       Returns:
           tuple: Array of years for simulations, gross paths results before fees, net paths results after fees
       """
    years = 20 
    init_val = 1000000 
    np.random.seed(42) 
    # Simulate annual returns across 10,000 paths 
    gross_returns = np.random.normal(stat_vals_basic.mu_gross, stat_vals_basic.sigma, (years, slider_vals.n_sims)) 
    net_returns = np.random.normal(stat_vals_basic.mu_gross - (slider_vals.fee_bps/10000) , 
                                   stat_vals_basic.sigma, (years, slider_vals.n_sims)) 
    # Calculate cumulative compounding paths 
    gross_paths = init_val * np.vstack([np.ones(slider_vals.n_sims), np.cumprod(1 + gross_returns, axis=0)]) 
    net_paths = init_val * np.vstack([np.ones(slider_vals.n_sims), np.cumprod(1 + net_returns, axis=0)])
    print(f"Median Gross Terminal Wealth (20 yrs): ${np.median(gross_paths[-1]):,.0f}") 
    print(f"Median Net Terminal Wealth (20 yrs): ${np.median(net_paths[-1]):,.0f}") 
    print(f"Median Fee Cost Friction: ${np.median(gross_paths[-1]) - np.median(net_paths[-1]):,.0f}")
    years_arr = np.arange(years + 1)
    return years_arr, gross_paths, net_paths
    
def plot_fandown_chart(mc_paths, fee_bps):
    """
       Plot fandown chart for data.
       Parameters:
           mc_paths (tuple): Monte Carlo paths for gross and net paths.
           fee_bps (float): Management fee basis points.

       Returns:
           fig: Figure object.
       """
    fig, ax = plt.subplots() 
    gross_p5, gross_p50, gross_p95 = np.percentile(mc_paths.gross_paths, [5, 50, 95], axis=1)
    net_p5, net_p50, net_p95 = np.percentile(mc_paths.net_paths, [5, 50, 95], axis=1)
    ax.plot(mc_paths.years_arr, gross_p50, color="#1f77b4", linewidth=2.5, label="Gross Benchmark Median (0.0% Fee)") 
    ax.fill_between(mc_paths.years_arr, gross_p5, gross_p95, color="#1f77b4", alpha=0.15, label="Gross 90% CI (5th-95th)") 
    ax.plot(mc_paths.years_arr, net_p50, color="#d62728", linewidth=2.5, linestyle="--", label=f"Net Portfolio Median ({fee_bps/100:.2f}% Annual Fee)") 
    ax.fill_between(mc_paths.years_arr, net_p5, net_p95, color="#d62728", alpha=0.15, label="Net 90% CI (5th-95th)") 
    ax.set_title("20-Year Wealth Trajectories (Compounding Spread)", fontsize=12, fontweight="bold") 
    ax.set_xlabel("Years Elapsed") 
    ax.set_ylabel("Portfolio Value In Millions ($)") 
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*1e-6:,.0f}"))
    ax.legend(loc="upper left") 
    plt.tight_layout() 
    return fig 
    
def plot_histogram(mc_paths, fee_bps): 
    # Calculate probability of shortfall
    """
       Plot histogram chart for data.
       Parameters:
           mc_paths (tuple): Monte Carlo paths for gross and net paths.
           fee_bps (float): Management fee.

       Returns:
           fig: Figure object.
       """
    shortfall_pct = np.mean(mc_paths.net_paths[-1] < np.median(mc_paths.gross_paths[-1])) * 100 
    print(f"Percentage of net outcomes falling below gross benchmark median: {shortfall_pct:.1f}%")
    st.markdown(
                f'<div class="calculation-highlight"> \
                Percentage of net outcomes falling below gross benchmark median: {shortfall_pct:.1f}%</div>'
                ,
                unsafe_allow_html=True
               )
    fig, ax = plt.subplots() 
    sns.histplot(mc_paths.gross_paths[-1], bins=50, color="#1f77b4", alpha=0.35, kde=True, label="Gross Benchmark", ax=ax) 
    sns.histplot(mc_paths.net_paths[-1], bins=50, color="#d62728", alpha=0.35, kde=True, label=f"Net Portfolio ({fee_bps/100:.2f}% Fee)", ax=ax)
    ax.axvline(np.median(mc_paths.gross_paths[-1]), color="#1f77b4", linestyle="--", linewidth=2, label=f"Gross Median: ${np.median(mc_paths.gross_paths[-1]):,.0f}") 
    ax.axvline(np.median(mc_paths.net_paths[-1]), color="#d62728", linestyle="--", linewidth=2, label=f"Net Median: ${np.median(mc_paths.net_paths[-1]):,.0f}") 
    ax.set_title("Terminal Wealth Distribution at Year 20", fontsize=12, fontweight="bold") 
    ax.set_xlabel("Terminal Portfolio Value In Millions ($)") 
    ax.set_ylabel("Simulation Frequency") 
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*1e-6:,.0f}"))
    ax.legend(loc="upper right")
    ax.set_xlim(0, np.percentile(mc_paths.gross_paths[-1], 98))
    plt.tight_layout() 
    return fig
    
def analyze_real_portfolio(daily_returns, frac_equity):
    # 1. Load and clean your portfolio YTD CSV 
    """
       Analyze an actual portfolio of stocks and bonds.
       Parameters:
           daily_returns (DataFrame): Daily returns as floats.
           frac_equity (float): Equity fraction of portfolio.

       Returns:
           Nothing.
       """
    port_df = pd.read_csv(PROJECT_DIR/"Performance_export.csv") 
    port_df.columns = ["Date", "Cum_Return"] 
    # 2. Parse dates and clean percentage values if stored as strings 
    
    port_df["Date"] = pd.to_datetime(port_df["Date"], format="%m-%d-%Y") 
    port_df.set_index("Date", inplace=True) 
    port_df.sort_index(ascending=True, inplace=True) 
    port_df.index = port_df.index.tz_localize(None).normalize() 
    daily_returns.index = daily_returns.index.tz_localize(None).normalize() 
    port_df["Cum_Return"] = port_df["Cum_Return"] / 100.0 
    #port_df["Port_Daily"] = (1.0 + port_df["Cum_Return"]) / (1.0 + port_df["Cum_Return"].shift(1)) - 1.0 
    port_df.dropna(inplace=True)
    
    if port_df["Cum_Return"].dtype == object: 
        port_df["Cum_Return"] = port_df["Cum_Return"].str.rstrip("%").astype(float) / 100.0 
    # 3. Derive daily returns from the cumulative series 
    port_df["Port_Daily"] = (1.0 + port_df["Cum_Return"]) / (1.0 + port_df["Cum_Return"].shift(1)) - 1.0 
    #port_df["Port_Daily"] = (port_df["Cum_Return"] - port_df["Cum_Return"].shift(1)) / (1.0 + port_df["Cum_Return"].shift(1))
    port_df.dropna(inplace=True) 
    # 4. Align with your 73/27 benchmark on the exact same dates 
    aligned_df = port_df.join(daily_returns[f"Blended_{frac_equity}_{1 - frac_equity:.2f}"], how="inner").dropna() 
    
    aligned_df["Active_Diff"] = aligned_df["Port_Daily"] - aligned_df[f"Blended_{frac_equity}_{1 - frac_equity:.2f}"] 
    
    print(f"Matched rows count: {len(aligned_df)}") 
    print(aligned_df[["Port_Daily", f"Blended_{frac_equity}_{1 - frac_equity:.2f}", "Active_Diff"]].head())
    
    # 5. Run the paired difference t-test from scipy
    t_stat, p_val = stats.ttest_1samp(aligned_df["Active_Diff"], 0.0) 
    ann_alpha = aligned_df["Active_Diff"].mean()*252 
    tracking_err = aligned_df["Active_Diff"].std() * np.sqrt(252)
    print(f"Annualized Active Alpha: {ann_alpha:.2f}%")
    print(f"Annualized Tracking Error: {tracking_err:.2f}%") 
    print(f"t-statistic: {t_stat:.2f}") 
    print(f"p-value: {p_val:.4f}")
    st.markdown(
                f'<div class="calculation-highlight">REAL PORTFOLIO ANALYSIS DATA</div>'
                f'<div class="calculation-highlight">Annualized Active Alpha: {ann_alpha:.2f}%</div>'
                f'<div class="calculation-highlight">Annualized Tracking Error: {tracking_err:.2f}%</div>' 
                f'<div class="calculation-highlight">t-statistic: {t_stat:.2f}</div>'
                f'<div class="calculation-highlight">p-value: {p_val:.4f}</div>',
                unsafe_allow_html=True
               )
    
if __name__ == '__main__':
    execute_pipeline()
    
    