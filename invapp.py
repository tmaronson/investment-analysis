# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:38:36 2026

@author: tmaro
"""
import yfinance as yf 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st

@st.cache_data
def data_download():
    # Compute annualized parameters (252 trading days) 
    # Download 20 years of historical daily adjusted prices
    tickers = ["SPY", "BND"] 
    raw_data = yf.download(tickers, start="2004-01-01", auto_adjust=True)["Close"]
    return raw_data
    
def execute_pipeline():
    raw_data_df = data_download()
    daily_returns = raw_data_df.pct_change().dropna()
    
    # Interactive Sliders 
    equity_pct = st.sidebar.slider("Equity Allocation (%)", min_value=10, max_value=100, value=73, step=1) 
    fee_bps = st.sidebar.slider("Management Fee (Basis Points)", min_value=0, max_value=200, value=60, step=5) 
    n_sims = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=25000, value=10000, step=1000) 
    
    # 2. Compute the blended benchmark daily return series
    frac_equity = equity_pct/100
    daily_returns[f"Blended_{frac_equity}_{1 - frac_equity}"] = (frac_equity * daily_returns["SPY"]) + ((1.0 - frac_equity) * daily_returns["BND"]) 
    print(daily_returns.tail())
    st.write(daily_returns.tail())
    ann_returns = daily_returns.mean() * 252 
    ann_volatility = daily_returns.std() * np.sqrt(252) 
    correlation = daily_returns["SPY"].corr(daily_returns["BND"]) 
    print("Annualized Returns:\n", ann_returns) 
    print("\nAnnualized Volatility:\n", ann_volatility) 
    print(f"\nStock/Bond Correlation: {correlation:.3f}")
    st.write("Annualized Returns:\n", ann_returns) 
    st.write("\nAnnualized Volatility:\n", ann_volatility) 
    st.write(f"\nStock/Bond Correlation: {correlation:.3f}")
       
    
    # Render Charts st.pyplot(plot_fan_chart(years_arr, gross_paths, net_paths, fee_bps)) st.pyplot(plot_histogram(gross_paths[-1], net_paths[-1], fee_bps)) 
    
    # Monte Carlo simulation
    years, gross_paths, net_paths = monte_carlo_sim(ann_returns, ann_volatility,fee_bps, n_sims)
    # Plot fan chart in Streamlit.
    fig = plot_fandown_chart(years, gross_paths, net_paths, fee_bps)
    st.pyplot(fig)
    # Plot histogram in Streamlit.
    fig = plot_histogram(gross_paths[-1], net_paths[-1], fee_bps)
    st.pyplot(fig)

def monte_carlo_sim(ann_returns, ann_volatility, fee_bps, n_sims):
    # Monte Carlo simulation parameters 
     
    years = 20 
    init_val = 1000000 
    np.random.seed(42) 
    # Simulate annual returns across 10,000 paths 
    gross_returns = np.random.normal(ann_returns, ann_volatility, (years, n_sims)) 
    net_returns = np.random.normal(ann_returns - (fee_bps/10000) , ann_volatility, (years, n_sims)) 
    # Calculate cumulative compounding paths 
    gross_paths = init_val * np.vstack([np.ones(n_sims), np.cumprod(1 + gross_returns, axis=0)]) 
    net_paths = init_val * np.vstack([np.ones(n_sims), np.cumprod(1 + net_returns, axis=0)])
    print(f"Median Gross Terminal Wealth (20 yrs): ${np.median(gross_paths[-1]):,.0f}") 
    print(f"Median Net Terminal Wealth (20 yrs): ${np.median(net_paths[-1]):,.0f}") 
    print(f"Median Fee Cost Friction: ${np.median(gross_paths[-1]) - np.median(net_paths[-1]):,.0f}")
    st.write(f"Median Gross Terminal Wealth (20 yrs): ${np.median(gross_paths[-1]):,.0f}") 
    st.write(f"Median Net Terminal Wealth (20 yrs): ${np.median(net_paths[-1]):,.0f}") 
    st.write(f"Median Fee Cost Friction: ${np.median(gross_paths[-1]) - np.median(net_paths[-1]):,.0f}")
    return years, gross_paths, net_paths
    
def plot_fandown_chart(years_arr, gross_paths, net_paths, fee_bps):
    fig, ax = plt.subplots(figsize=(11, 5.5)) 
    gross_p5, gross_p50, gross_p95 = np.percentile(gross_paths, [5, 50, 95], axis=1)
    net_p5, net_p50, net_p95 = np.percentile(net_paths, [5, 50, 95], axis=1)
    ax.plot(years_arr, gross_p50, color="#1f77b4", linewidth=2.5, label="Gross Benchmark Median (0.0% Fee)") 
    ax.fill_between(years_arr, gross_p5, gross_p95, color="#1f77b4", alpha=0.15, label="Gross 90% CI (5th-95th)") 
    ax.plot(years_arr, net_p50, color="#d62728", linewidth=2.5, linestyle="--", label=f"Net Portfolio Median ({fee_bps/100:.2f}% Annual Fee)") 
    ax.fill_between(years_arr, net_p5, net_p95, color="#d62728", alpha=0.15, label="Net 90% CI (5th-95th)") 
    ax.set_title("20-Year Wealth Trajectories (Compounding Spread)", fontsize=12, fontweight="bold") 
    ax.set_xlabel("Years Elapsed") 
    ax.set_ylabel("Portfolio Value ($)") 
    ax.yaxis.set_major_formatter('${x:,.0f}') 
    ax.legend(loc="upper left") 
    plt.tight_layout() 
    return fig 
    
def plot_histogram(gross_terminal, net_terminal, fee_bps): 
    fig, ax = plt.subplots(figsize=(11, 5.5)) 
    sns.histplot(gross_terminal, bins=50, color="#1f77b4", alpha=0.35, kde=True, label="Gross Benchmark", ax=ax) 
    sns.histplot(net_terminal, bins=50, color="#d62728", alpha=0.35, kde=True, label=f"Net Portfolio ({fee_bps/100:.2f}% Fee)", ax=ax)
    ax.axvline(np.median(gross_terminal), color="#1f77b4", linestyle="--", linewidth=2, label=f"Gross Median: ${np.median(gross_terminal):,.0f}") 
    ax.axvline(np.median(net_terminal), color="#d62728", linestyle="--", linewidth=2, label=f"Net Median: ${np.median(net_terminal):,.0f}") 
    ax.set_title("Terminal Wealth Distribution at Year 20", fontsize=12, fontweight="bold") 
    ax.set_xlabel("Terminal Portfolio Value ($)") 
    ax.set_ylabel("Simulation Frequency") 
    ax.xaxis.set_major_formatter('${x:,.0f}') 
    ax.legend(loc="upper right") 
    plt.tight_layout() 
    return fig
    
    
    # 3. Calculate probability of shortfall
    shortfall_pct = np.mean(net_terminal < np.median(gross_terminal)) * 100 
    print(f"Percentage of net outcomes falling below gross benchmark median: {shortfall_pct:.1f}%")
    st.write(f"Percentage of net outcomes falling below gross benchmark median: {shortfall_pct:.1f}%")
      
    



if __name__ == '__main__':
    execute_pipeline()