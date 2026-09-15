# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:38:36 2026

@author: tmaro
"""
import yfinance as yf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st

def execute_pipeline():
    # Compute annualized parameters (252 trading days) 
    # Download 20 years of historical daily adjusted prices
    tickers = ["SPY", "BND"] 
    raw_data = yf.download(tickers, start="2004-01-01", auto_adjust=True)["Close"] 
    daily_returns = raw_data.pct_change().dropna()
    
    # 2. Compute the blended 73/27 benchmark daily return series
    daily_returns["Blended_73_27"] = (0.73 * daily_returns["SPY"]) + (0.27 * daily_returns["BND"]) 
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
    years, gross_paths, net_paths = monte_carlo_sim()
    plot_fandown_charts(years, gross_paths, net_paths)
    plot_histogram(gross_paths, net_paths)


def monte_carlo_sim():
    # Monte Carlo simulation parameters 
    n_sims = 10000 
    years = 20 
    init_val = 1000000 
    mu_gross = 0.098071 
    mu_net = mu_gross - 0.0060 # 60 bps annual fee drag 
    sigma = 0.144087 
    np.random.seed(42) 
    # Simulate annual returns across 10,000 paths 
    gross_returns = np.random.normal(mu_gross, sigma, (years, n_sims)) 
    net_returns = np.random.normal(mu_net, sigma, (years, n_sims)) 
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
    
def plot_fandown_charts(years=20, gross_paths=10000, net_paths=10000):
    # 1. Calculate trajectory percentiles across time (axis=1) 
    years_arr = np.arange(years + 1) 
    gross_p5, gross_p50, gross_p95 = np.percentile(gross_paths, [5, 50, 95], axis=1) 
    net_p5, net_p50, net_p95 = np.percentile(net_paths, [5, 50, 95], axis=1) 
    # 2. Set up the visual styling with Seaborn 
    sns.set_theme(style="whitegrid") 
    fig, ax = plt.subplots(figsize=(11, 6)) 
    # Plot Gross paths (Benchmark Control) 
    ax.plot(years_arr, gross_p50, color="#1f77b4", linewidth=2.5, label="Gross Benchmark Median (0.0% Fee)") 
    ax.fill_between(years_arr, gross_p5, gross_p95, color="#1f77b4", alpha=0.15, label="Gross 90% Confidence Interval (5th-95th)")
    # Plot Net paths (60 bps Fee Drag) 
    ax.plot(years_arr, net_p50, color="#d62728", linewidth=2.5, linestyle="--", label="Net Portfolio Median (0.60% Annual Fee)") 
    ax.fill_between(years_arr, net_p5, net_p95, color="#d62728", alpha=0.15, label="Net 90% Confidence Interval (5th-95th)") 
    # 3. Format axes and labels 
    ax.set_title(f"20-Year Monte Carlo Wealth Trajectories (10,000 Sims, $1M Initial)", fontsize=13, fontweight="bold", pad=12) 
    ax.set_xlabel("Years Elapsed", fontsize=11) 
    ax.set_ylabel("Portfolio Value ($)", fontsize=11) 
    ax.set_xlim(0, years) 
    ax.yaxis.set_major_formatter('${x:,.0f}') 
    ax.legend(loc="upper left", frameon=True) 
    plt.tight_layout() 
    plt.show()
    
def plot_histogram(gross_paths=10000, net_paths=10000):
    # 1. Isolate year 20 terminal wealth arrays
    gross_terminal = gross_paths[-1] # last member 
    net_terminal = net_paths[-1] # last member
    
    #2. Plot overlapping distributions with KDE curves
    fig, ax = plt.subplots(figsize=(11, 6)) 
    sns.histplot(gross_terminal, bins=60, color="#1f77b4", alpha=0.35, kde=True, label="Gross Benchmark (0.0% Fee)", ax=ax) 
    sns.histplot(net_terminal, bins=60, color="#d62728", alpha=0.35, kde=True, label="Net Portfolio (0.60% Annual Fee)", ax=ax)
    
    # Add vertical median lines
    ax.axvline(np.median(gross_terminal), color="#1f77b4", linestyle="--", linewidth=2, label=f"Gross Median: ${np.median(gross_terminal):,.0f}") 
    ax.axvline(np.median(net_terminal), color="#d62728", linestyle="--", linewidth=2, label=f"Net Median: ${np.median(net_terminal):,.0f}")
    
    # Format axes and titles
    ax.set_title("Terminal Wealth Distribution at Year 20 (10,000 Simulations, $1M Initial)", fontsize=13, fontweight="bold", pad=12) 
    ax.set_xlabel("Terminal Portfolio Value ($)", fontsize=11) 
    ax.set_ylabel("Simulation Frequency", fontsize=11) 
    ax.xaxis.set_major_formatter('${x:,.0f}') 
    ax.set_xlim(0, np.percentile(gross_terminal, 98)) 
    ax.legend(loc="upper right", frameon=True) 
    plt.tight_layout() 
    plt.show()
    
    # 3. Calculate probability of shortfall
    shortfall_pct = np.mean(net_terminal < np.median(gross_terminal)) * 100 
    print(f"Percentage of net outcomes falling below gross benchmark median: {shortfall_pct:.1f}%")
    st.write(f"Percentage of net outcomes falling below gross benchmark median: {shortfall_pct:.1f}%")
      
    



if __name__ == '__main__':
    execute_pipeline()