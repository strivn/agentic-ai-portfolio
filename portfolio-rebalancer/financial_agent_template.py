#!/usr/bin/env python3
"""
Financial Agent Tools - Student Template

This module provides a template for implementing a set of financial analysis tools:
- Stock Price Lookup Tool: Get current price of a stock
- Portfolio Rebalancing Tool: Analyze and rebalance investment portfolios
- Market Trend Analysis Tool: Analyze recent market trends

Students should implement the functions marked with "STUDENT IMPLEMENTATION"

Usage:
    python financial_agent_template.py
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from langchain.agents import Tool, AgentType, initialize_agent

# Function to get the LLM (placeholder)
def get_llm(llm_provider="openai", llm_model="default"):
    """
    Returns an LLM instance based on the provider and model specified.
    
    Args:
        llm_provider (str): The LLM provider (e.g., "openai", "groq")
        llm_model (str): The model to use
        
    Returns:
        An LLM instance
    """
    # This is a placeholder - replace with actual implementation
    # In a real implementation, you would import the proper libraries
    # and return the appropriate LLM instance
    if llm_provider == "openai":
        # Replace with actual OpenAI LLM initialization
        return "OpenAI LLM"
    elif llm_provider == "groq":
        if llm_model == "llama3-8b":
            # Replace with actual Groq LLaMA3-8B initialization
            return "Groq LLaMA3-8B"
        elif llm_model == "llama3-70b":
            # Replace with actual Groq LLaMA3-70B initialization
            return "Groq LLaMA3-70B"
    
    # Default case
    return "Default LLM"


# Task 1: Implement Stock Price Lookup Tool
def get_stock_price(symbol):
    """
    Fetches the latest stock price for the given symbol.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        dict: Stock information including current price and daily change
    """
    try:
        # STUDENT IMPLEMENTATION:
        # 1. Use yfinance to fetch the latest stock data
        # 2. Extract relevant information like current price and daily change
        # 3. Return the data in a structured format
        pass
    
    except Exception as e:
        return f"Error retrieving stock price for {symbol}: {str(e)}"


# Create the stock price lookup tool
stock_price_tool = Tool(
    name="StockPriceLookup",
    func=get_stock_price,
    description="Useful for getting the current price of a stock. Input should be a valid stock ticker symbol."
)


# Task 2: Implement Portfolio Rebalancing Tool
def rebalance_portfolio(portfolio_str):
    """
    Takes a portfolio string representation and suggests rebalancing actions.
    
    Args:
        portfolio_str (str): String representation of portfolio, e.g., "{'AAPL': 0.5, 'TSLA': 0.3, 'GOOGL': 0.2}"
        
    Returns:
        str: Rebalancing recommendations
    """
    try:
        # STUDENT IMPLEMENTATION:
        # 1. Parse the portfolio string into a dictionary
        # 2. Check if the portfolio is balanced (equal weight for this assignment)
        # 3. Suggest buying or selling actions to achieve balance
        # 4. Return recommendations in a clear, structured format
        pass
    
    except Exception as e:
        return f"Error analyzing portfolio: {str(e)}"


# Create the portfolio rebalancing tool
rebalance_tool = Tool(
    name="PortfolioRebalancer",
    func=rebalance_portfolio,
    description="Analyzes a portfolio and suggests rebalancing actions. Input should be a dictionary mapping stock symbols to their current weight in the portfolio."
)


# Task 3: Implement Market Trend Analysis Tool
def market_trend_analysis():
    """
    Fetches stock market index trends over the past week.
    
    Returns:
        str: Analysis of market trends
    """
    try:
        # STUDENT IMPLEMENTATION:
        # 1. Use yfinance to fetch data for a market index (e.g., SPY for S&P 500)
        # 2. Calculate key metrics like 5-day return, volatility, etc.
        # 3. Return a summary of the market trend
        pass
    
    except Exception as e:
        return f"Error analyzing market trends: {str(e)}"


# Create the market trend analysis tool
trend_tool = Tool(
    name="MarketTrendAnalyzer",
    func=market_trend_analysis,
    description="Provides an analysis of recent market trends. No input required."
)


# Function to create and run an agent with the selected LLM
def create_and_run_agent(llm_provider="openai", llm_model="default", query=""):
    """
    Creates and runs an agent with the selected LLM and tools.
    
    Args:
        llm_provider (str): The LLM provider to use
        llm_model (str): The model to use
        query (str): The query to run
        
    Returns:
        str: The result of the agent's execution
    """
    llm = get_llm(llm_provider, llm_model)
    
    tools = [stock_price_tool, rebalance_tool, trend_tool]
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent.run(query)


# Test cases with different LLMs
def run_test_cases():
    
    """Runs test cases for the agent with different LLMs and portfolios."""
    user_portfolio_1 = {"AAPL": 0.50, "TSLA": 0.30, "GOOGL": 0.20} 
    user_portfolio_2 = {"MSFT": 0.25, "NVDA": 0.25, "AMZN": 0.25, "META": 0.25}
    
    # Test with OpenAI
    print("OpenAI GPT-4 Results:")
    print("Portfolio 1:", create_and_run_agent("openai", "default", f"Analyze this portfolio and recommend changes: {user_portfolio_1}"))
    print("Portfolio 2:", create_and_run_agent("openai", "default", f"Analyze this portfolio and recommend changes: {user_portfolio_2}"))
    
    # Test with Groq LLaMA3-8B
    print("\nGroq LLaMA3-8B Results:")
    print("Portfolio 1:", create_and_run_agent("groq", "llama3-8b", f"Analyze this portfolio and recommend changes: {user_portfolio_1}"))
    print("Portfolio 2:", create_and_run_agent("groq", "llama3-8b", f"Analyze this portfolio and recommend changes: {user_portfolio_2}"))
    
    # Test with Groq LLaMA3-70B
    print("\nGroq LLaMA3-70B Results:")
    print("Portfolio 1:", create_and_run_agent("groq", "llama3-70b", f"Analyze this portfolio and recommend changes: {user_portfolio_1}"))
    print("Portfolio 2:", create_and_run_agent("groq", "llama3-70b", f"Analyze this portfolio and recommend changes: {user_portfolio_2}"))


if __name__ == "__main__":
    run_test_cases()
