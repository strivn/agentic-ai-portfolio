#!/usr/bin/env python3
"""
Financial Agent Tools

This module provides a set of tools for financial analysis and portfolio management:
- Stock Price Lookup Tool: Get current price of a stock
- Portfolio Rebalancing Tool: Analyze and rebalance investment portfolios
- Market Trend Analysis Tool: Analyze recent market trends

Usage:
    python financial_agent.py
"""
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


# Function to get the LLM
def get_llm(llm_provider="openai", llm_model="default"):
    """
    Returns an LLM instance based on the provider and model specified.

    Args:
        llm_provider (str): The LLM provider (e.g., "openai", "groq")
        llm_model (str): The model to use

    Returns:
        An LLM instance
    """

    load_dotenv('../.env')
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if llm_provider.lower() == "openai":
        return ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY,
        )
    elif llm_provider.lower() == "groq":
        if llm_model.lower() == "llama3-70b":
            return ChatGroq(
                model_name="llama3-70b-8192",
                temperature=0,
                api_key=GROQ_API_KEY,
            )
        else:
            return ChatGroq(
                model_name="llama3-8b-8192",
                temperature=0,
                api_key=GROQ_API_KEY,
            )
    else:
        return ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY,
        )  # default


# Task 1: Implement Stock Price Lookup Tool
def get_stock_price(symbol):
    """
    Fetches the latest stock price for the given symbol.

    Args:
        symbol (str): A single stock ticker symbol (e.g., AAPL or META). Only the ticker (no additional symbols like $ or quotes).

    Returns:
        dict: Stock information including current price and daily change
    """
    try:
        # data cleansing just in case. 
        symbol = symbol.replace("$", "").replace("'", "")
        
        # Create a Ticker object
        ticker = yf.Ticker(symbol)

        # Get the latest stock data
        hist = ticker.history(period="1d")

        if hist.empty:
            return f"No data found for symbol {symbol}"

        # Get info for additional data
        info = ticker.fast_info

        # Extract relevant information
        current_price = info.last_price if hasattr(
            info, 'last_price') else hist['Close'].iloc[-1]
        previous_close = info.previous_close if hasattr(
            info, 'previous_close') else None

        # Calculate daily change if possible
        if previous_close:
            daily_change = current_price - previous_close
            daily_change_percent = (daily_change / previous_close) * 100
        else:
            daily_change = None
            daily_change_percent = None

        # Return the data in a structured format
        result = {
            "symbol": symbol,
            "current_price": current_price,
            "previous_close": previous_close,
            "daily_change": daily_change,
            "daily_change_percent": daily_change_percent,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return result

    except Exception as e:
        return f"Error retrieving stock price for {symbol}: {str(e)}"


# Create the stock price lookup tool
stock_price_tool = Tool(
    name="StockPriceLookup",
    func=get_stock_price,
    description="Useful for getting the current price of a stock. Input should be a single valid stock ticker symbol (e.g. AAPL)."
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
        # Parse the portfolio string into a dictionary
        # Remove any single quotes inside the string to avoid parsing errors
        portfolio_str = portfolio_str.replace("'", "\"")

        # Handle case where input is already a dictionary
        if isinstance(portfolio_str, dict):
            portfolio = portfolio_str
        else:
            import json
            portfolio = json.loads(portfolio_str)

        # Get the number of assets
        num_assets = len(portfolio)

        # For equal weighting, each asset should have 1/num_assets weight
        target_weight = 1.0 / num_assets

        # Calculate rebalancing actions
        rebalancing_actions = {}
        recommendations = []

        for symbol, weight in portfolio.items():
            weight_diff = target_weight - weight
            rebalancing_actions[symbol] = weight_diff

            if abs(weight_diff) < 0.01:  # Threshold of 1%
                action = "maintain"
            elif weight_diff > 0:
                action = "buy"
            else:
                action = "sell"

            recommendations.append(
                f"{symbol}: Current weight {weight:.2f}, target weight {target_weight:.2f}, action: {action} {abs(weight_diff):.2f}")

        # Return recommendations in a structured format
        result = "Portfolio Rebalancing Recommendations (Equal Weight Strategy):\n"
        result += "\n".join(recommendations)

        return result

    except Exception as e:
        return f"Error analyzing portfolio: {str(e)}"


# Create the portfolio rebalancing tool
rebalance_tool = Tool(
    name="PortfolioRebalancer",
    func=rebalance_portfolio,
    description="Analyzes a portfolio and suggests rebalancing actions. Input should be a dictionary mapping stock symbols to their current weight in the portfolio."
)


# Task 3: Implement Market Trend Analysis Tool
def market_trend_analysis(dummy_input):
    """
    Fetches stock market index trends over the past week.

    Args:
        N/A. The function need no input or parameter.

    Returns:
        str: Analysis of market trends
    """
    try:
        # Use SPY as a proxy for S&P 500
        spy = yf.Ticker("SPY")

        # Get data for the past 5 trading days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        # Fetch historical data
        hist = spy.history(start=start_date, end=end_date)

        if hist.empty:
            return "No market data available for the past week."

        # Calculate key metrics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        period_return = ((end_price / start_price) - 1) * 100

        # Calculate volatility (standard deviation of daily returns)
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * 100

        # Determine market trend
        if period_return > 1:
            trend = "strongly bullish"
        elif period_return > 0:
            trend = "mildly bullish"
        elif period_return > -1:
            trend = "mildly bearish"
        else:
            trend = "strongly bearish"

        # Format the result
        result = f"Market Trend Analysis (Past 5 Trading Days):\n"
        result += f"S&P 500 5-day return: {period_return:.2f}%\n"
        result += f"Volatility: {volatility:.2f}%\n"
        result += f"Market trend: {trend}\n"
        result += f"Start price: ${start_price:.2f}\n"
        result += f"Current price: ${end_price:.2f}\n"
        result += f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return result

    except Exception as e:
        return f"Error analyzing market trends: {str(e)}"


# Create the market trend analysis tool
trend_tool = Tool(
    name="MarketTrendAnalyzer",
    func=market_trend_analysis,
    description="Provides an analysis of recent market trends. No input required. Do not pass any parameters in"
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
    
    system_prompt = """
    You are a personal finance AI assistant. Your task is to analyze the given portfolio and recommend changes. 
    You may do the following:

    1. Retrieving stock prices for assets in a given portfolio
    2. Checking if the portfolio is imbalanced based on an equal-weight strategy
    3. Recommending actions if the portfolio needs rebalancing
    
    Example:
    A user has AAPL (50%), TSLA (30%), and GOOGL (20%) in their portfolio
    You should recognize that AAPL is overweight and suggest a rebalance to equal 33% per stock

    Analyze this portfolio and recommend changes: 
    """
    
    start_time = time.time()
    print("\n================================================================")
    print(f"start_time: {start_time}")
        
    # Test with OpenAI
    print("\n================================================================")
    print("OpenAI GPT-4 Results:")
    print("Portfolio 1:", create_and_run_agent("openai", "default",
          f"{system_prompt}  {user_portfolio_1}"))
    print("Portfolio 2:", create_and_run_agent("openai", "default",
          f"{system_prompt}  {user_portfolio_2}"))
    print(f"GPT-4o-Mini time: {time.time() - start_time}")
    start_time = time.time()

    # Test with Groq LLaMA3-8B
    print("\n================================================================")
    print("\nGroq LLaMA3-8B Results:")
    print("Portfolio 1:", create_and_run_agent("groq", "llama3-8b",
          f"{system_prompt}  {user_portfolio_1}"))
    print("Portfolio 2:", create_and_run_agent("groq", "llama3-8b",
          f"{system_prompt}  {user_portfolio_2}"))
    print(f"GPT-4o-Mini time: {time.time() - start_time}")
    start_time = time.time()

    # Test with Groq LLaMA3-70B
    print("\n================================================================")
    print("\nGroq LLaMA3-70B Results:")
    print("Portfolio 1:", create_and_run_agent("groq", "llama3-70b",
          f"{system_prompt}  {user_portfolio_1}"))
    print("Portfolio 2:", create_and_run_agent("groq", "llama3-70b",
          f"{system_prompt}  {user_portfolio_2}"))
    print(f"GPT-4o-Mini time: {time.time() - start_time}")


if __name__ == "__main__":
    run_test_cases()
