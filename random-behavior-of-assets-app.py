import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats
import base64
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Random Behavior of Assets",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better mobile display
st.markdown("""
<style>
    .katex-html {
        overflow-x: auto;
        overflow-y: hidden;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        h1 {
            font-size: 1.8rem !important;
        }
        h2 {
            font-size: 1.5rem !important;
        }
        h3 {
            font-size: 1.2rem !important;
        }
        p, li {
            font-size: 0.9rem !important;
        }
        .stRadio label, .stCheckbox label {
            font-size: 0.9rem !important;
        }
    }
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        font-size: 0.8rem;
        color: #666;
    }
    .centered-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

# Create CC BY-NC license badge
def get_cc_badge():
    cc_image = '''
    <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">
    <img alt="Creative Commons License" style="border-width:0" 
    src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a>
    '''
    return cc_image

# Header
st.title("Understanding the Random Behavior of Assets")
st.markdown("### An Interactive Guide to Quantitative Finance")

# Introduction Section
st.markdown("""
This interactive guide explores how we can model the random behavior of financial assets.
You'll learn about:
- Why randomness is crucial in financial modeling
- How to examine and model asset returns
- The mathematics of asset price movements
- Building and simulating a basic model for asset prices

Use the navigation below to explore different concepts.
""")

# Create sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a topic:",
    ["Introduction", 
     "Importance of Randomness", 
     "Examining Returns", 
     "Asset Price Model", 
     "Simulation",
     "About & Disclaimers"]
)

# Introduction Page
if page == "Introduction":
    st.header("Introduction to Financial Asset Modeling")
    
    st.markdown("""
    In quantitative finance, we need mathematical models that can accurately represent the behavior of financial assets.
    The prices of equities, currencies, commodities, and indices all exhibit both predictable trends and unpredictable fluctuations.
    
    There are three common approaches to analyzing financial markets:

    1. **Fundamental Analysis**: Studying a company's financials, management, competitive position, etc. to determine its "true" value.
    
    2. **Technical Analysis**: Examining price patterns and trading volumes to predict future movements.
    
    3. **Quantitative Analysis**: Treating financial quantities as random variables and developing mathematical models to describe their behavior.
    
    This guide focuses on the quantitative approach, which has formed the foundation for modern portfolio theory, derivatives pricing, and risk management.
    """)
    
    st.subheader("Why Model Randomness?")
    st.markdown("""
    When looking at any financial time series, such as a stock price:
    - The future is never certain
    - Prices exhibit both trends and noise
    - Short-term movements are largely unpredictable
    - Long-term movements may follow identifiable patterns
    
    By building models that incorporate randomness, we can:
    - Create realistic simulations
    - Price derivative securities
    - Develop risk management strategies
    - Optimize investment portfolios
    """)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/S%26P_500_daily.png/800px-S%26P_500_daily.png", 
             caption="Example of a financial time series showing both trend and randomness")
    
# Importance of Randomness Page
elif page == "Importance of Randomness":
    st.header("Why Randomness Matters: Jensen's Inequality")
    
    st.markdown(r"""
    To understand why modeling randomness is crucial, especially for derivatives pricing, we need to look at Jensen's inequality.
    
    Consider a simple example: A stock is currently priced at $100. After one year, it could either be worth $50 or $150, with equal probability.
    
    There are two ways we might try to value a call option with a strike price of $100:
    
    **Approach 1**: Take the expected future stock price and calculate the option payoff.
    - Expected future stock price = $100 (average of $50 and $150)
    - Call option payoff = $\max(S - K, 0) = \max(100 - 100, 0) = 0$
    
    **Approach 2**: Calculate the option payoff for each scenario and take the average.
    - If stock goes to $50: Payoff = $\max(50 - 100, 0) = 0$
    - If stock goes to $150: Payoff = $\max(150 - 100, 0) = 50$
    - Expected payoff = $(0 + 50)/2 = 25$
    
    The second approach gives a much higher value! This is Jensen's inequality in action:
    """)
    
    st.latex(r"E[f(S)] \geq f(E[S])")
    
    st.markdown(r"""
    For a convex function $f(S)$ (like an option payoff) of a random variable $S$ (like a stock price), the expected value of the function is greater than or equal to the function of the expected value.
    
    This difference can be approximated as:
    """)
    
    st.latex(r"\frac{1}{2}f''(E[S])E[\epsilon^2]")
    
    st.markdown(r"""
    Where:
    - $f''(E[S])$ represents the convexity of the option
    - $E[\epsilon^2]$ represents the variance of the asset
    
    This shows why both option convexity and asset randomness (volatility) are crucial in options pricing.
    """)
    
    # Interactive demo of Jensen's inequality
    st.subheader("Interactive Demo: Jensen's Inequality")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        stock_price = st.slider("Current stock price ($)", 50, 150, 100)
        up_move = st.slider("Up move (%)", 10, 100, 50)
        down_move = st.slider("Down move (%)", 10, 100, 50)
        strike = st.slider("Option strike price ($)", 50, 150, 100)
    
    with col2:
        # Calculate results
        up_price = stock_price * (1 + up_move/100)
        down_price = stock_price * (1 - down_move/100)
        expected_price = (up_price + down_price) / 2
        
        payoff_at_expected = max(expected_price - strike, 0)
        payoff_up = max(up_price - strike, 0)
        payoff_down = max(down_price - strike, 0)
        expected_payoff = (payoff_up + payoff_down) / 2
        
        st.markdown(f"""
        **Results:**
        
        Expected stock price = ${expected_price:.2f}
        
        Payoff at expected price = ${payoff_at_expected:.2f}
        
        Expected payoff = ${expected_payoff:.2f}
        
        **Difference** = ${expected_payoff - payoff_at_expected:.2f}
        """)
    
    st.markdown("""
    This demonstrates why we need to model the full distribution of possible outcomes, not just the expected value.
    The randomness (volatility) of the asset directly affects the value of options and other derivatives.
    """)

# Examining Returns Page
elif page == "Examining Returns":
    st.header("Analyzing Asset Returns")
    
    st.markdown(r"""
    When modeling assets, we focus on returns rather than absolute prices because:
    
    1. Returns are more comparable across different assets
    2. Returns tend to have more stable statistical properties
    3. Investors care about percentage gains, not absolute gains
    
    ### Calculating Returns
    
    For an asset with price $S_i$ on day $i$, the simple return is:
    """)
    
    st.latex(r"R_i = \frac{S_{i+1} - S_i}{S_i}")
    
    st.markdown(r"""
    ### Statistical Properties of Returns
    
    From historical data, we can calculate:
    
    **Mean return:** 
    """)
    
    st.latex(r"\bar{R} = \frac{1}{M}\sum_{i=1}^{M}R_i")
    
    st.markdown("**Sample standard deviation:**")
    
    st.latex(r"\sqrt{\frac{1}{M-1}\sum_{i=1}^{M}(R_i - \bar{R})^2}")
    
    st.markdown(r"""
    ### Empirical Observation
    
    When analyzing real financial data, we typically observe:
    
    1. Mean daily return is much smaller than the standard deviation
    2. Returns often approximately follow a normal distribution
    3. Volatility tends to cluster (periods of high volatility are followed by more high volatility)
    
    Let's explore these properties with some simulated data.
    """)
    
    # Simulate some returns data
    np.random.seed(42)
    n_days = 500
    mu = 0.0005  # Mean daily return (about 12% annually)
    sigma = 0.01  # Daily volatility (about 16% annually)
    
    returns = np.random.normal(mu, sigma, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Day': np.arange(n_days),
        'Price': prices,
        'Return': np.append([0], np.diff(prices) / prices[:-1])
    })
    
    # Plot the data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Price plot
    ax1.plot(data['Day'], data['Price'])
    ax1.set_title('Simulated Asset Price')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Returns plot
    ax2.plot(data['Day'][1:], data['Return'][1:], linewidth=0.8)
    ax2.set_title('Daily Returns')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Return')
    ax2.grid(True)
    
    st.pyplot(fig)
    
    # Return statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mean Daily Return", f"{mean_return:.4f}")
        
    with col2:
        st.metric("Standard Deviation", f"{std_return:.4f}")
    
    st.markdown("""
    Notice how the mean return is much smaller than the standard deviation. This is typical in financial markets and indicates that in the short term, randomness (volatility) dominates the price movement. Only over longer time periods does the drift (average return) become apparent.
    """)
    
    # Histogram of returns
    st.subheader("Distribution of Returns")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(returns, bins=30, density=True, alpha=0.6, color='blue')
    
    # Plot normal distribution
    x = np.linspace(min(returns), max(returns), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
    
    ax.set_title('Histogram of Daily Returns vs. Normal Distribution')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    ax.legend(['Normal Distribution', 'Actual Returns'])
    
    st.pyplot(fig)
    
    st.markdown("""
    The histogram shows that returns often approximate a normal distribution, though real market data typically has fatter tails (more extreme events) than a normal distribution would predict.
    """)

    # Interactive section to generate different distributions
    st.subheader("Generate Your Own Return Distribution")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        custom_mu = st.slider("Mean daily return", -0.005, 0.005, 0.0005, 0.0001, format="%.4f")
        custom_sigma = st.slider("Daily volatility", 0.001, 0.05, 0.01, 0.001, format="%.3f")
        custom_days = st.slider("Number of days", 100, 1000, 500)
        distribution = st.selectbox("Distribution", ["Normal", "t-distribution (fatter tails)"])
        
    with col2:
        if st.button("Generate New Returns"):
            np.random.seed(None)  # Random seed
        
        if distribution == "Normal":
            custom_returns = np.random.normal(custom_mu, custom_sigma, custom_days)
        else:
            # t-distribution with 5 degrees of freedom (fatter tails)
            custom_returns = stats.t.rvs(df=5, loc=custom_mu, scale=custom_sigma, size=custom_days)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot histogram
        ax.hist(custom_returns, bins=30, density=True, alpha=0.6, color='blue')
        
        # Plot normal distribution for comparison
        x = np.linspace(min(custom_returns), max(custom_returns), 100)
        ax.plot(x, stats.norm.pdf(x, custom_mu, custom_sigma), 'r-', linewidth=2)
        
        ax.set_title('Generated Returns Distribution')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Calculate statistics
        mean_custom = np.mean(custom_returns)
        std_custom = np.std(custom_returns)
        skew_custom = stats.skew(custom_returns)
        kurt_custom = stats.kurtosis(custom_returns)
        
        st.markdown(f"""
        **Statistics of generated returns:**
        - Mean: {mean_custom:.6f}
        - Standard Deviation: {std_custom:.6f}
        - Skewness: {skew_custom:.4f} (0 for normal distribution)
        - Kurtosis: {kurt_custom:.4f} (0 for normal distribution, > 0 indicates fatter tails)
        """)

# Asset Price Model Page
elif page == "Asset Price Model":
    st.header("Developing an Asset Price Model")
    
    st.markdown(r"""
    Based on our observations of returns, we can develop a mathematical model for asset prices. We'll build this model step by step.
    
    ### The Random Walk Model
    
    For an asset with price $S_i$ at time step $i$, we model the return as:
    """)
    
    st.latex(r"R_i = \frac{S_{i+1} - S_i}{S_i} = \mu \delta t + \sigma \phi \sqrt{\delta t}")
    
    st.markdown(r"""
    Where:
    - $\mu$ is the drift (expected return)
    - $\sigma$ is the volatility (standard deviation of returns)
    - $\delta t$ is the time step
    - $\phi$ is a random number drawn from a standard normal distribution
    
    We can rewrite this to get a formula for the next price:
    """)
    
    st.latex(r"S_{i+1} = S_i(1 + \mu \delta t + \sigma \phi \sqrt{\delta t})")
    
    st.markdown(r"""
    ### Scaling with Time
    
    A key insight is how the mean and standard deviation of returns scale with time:
    
    - Mean return scales linearly with time: $\text{mean} = \mu \delta t$
    - Standard deviation scales with the square root of time: $\text{std dev} = \sigma \sqrt{\delta t}$
    
    This scaling ensures that our model remains meaningful as we make the time step smaller.
    
    ### Continuous-Time Limit: The Wiener Process
    
    As we take the time step $\delta t$ to zero, we get the continuous-time model:
    """)
    
    st.latex(r"dS = \mu S dt + \sigma S dX")
    
    st.markdown(r"""
    Where:
    - $dS$ is the infinitesimal change in the asset price
    - $dt$ is an infinitesimal time step
    - $dX$ is a Wiener process increment with $E[dX] = 0$ and $E[dX^2] = dt$
    
    This is a stochastic differential equation (SDE) known as Geometric Brownian Motion. It is the foundation of the Black-Scholes option pricing model and many other financial theories.
    
    ### Key Parameters
    
    **Drift ($\mu$)**: 
    - Represents the expected return of the asset
    - Usually quoted as an annualized percentage
    - Hard to estimate accurately from historical data due to noise
    
    **Volatility ($\sigma$)**:
    - Measures the standard deviation of returns
    - Usually quoted as an annualized percentage
    - More reliable to estimate than the drift
    - The most important parameter for options pricing
    
    Let's explore how these parameters affect the asset price path.
    """)
    
    # Interactive model parameter exploration
    st.subheader("Explore Asset Price Model Parameters")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        initial_price = st.slider("Initial price", 10, 200, 100)
        drift = st.slider("Annual drift (μ)", -0.2, 0.5, 0.1, 0.01)
        volatility = st.slider("Annual volatility (σ)", 0.01, 1.0, 0.2, 0.01)
        time_years = st.slider("Time horizon (years)", 0.1, 5.0, 1.0, 0.1)
        paths = st.slider("Number of paths", 1, 20, 5)
    
    # Simulate price paths
    dt = 1/252  # Daily time steps (252 trading days per year)
    n_steps = int(time_years / dt)
    time_points = np.linspace(0, time_years, n_steps)
    
    # Create price paths
    np.random.seed(42)  # For reproducibility
    all_paths = np.zeros((paths, n_steps))
    
    for i in range(paths):
        # Generate random returns
        returns = np.random.normal(drift*dt, volatility*np.sqrt(dt), n_steps)
        
        # Convert to price path (starting at initial_price)
        price_path = initial_price * np.exp(np.cumsum(returns))
        all_paths[i, :] = price_path
    
    # Plot the paths
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(paths):
        ax.plot(time_points, all_paths[i, :])
    
    # Add the expected path
    expected_path = initial_price * np.exp(drift * time_points)
    ax.plot(time_points, expected_path, 'k--', linewidth=2)
    
    ax.set_title('Simulated Asset Price Paths')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend(['Path ' + str(i+1) for i in range(paths)] + ['Expected Path'])
    
    with col2:
        st.pyplot(fig)
    
    st.markdown(r"""
    ### Model Characteristics
    
    1. **Short vs. Long Term**:
       - In the short term, randomness (volatility) dominates
       - In the long term, the drift becomes more important
    
    2. **Log-Normal Distribution**:
       - The model implies that asset prices follow a log-normal distribution
       - This ensures prices remain positive
    
    3. **Independence of Returns**:
       - The model assumes each price change is independent of past changes
       - This is a simplification, as real markets may show autocorrelation
    
    4. **Constant Volatility**:
       - The basic model assumes constant volatility
       - More sophisticated models allow for time-varying volatility
    
    Despite its simplifications, this model works remarkably well for many financial applications and forms the foundation of modern quantitative finance.
    """)

# Simulation Page
elif page == "Simulation":
    st.header("Asset Price Simulation")
    
    st.markdown("""
    Let's simulate asset price paths using our model and explore different scenarios. You can adjust the parameters to see how they affect the price trajectory.
    """)
    
    # Parameters input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        s0 = st.number_input("Initial price ($)", min_value=1.0, max_value=1000.0, value=100.0, step=10.0)
        mu = st.number_input("Annual drift (μ)", min_value=-0.5, max_value=0.5, value=0.1, step=0.01)
        
    with col2:
        sigma = st.number_input("Annual volatility (σ)", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
        T = st.number_input("Simulation period (years)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
    with col3:
        dt = st.selectbox("Time step", ["Daily", "Weekly", "Monthly"], index=0)
        n_paths = st.slider("Number of simulations", min_value=1, max_value=100, value=10)
    
    # Set time step based on selection
    if dt == "Daily":
        dt_value = 1/252
        steps_per_year = 252
    elif dt == "Weekly":
        dt_value = 1/52
        steps_per_year = 52
    else:  # Monthly
        dt_value = 1/12
        steps_per_year = 12
    
    # Number of time steps
    n_steps = int(T * steps_per_year)
    
    # Time points for x-axis
    time_points = np.linspace(0, T, n_steps + 1)
    
    # Run simulation button
    if st.button("Run Simulation"):
        # Clear previous state
        np.random.seed(None)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # Arrays to store results
        all_paths = np.zeros((n_paths, n_steps + 1))
        all_paths[:, 0] = s0  # Set initial price
        
        # Generate all paths
        for i in range(n_paths):
            # Path generation using the model
            for j in range(n_steps):
                all_paths[i, j+1] = all_paths[i, j] * (1 + mu * dt_value + sigma * np.sqrt(dt_value) * np.random.normal(0, 1))
        
        # Calculate statistics across paths
        mean_path = np.mean(all_paths, axis=0)
        median_path = np.median(all_paths, axis=0)
        std_path = np.std(all_paths, axis=0)
        
        # Confidence intervals
        upper_95 = mean_path + 1.96 * std_path / np.sqrt(n_paths)
        lower_95 = mean_path - 1.96 * std_path / np.sqrt(n_paths)
        
        # Plot 1: All paths
        for i in range(n_paths):
            ax1.plot(time_points, all_paths[i, :], linewidth=0.8, alpha=0.7)
            
        ax1.plot(time_points, s0 * np.exp(mu * time_points), 'k--', linewidth=2)
        ax1.set_title('Simulated Price Paths')
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        
        # Plot 2: Mean path with confidence interval
        ax2.plot(time_points, mean_path, 'b-', linewidth=2)
        ax2.fill_between(time_points, lower_95, upper_95, color='b', alpha=0.2)
        ax2.plot(time_points, s0 * np.exp(mu * time_points), 'k--', linewidth=2)
        ax2.set_title('Mean Price Path with 95% Confidence Interval')
        ax2.set_xlabel('Time (years)')
        ax2.set_ylabel('Price')
        ax2.grid(True)
        ax2.legend(['Mean Path', '95% Confidence Interval', 'Expected Path'])
        
        # Plot 3: Distribution of final prices
        final_prices = all_paths[:, -1]
        ax3.hist(final_prices, bins=30, density=True, alpha=0.6)
        
        # Add theoretical log-normal distribution
        x = np.linspace(min(final_prices) * 0.8, max(final_prices) * 1.2, 100)
        theoretical_mean = np.log(s0) + (mu - 0.5 * sigma**2) * T
        theoretical_std = sigma * np.sqrt(T)
        pdf = (1 / (x * theoretical_std * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - theoretical_mean)**2 / (2 * theoretical_std**2))
        ax3.plot(x, pdf, 'r-', linewidth=2)
        
        ax3.set_title('Distribution of Final Prices')
        ax3.set_xlabel('Price')
        ax3.set_ylabel('Density')
        ax3.grid(True)
        ax3.legend(['Theoretical Distribution', 'Simulated Distribution'])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Final Price", f"${s0 * np.exp(mu * T):.2f}")
            
        with col2:
            st.metric("Mean Simulated Final Price", f"${np.mean(final_prices):.2f}")
            
        with col3:
            st.metric("Std Dev of Final Price", f"${np.std(final_prices):.2f}")
        
        # Additional statistics
        st.markdown("""
        ### Return Distribution Statistics
        """)
        
        returns = (all_paths[:, 1:] / all_paths[:, :-1]) - 1
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Return per Time Step", f"{mean_return:.4%}")
            
        with col2:
            st.metric("Std Dev of Returns", f"{std_return:.4%}")
            
        with col3:
            annualized_return = mean_return * steps_per_year
            annualized_vol = std_return * np.sqrt(steps_per_year)
            st.metric("Annualized Return", f"{annualized_return:.2%}")
    
    st.markdown(r"""
    ### Understanding the Simulation
    
    The simulation implements the stochastic process:
    
    $$S_{i+1} = S_i(1 + \mu \delta t + \sigma \phi \sqrt{\delta t})$$
    
    Where $\phi$ is a random number drawn from a standard normal distribution.
    
    In the continuous limit, this becomes:
    
    $$dS = \mu S dt + \sigma S dX$$
    
    This model has an analytical solution:
    
    $$S_T = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)T + \sigma \sqrt{T} Z\right)$$
    
    Where $Z$ is a standard normal random variable. This implies that the logarithm of the asset price follows a normal distribution, or equivalently, that the asset price follows a log-normal distribution.
    """)
    
    st.subheader("Interactive Experiment: Central Limit Theorem")
    
    st.markdown("""
    One reason we use the normal distribution in our model is the Central Limit Theorem, which states that the sum of many independent random variables tends toward a normal distribution.
    
    Let's demonstrate this with a simple coin tossing experiment:
    """)
    
    n_tosses = st.slider("Number of coin tosses", 1, 1000, 10)
    n_experiments = 10000
    
    if st.button("Run Experiment"):
        # Simulate coin tosses (1 = heads, -1 = tails)
        tosses = np.random.choice([1, -1], size=(n_experiments, n_tosses))
        sums = np.sum(tosses, axis=1)
        
        # Plot histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(sums, bins=np.arange(min(sums)-0.5, max(sums)+1.5), density=True, alpha=0.6)
        
        # Add normal curve for comparison
        x = np.linspace(min(sums), max(sums), 100)
        mean = 0  # Expected value for fair coin
        std = np.sqrt(n_tosses)  # Standard deviation for coin tosses
        normal_pdf = stats.norm.pdf(x, mean, std)
        
        ax.plot(x, normal_pdf, 'r-', linewidth=2)
        
        ax.set_title(f'Sum of {n_tosses} Coin Tosses (Heads=+1, Tails=-1)')
        ax.set_xlabel('Sum')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        ax.legend(['Normal Distribution', 'Simulated Distribution'])
        
        st.pyplot(fig)
        
        st.markdown(f"""
        As the number of tosses increases, the distribution approaches a normal distribution with 
        mean = {mean} and standard deviation = {std:.2f}.
        
        This demonstrates why the normal distribution appears naturally in finance - asset returns can be viewed as 
        the cumulative result of many small, independent random influences.
        """)

# About & Disclaimers Page
elif page == "About & Disclaimers":
    st.header("About This Guide")
    
    st.markdown("""
    ### Author
    
    **Luís Simões da Cunha**
    
    This interactive guide was created to help understand the fundamental concepts 
    of asset price modeling and stochastic processes in finance.
    """)
    
    # CC-BY-NC License
    st.markdown("### License")
    st.markdown(get_cc_badge(), unsafe_allow_html=True)
    st.markdown("""
    This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).
    
    You are free to:
    - Share — copy and redistribute the material in any medium or format
    - Adapt — remix, transform, and build upon the material
    
    Under the following terms:
    - Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
    - NonCommercial — You may not use the material for commercial purposes.
    """)
    
    # Disclaimers
    st.markdown("### Disclaimers")
    st.markdown("""
    **Financial Disclaimer:**
    
    The information provided in this guide is for educational purposes only and does not constitute financial advice. The author is not a financial advisor. Before making any investment decisions, please consult with a qualified financial professional. Past performance is not indicative of future results.
    
    **Accuracy Disclaimer:**
    
    While every effort has been made to ensure the accuracy of the information presented, the author makes no warranties or representations as to its completeness or accuracy. The financial markets are complex and unpredictable, and no model can accurately predict future market movements with certainty.
    
    **Risk Warning:**
    
    Investing in financial markets involves risk. The value of investments may go down as well as up, and investors may lose some or all of their invested capital. The models presented in this guide are simplifications of reality and may not capture all relevant factors affecting asset prices.
    
    **Use of Information:**
    
    By using this guide, you acknowledge that you are responsible for your own investment decisions and that you will not hold the author liable for any losses or damages resulting from the use of information contained herein.
    """)
    
    # References
    st.markdown("### Further Reading")
    st.markdown("""
    For those interested in exploring these concepts further, consider researching:
    
    1. Stochastic Calculus and Financial Applications
    2. Option Pricing Theory and the Black-Scholes Model
    3. Statistical Analysis of Financial Time Series
    4. Monte Carlo Methods in Finance
    5. Volatility Models and Forecasting
    """)

# Add footer
st.markdown("""
<div class="footer">
    <p>© 2023 Luís Simões da Cunha. This work is licensed under CC BY-NC 4.0.</p>
    <p>Created for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)