import numpy as np
from scipy.stats import norm, percentileofscore
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Parameters
S0 = 2.50  # Initial Underlying Price
K = 2.60  # Strike Price
T_days = 90  # Total duration in days
T = T_days / 365.0
r = 0.02  # Risk-free rate
sigma_target = 0.10
sigma_base = 0.30  # Long-term mean volatility
theta = 50.0  # Mean reversion speed (Increased for stronger reversion)
sigma_vol = 0.60  # Volatility of volatility (Increased for wider spread)
N_SIMULATIONS = 1000
N_STEPS = 90  # 1 step per day

dt = T / N_STEPS


def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return np.maximum(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def simulate_path(seed):
    np.random.seed(seed)
    prices = np.zeros(N_STEPS + 1)
    sigmas = np.zeros(N_STEPS + 1)
    option_prices = np.zeros(N_STEPS + 1)

    prices[0] = S0
    sigmas[0] = sigma_base
    option_prices[0] = black_scholes_call(S0, K, T, r, sigma_base)

    # Independent Brownian Motions
    Z_S = np.random.standard_normal(N_STEPS)
    Z_vol = np.random.standard_normal(N_STEPS)

    for t in range(N_STEPS):
        # OU Process for Volatility
        # d_sigma = theta * (mu - sigma) * dt + vol_vol * dW
        d_sigma = (
            theta * (sigma_base - sigmas[t]) * dt + sigma_vol * np.sqrt(dt) * Z_vol[t]
        )
        sigmas[t + 1] = max(
            0.05, sigmas[t] + d_sigma
        )  # Reflecting or truncating at 0.05

        # GBM for Price using the *current* volatility for the step
        # dS = r * S * dt + sigma * S * dW
        prices[t + 1] = prices[t] * np.exp(
            (r - 0.5 * sigma_target**2) * dt + sigma_target * np.sqrt(dt) * Z_S[t]
        )

        # Option Price
        time_left = T - (t + 1) * dt
        if time_left < 1e-9:
            option_prices[t + 1] = np.maximum(prices[t + 1] - K, 0)
        else:
            # Use the updated volatility for pricing
            option_prices[t + 1] = black_scholes_call(
                prices[t + 1], K, time_left, r, sigmas[t + 1]
            )

    return prices, sigmas, option_prices


# Run Simulations
print(f"Running {N_SIMULATIONS} simulations...")
results = Parallel(n_jobs=-1)(delayed(simulate_path)(i) for i in range(N_SIMULATIONS))

# --- 1. Plot Paths (Price, Option, Sigma) ---
print("Generating path plots...")
all_prices = np.array([r[0] for r in results])
all_sigmas = np.array([r[1] for r in results])
all_option_prices = np.array([r[2] for r in results])
time_points = np.arange(N_STEPS + 1)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(time_points, all_prices.T, alpha=0.05, color="#708946")
plt.title("Underlying Price Paths (GBM)")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(time_points, all_option_prices.T, alpha=0.05, color="#326DC6")
plt.title("Option Price Paths")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(time_points, all_sigmas.T, alpha=0.05, color="#BE5A25")
plt.title("Implied Volatility Paths (OU Process)")
plt.ylabel("Volatility")
plt.xlabel("Days")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./output/monte_carlo_paths.png", dpi=300)
print("Path plots saved to ./output/monte_carlo_paths.png")

# --- 2. Analysis: IV Rank vs IV Change Rate ---
day_base_idx = 30
day_obs_idx = 40  # 30 + 15

iv_ranks = []
price_changes = []

print("Processing results...")
for i in range(N_SIMULATIONS):
    prices, sigmas, option_prices = results[i]

    # Calculate 30-day rolling IV quantile at Day 30
    # History: Day 0 to Day 30 (inclusive)
    iv_history = sigmas[: day_base_idx + 1]
    current_iv = sigmas[day_base_idx]

    # Percentile of current_iv within its history
    rank = percentileofscore(iv_history, current_iv) / 100.0
    iv_ranks.append(rank)

    # Calculate IV Change Rate 15 days later
    price_base = option_prices[day_base_idx]
    price_obs = option_prices[day_obs_idx]

    if price_base < 1e-6:
        change = 0.0
    else:
        #change = (price_obs - price_base) / price_base
        change = np.log(price_obs / price_base)
    price_changes.append(change)

# Plotting Scatter
plt.figure(figsize=(10, 8))
plt.scatter(iv_ranks, price_changes, alpha=0.5, c="purple", edgecolors="w", s=50)
plt.title(
    f"IV Rank (Day 30) vs Option Price Change Rate (Next 10 Days)\nSimulations: {N_SIMULATIONS}, Mean Reversion Theta: {theta}"
)
plt.xlabel("Implied Volatility Percentile Rank (Day 0-30)")
plt.ylabel("Option Price Change Rate (Day 30 -> Day 40)")
plt.axhline(0, color="black", linestyle="--", alpha=0.7)
plt.axvline(0.5, color="black", linestyle="--", alpha=0.7)
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(iv_ranks, price_changes, 1)
p = np.poly1d(z)
plt.plot(
    iv_ranks, p(iv_ranks), "r--", alpha=0.8, label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}"
)
plt.legend()

output_path = "./output/iv_mean_reversion_analysis.png"
plt.savefig(output_path, dpi=300)
print(f"Analysis complete. Plot saved to {output_path}")
