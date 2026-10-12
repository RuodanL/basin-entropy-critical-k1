#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 13:05:31 2026

@author: rliu8
"""

from math import lcm, gcd, log, sqrt, pi, fsum
from pathlib import Path
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "axes.unicode_minus": False,
})

def loop_lengths(perm):
    m = len(perm)
    lengths = []
    for i in range(m):
        if perm[i] < 0:
            continue
        j = i
        length = 0
        while perm[j] >= 0:
            nxt = perm[j]
            perm[j] = -1
            j = nxt
            length += 1
        lengths.append(length)
    return lengths

def random_max_attractor_length(m, rng=None, return_loop_info=False):
    if rng is None:
        rng = np.random.default_rng()
    p = rng.permutation(m)
    lengths = loop_lengths(p)
    n_cycles = len(lengths)
    multipliers = rng.integers(1, 3, n_cycles)
    max_attractor_length_per_loop = np.multiply(lengths, multipliers)
    max_attractor_length = lcm(*max_attractor_length_per_loop)
    if return_loop_info:
        return (max_attractor_length, lengths, multipliers)
    return max_attractor_length

def random_max_attractor_length_given_N(N, nsim=1, rng=None, return_loop_info=False):
    if rng is None:
        rng = np.random.default_rng()
    factors = np.linspace(1 / N, 1, N)
    p_core_size = np.multiply(np.cumprod(factors[::-1]), factors)
    random_core_sizes = rng.choice(N, size=nsim, p=p_core_size) + 1
    if return_loop_info:
        return [random_max_attractor_length(m, rng=rng, return_loop_info=True) for m in random_core_sizes]
    max_attractor_lengths = [random_max_attractor_length(m, rng=rng) for m in random_core_sizes]
    return max_attractor_lengths

def prime_factors(n):
    n = int(n)
    factors = {}
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        factors[n] = 1
    return factors

def probability_missing_prime_power(m_i, multiplier, p, a):
    m_i = int(m_i)
    r_i = m_i * int(multiplier)
    b = prime_factors(r_i).get(p, 0)
    if a > b:
        return 1.0
    t = r_i // p ** (b - a + 1)
    g = gcd(t, m_i)
    if multiplier == 2 and t // g % 2 == 1:
        return 0.0
    return 2.0 ** (g - m_i)

def basin_entropy(lengths, multipliers, Amax=None):
    lengths = [int(n) for n in lengths]
    multipliers = [int(s) for s in multipliers]
    m = sum(lengths)
    L = lcm(*lengths)
    if Amax is None:
        Amax = lcm(*(n * s for n, s in zip(lengths, multipliers)))
    exponents = {}
    for n, s in zip(lengths, multipliers):
        for p, a in prime_factors(n * s).items():
            exponents[p] = max(exponents.get(p, 0), a)
    terms = []
    for p, a_p in exponents.items():
        for a in range(1, a_p + 1):
            probability = 1.0
            for m_i, multiplier in zip(lengths, multipliers):
                probability *= probability_missing_prime_power(m_i, multiplier, p, a)
            terms.append(log(p) * probability)
    delta = fsum(terms)
    mean_log_A = log(Amax) - delta
    return dict(m=m, log_L=log(L), log_Amax=log(Amax), Amax_over_L=int(Amax // L), delta=delta, mean_log_A=mean_log_A, H=m * log(2) - mean_log_A)

def mean_number_of_relevant_nodes(N):
    factors = np.linspace(1 / N, 1, N)
    p_core_size = np.multiply(np.cumprod(factors[::-1]), factors)
    return float(np.dot(np.arange(1, N + 1), p_core_size))

@lru_cache(maxsize=2)
def core_size_distribution_efficient(N, tail_tol=1e-14):
    N = int(N)
    probabilities = []
    survival = 1.0
    for m in range(1, N + 1):
        probability_m = survival * m / N
        probabilities.append(probability_m)
        survival *= 1.0 - m / N
        if survival < tail_tol:
            break
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= probabilities.sum()
    core_sizes = np.arange(1, len(probabilities) + 1, dtype=int)
    return (core_sizes, probabilities)

def random_max_attractor_length_given_N_efficient(N, nsim=1, rng=None, return_loop_info=False):
    if rng is None:
        rng = np.random.default_rng()
    core_sizes, p_core_size = core_size_distribution_efficient(int(N))
    random_core_sizes = rng.choice(core_sizes, size=nsim, p=p_core_size)
    if return_loop_info:
        return [random_max_attractor_length(int(m), rng=rng, return_loop_info=True) for m in random_core_sizes]
    return [random_max_attractor_length(int(m), rng=rng) for m in random_core_sizes]

def mean_number_of_relevant_nodes_efficient(N):
    core_sizes, p_core_size = core_size_distribution_efficient(int(N))
    return float(np.dot(core_sizes, p_core_size))

def save_figure(f, prefix, name):
    filename = f"{prefix}_{name}.pdf"
    f.savefig(filename, dpi=300, bbox_inches="tight")
    print("Saved:", filename, flush=True)


def plot_figure_i(ms, delta_data, sharp_data, C_star, prefix):
    # Merge locations to 12 decimal places for display, not for computation.
    points = []
    for m, row in zip(ms, delta_data):
        y, counts = np.unique(np.round(row, 12), return_counts=True)
        points.extend(zip(np.full(y.size, m), y, counts))
    x, y, counts = np.asarray(points).T
    colors = np.zeros((len(x), 4))
    colors[:, 3] = (0.15 + 0.75 * np.log(counts) / np.log(counts.max())
                    if counts.max() > 1 else 0.4)

    f, ax = plt.subplots(figsize=(3.375, 2.5), layout="constrained")
    ax.scatter(x, y, s=9, c=colors, linewidths=0)
    ax.axhline(C_star, color="k", ls="--", lw=1)
    sharp = [s for s in sharp_data if s["m"] <= max(ms)]
    ax.step([s["m"] for s in sharp] + [max(ms)],
            [s["delta"] for s in sharp] + [sharp[-1]["delta"]],
            where="post", color="r", lw=1, label="Sharp upper bound")
    ax.legend(loc="upper center", bbox_to_anchor=(0.68, 0.94), frameon=False, fontsize=8, handlelength=2.5)
    ax.set(xscale="log", ylim=(-0.025, 0.8))
    ax.set_xlabel(r"$m$, number of relevant nodes", math_fontfamily="cm")
    #ax.set_ylabel(r"$\ln \ell_{\max}-\mathbb{E}_X[\ln \ell(X)]$", math_fontfamily="cm")
    ax.set_ylabel(r"Maximal-mean log-period gap")
    ax.set_yticks([0, .2, .4, .6, C_star], ["0", "0.2", "0.4", "0.6", r"$C_*$"])
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(f, prefix, "figure_i")
    print(f"Figure i: {delta_data.size} samples -> {len(x)} displayed locations")
    return f, ax


def plot_figure_ii(order_data, m_fixed, C_star, prefix):
    f, ax = plt.subplots(figsize=(3.375, 2.5), layout="constrained")
    valid = [s for s in order_data if s["log_L"] > 0]
    xs = [s["log_L"] for s in valid]
    x = np.linspace(max(min(xs)*0.9, min(xs)-0.5), max(xs)+0.5, 200)
    for ratio, color in [(1, blue), (2, orange)]:
        selected = [s for s in valid if s["Amax_over_L"] == ratio]
        label = (r"$\ell_{\max}=\operatorname{ord}(\sigma)$" if ratio == 1 else r"$\ell_{\max}=2\operatorname{ord}(\sigma)$")
        ax.scatter([s["log_L"] for s in selected],
                   [s["mean_log_A"]/s["log_L"] for s in selected],
                   s=10, alpha=.35, color=color, linewidths=0,
                   label=label)
    ax.axhline(1, color="k", ls=":", lw=.5)
    ax.plot(x, 1-C_star/x, "k--", lw=1)
    ax.plot(x, 1+log(2)/x, "k--", lw=1)
    t = x.min() + .55*np.ptp(x)
    for y, label, offset, va in [(1+log(2)/t, r"$y=1+\frac{\ln2}{\ln\operatorname{ord}(\sigma)}$", 8, "bottom"),
                                  (1-C_star/t, r"$y=1-\frac{C_*}{\ln\operatorname{ord}(\sigma)}$", -8, "top")]:
        ax.annotate(label, (t, y), xytext=(0, offset), textcoords="offset points",
                    ha="center", va=va, fontsize=8, math_fontfamily="cm")
    ax.set_xlabel(r"$\ln\operatorname{ord}(\sigma)$, log permutation order", math_fontfamily="cm")
    #ax.set_ylabel(r"$(m\ln2-H)/\ln\operatorname{ord}(\sigma)$", math_fontfamily="cm")
    ax.set_ylabel("Gap from maximum entropy\n" + r"(normalized by $\ln\operatorname{ord}(\sigma)$)", math_fontfamily="cm")
    #ax.text(.97, .05, f"$m={m_fixed}$", transform=ax.transAxes, ha="right", fontsize=8)
    legend = ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2,
              frameon=False, fontsize=8, markerscale=2)
    for text in legend.get_texts():
        text.set_math_fontfamily("cm")
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(f, prefix, "figure_ii")
    return f, ax


def plot_figure_iii(Ns, remainder_samples, prefix):
    x = np.log(Ns) * np.log(np.log(Ns))
    mean = remainder_samples.mean(axis=1)
    low, high = np.percentile(remainder_samples, percentiles, axis=1)
    band = f"{percentiles[0]:g}%–{percentiles[1]:g}% percentile"
    f, ax = plt.subplots(2, 1, figsize=(3.375, 4.5), sharex=True)
    #f.subplots_adjust(left=0.24, right=0.97, bottom=0.30, top=0.83, hspace=0.35)
    ax[0].plot(x, mean, "o-", color=blue, ms=3, lw=1, label="Ensemble mean")
    ax[0].fill_between(x, low, high, color=blue, alpha=.15, label=band)
    ax[0].plot([0, x.max()], [0, x.max()/2], "--", color=orange,
               lw=1)#, label=r"$(\ln N)^2/8$")
    t = 0.45 * x.max()
    ax[0].annotate(
        r"$y = \frac{1}{2}\ln N\,\ln\ln N$",
        math_fontfamily="cm",
        xy=(t, t/2),       # arrow tip
        xytext=(-45, 30),       # text position
        textcoords="offset points",
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            lw=1,
        ),
    )
    #ax[0].set_ylabel(r"$R_N=\sqrt{\pi N/2}\,\ln2-\mathbb{E}[H_N]$", math_fontfamily="cm")
    ax[0].set_ylim(-15, 32)
    ax[0].set_yticks([0, 20])
    ax[0].set_ylabel(r"Remainder $R_N$", math_fontfamily="cm")
    ax[0].tick_params(axis="x", which="both", labelbottom=False)

    ax[1].plot(x, mean/x, "o-", color=blue, ms=3, lw=1, label="Ensemble mean")
    ax[1].fill_between(x, low/x, high/x, color=blue, alpha=.15, label=band)
    ax[1].axhline(0.5, color="black", ls="--", lw=1)
    ax[1].set_xlim(0, x.max() * 1.03)
    ax[1].set_ylim(-0.35, 0.7)
    ax[1].set_yticks([0, 0.5])
    ax[1].set_yticklabels(["0", r"$1/2$"])
    ax[1].set_xlabel(r"$\ln N\,\ln\ln N$", math_fontfamily="cm")
    #ax[1].set_ylabel(r"$\dfrac{R_N-(\ln N)^2/8}{\ln N\,\ln\ln N}$", math_fontfamily="cm")
    ax[1].set_ylabel(r"$R_N/(\ln N\,\ln\ln N)$", math_fontfamily="cm")
    exponents = np.arange(np.ceil(np.log10(np.min(Ns))), np.floor(np.log10(np.max(Ns))) + 1, 2).astype(int)
    
    # Bottom axis: $(\ln N)^2$
    # ax[1].set_xlabel(r"$(\ln N)^2$", math_fontfamily="cm", labelpad=-1.5)
    # [y1,y2] = ax[1].get_ylim()
    # ax[1].set_ylim([y1-0.3*(y2-y1),y2])
    
    # exponents = np.arange(
    #     np.ceil(np.log10(np.min(Ns))),
    #     np.floor(np.log10(np.max(Ns))) + 1,
    #     2
    # ).astype(int)
    
    logN_ticks = exponents * np.log(10)
    x_N = logN_ticks * np.log(logN_ticks)
    
    for i in range(2):
        # Independent x-axis for N
        top = ax[i].twiny()
        top.set_xlim(ax[i].get_xlim())

        top.set_xticks(x_N)
        if i==0:
            top.set_xticklabels(["" for p in exponents])
        else:
            top.set_xticklabels([rf"$10^{{{p}}}$" for p in exponents])
        
        # Put the ticks AND labels on the bottom side
        top.xaxis.set_ticks_position("bottom")
        top.xaxis.set_label_position("bottom")
        
        # Hide the secondary axis spine
        top.spines["top"].set_visible(False)
        top.spines["bottom"].set_visible(False)
        top.spines["left"].set_visible(False)
        top.spines["right"].set_visible(False)
        
        # Move the labels upward, toward the existing x-axis
        top.tick_params(
            axis="x",
            which="major",
            bottom=True,
            top=False,
            direction="in",
            length=4,
            pad=-14,
        )
    
    top.set_xlabel(r'Network size $N$', math_fontfamily="cm", labelpad=-24)
    ax[1].legend(loc="center", bbox_to_anchor=(0.5, 1), ncol=1, frameon=False)

    # secax = ax[1].secondary_xaxis("top")
    # #secax.spines["bottom"].set_position(("outward", 38))
    # secax.set_xticks((exponents * np.log(10)) ** 2, [rf"$10^{{{p}}}$" for p in exponents])
    # secax.set_xlabel(r"Network size $N$")
    for i, a in enumerate(ax):
        a.spines[["top", "right"]].set_visible(False)
        #a.text(-0.13, 1.05, f"({chr(97+i)})", transform=a.transAxes, fontsize=10, fontweight="bold", fontfamily="serif", ha="left", va="bottom")
    f.align_ylabels(ax)
    save_figure(f, prefix, "figure_iii")
    return f, ax


def plot_additivity_error(ms, core_entropy_data, sum_loop_entropy_data,
                          additivity_error_data, prefix):
    scale = np.asarray(ms)*log(2)
    f, ax = plt.subplots(2, 1, figsize=(3.375, 4), sharex=True, layout="constrained")
    for data, label, ls, color in zip(
            [core_entropy_data, sum_loop_entropy_data],
            [r"$\mathbb{E}[H]$", r"$\mathbb{E}[\sum H_i]$"],
            ["-", "--"], ["blue", "orange"]):
        low, high = np.percentile(data, percentiles, axis=1)/scale
        ax[0].semilogx(ms, data.mean(axis=1)/scale, ls=ls, label=label, color=color)
        ax[0].fill_between(ms, low, high, color=color, alpha=.1)
    ax[0].set(ylim=(0, 1.03))
    ax[0].set_ylabel("Normalized entropy\n" + r"(divided by $m\ln2$)", math_fontfamily="cm")

    mean = additivity_error_data.mean(axis=1)/scale
    low, high = np.percentile(additivity_error_data, percentiles, axis=1)/scale
    worst = (ms-1)/ms
    which = mean > 0
    ax[1].loglog(ms[worst > 0], worst[worst > 0], "k-", label=r"$H - \sum H_i$")
    ax[1].loglog(ms[which], mean[which], "g--", label=r"$\mathbb{E}[H] - \mathbb{E}[\sum H_i]$")

    ax[1].fill_between(ms[which], low[which], high[which], color="green", alpha=0.1)
    ax[1].set_xlabel(r"$m$, number of relevant nodes", math_fontfamily="cm")
    ax[1].set_ylabel("Normalized error\n" + r"(divided by $m\ln2$)", math_fontfamily="cm")
    for i, a in enumerate(ax):
        a.spines[["top", "right"]].set_visible(False)
        legend = a.legend(loc="lower center", bbox_to_anchor=(0.5, 0.99 if i == 0 else 1.02), ncol=2, frameon=False, fontsize=10)
        for text in legend.get_texts():
            text.set_math_fontfamily("cm")
        #a.text(-0.18, 1.05, f"({chr(97+i)})", transform=a.transAxes, fontsize=10, fontweight="bold", fontfamily="serif", ha="left", va="bottom")
    save_figure(f, prefix, "additivity_error")
    return f, ax

# def plot_additivity_next_order(ms, core_entropy_data, sum_loop_entropy_data, prefix):
#     valid = ms >= 4
#     msv = ms[valid]
#     logm = np.log(msv)
#     x = logm * np.log(logm)

#     true_samples = (core_entropy_data[valid] - msv[:, None] * np.log(2) + 0.5 * logm[:, None]**2) / x[:, None]
#     add_samples = (sum_loop_entropy_data[valid] - msv[:, None] * np.log(2) + 0.5 * logm[:, None]**2) / x[:, None]

#     true_mean = true_samples.mean(axis=1)
#     true_low, true_high = np.percentile(true_samples, percentiles, axis=1)
#     add_mean = add_samples.mean(axis=1)
#     add_low, add_high = np.percentile(add_samples, percentiles, axis=1)

#     true_fill = "#BFD7EA"
#     add_fill = "#F6D3B8"

#     f, ax = plt.subplots(1, 1, figsize=(3.375, 2.6))

#     ax.plot(x, true_mean, "-", color=blue, lw=1.5, label=r"$\mathbb{E}[H]$")
#     ax.fill_between(x, true_low, true_high, color=true_fill, alpha=.45)

#     ax.plot(x, add_mean, "-", color=orange, lw=1.5, label=r"$\mathbb{E}[\sum_i H_i]$")
#     ax.fill_between(x, add_low, add_high, color=add_fill, alpha=.5)

#     ax.axhline(1, color=blue, ls="--", lw=1)
#     ax.axhline(0, color=orange, ls="--", lw=1)

#     ax.set_xlim(0, x.max() * 1.03)
#     ax.set_ylim(-0.18, 1.18)
#     ax.set_yticks([0, 1])
#     ax.set_xlabel(r"$\ln m\,\ln\ln m$", math_fontfamily="cm")
#     ax.set_ylabel("Normalized\nnext-order correction", math_fontfamily="cm")

#     exponents = np.arange(np.ceil(np.log10(np.min(msv))), np.floor(np.log10(np.max(msv))) + 1).astype(int)
#     logm_ticks = exponents * np.log(10)
#     x_m = logm_ticks * np.log(logm_ticks)

#     top = ax.twiny()
#     top.set_xlim(ax.get_xlim())
#     top.set_xticks(x_m)
#     top.set_xticklabels([rf"$10^{{{p}}}$" for p in exponents])
#     top.xaxis.set_ticks_position("bottom")
#     top.xaxis.set_label_position("bottom")
#     top.spines["top"].set_visible(False)
#     top.spines["bottom"].set_visible(False)
#     top.spines["left"].set_visible(False)
#     top.spines["right"].set_visible(False)
#     top.tick_params(axis="x", which="major", bottom=True, top=False, direction="in", length=4, pad=-14)
#     top.set_xlabel(r"$m$", math_fontfamily="cm", labelpad=-30)

#     legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
#     for text in legend.get_texts():
#         text.set_math_fontfamily("cm")
#     ax.spines[["top", "right"]].set_visible(False)

#     save_figure(f, prefix, "figure_additivity_next_order")
#     return f, ax

# Simulation parameters
log10_Nmin, log10_Nmax, n_Ns = 2, 8, 13
log10_mmin, log10_mmax, n_ms = 0, 4, 50
nsim, nsim_ms = 1000, 3000
m_fixed, nsim_fixed = 128, 1000
seed = 1989
use_efficient_N_sampler = True
percentiles = (2.5, 97.5)
blue, orange = "#216b86", "#b65132"


# Run simulations
rng = np.random.default_rng(seed)
Ns = np.logspace(log10_Nmin, log10_Nmax, n_Ns).astype(np.int64)
ms = np.unique(np.logspace(log10_mmin, log10_mmax, n_ms).astype(int))
output_dir = Path.cwd()
prefix = str(output_dir / (f"basin_entropy_N{Ns[0]}_{Ns[-1]}_Ns{len(Ns)}_nsim{nsim}"
                   f"_m{m_fixed}_ncore{nsim_fixed}_nms{len(ms)}_nsimms{nsim_ms}"))
sampler = (random_max_attractor_length_given_N_efficient if use_efficient_N_sampler
           else random_max_attractor_length_given_N)
mean_core = (mean_number_of_relevant_nodes_efficient if use_efficient_N_sampler
             else mean_number_of_relevant_nodes)

remainder_samples = np.empty((len(Ns), nsim))
for i, N in enumerate(Ns):
    N = int(N)
    logN = log(N)
    shift = (mean_core(N) - sqrt(pi*N/2))*log(2) + logN**2/8
    networks = sampler(N, nsim, rng=rng, return_loop_info=True)
    for j, (Amax, lengths, multipliers) in enumerate(networks):
        stats = basin_entropy(lengths, multipliers, Amax)
        remainder_samples[i, j] = shift - stats["mean_log_A"]

order_data = []
for _ in range(nsim_fixed):
    Amax, lengths, multipliers = random_max_attractor_length(
        m_fixed, rng=rng, return_loop_info=True)
    order_data.append(basin_entropy(lengths, multipliers, Amax))

# Prime-loop construction, stopping after its total size covers the plot.
primes, total, p = [], 0, 2
while total <= max(ms):
    if prime_factors(p) == {p: 1}:
        primes.append(p)
        total += p
    p += 1
sharp_data = [basin_entropy([1], [1])]
sharp_data += [basin_entropy(primes[:j], [1]*j) for j in range(1, len(primes)+1)]
C_star = fsum(log(p)/2.0**(p-1) for p in range(2, 200)
              if prime_factors(p) == {p: 1})

shape = (len(ms), nsim_ms)
delta_data, core_entropy_data, sum_loop_entropy_data, error_data = [
    np.empty(shape) for _ in range(4)]
for i, m in enumerate(ms):
    for j in range(nsim_ms):
        Amax, lengths, multipliers = random_max_attractor_length(
            int(m), rng=rng, return_loop_info=True)
        stats = basin_entropy(lengths, multipliers, Amax)
        local_log = fsum(basin_entropy([n], [s])["mean_log_A"]
                        for n, s in zip(lengths, multipliers))
        delta_data[i, j] = stats["delta"]
        core_entropy_data[i, j] = stats["H"]
        sum_loop_entropy_data[i, j] = m*log(2)-local_log
        # Subtract log periods, avoiding cancellation of two O(m) entropies.
        error = local_log-stats["mean_log_A"]
        error_data[i, j] = 0.0 if abs(error) < 1e-12 else error

# Plotting
#f1, ax1 = plot_figure_i(ms, delta_data, sharp_data, C_star, prefix)
#f2, ax2 = plot_figure_ii(order_data, m_fixed, C_star, prefix)
#f3, ax3 = plot_figure_iii(Ns, remainder_samples, prefix)
f4, ax4 = plot_additivity_error(ms, core_entropy_data, sum_loop_entropy_data, error_data, prefix)
#f5, ax5 = plot_additivity_next_order(ms, core_entropy_data, sum_loop_entropy_data, prefix)