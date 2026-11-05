import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from itertools import combinations
import warnings; warnings.filterwarnings("ignore")
BLUE, RED, GREEN, AMBER, PURPLE = "#2C73D2", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"
P = "output/plots"
import os; os.makedirs(P, exist_ok=True)

df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin-1")
df["is_late"] = df["Late_delivery_risk"]
df["delay"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
df["is_fraud"] = (df["Order Status"] == "SUSPECTED_FRAUD").astype(int)

print("=" * 70)
print("A/B TEST: SHIPPING MODE IMPACT ON DELIVERY PERFORMANCE")
print("=" * 70)

modes = ["Standard Class", "First Class", "Second Class", "Same Day"]
print(f"\n{'='*25} 1. SHIPPING MODE OVERVIEW {'='*25}")

for m in modes:
    sub = df[df["Shipping Mode"] == m]
    print(f"\n  {m}:")
    print(f"    Orders:     {len(sub):,} ({len(sub)/len(df)*100:.0f}%)")
    print(f"    Late %:     {sub['is_late'].mean()*100:.1f}%")
    print(f"    Avg delay:  {sub['delay'].mean():+.2f} days")
    print(f"    Avg profit: ${sub['Benefit per order'].mean():.2f}")
    print(f"    Fraud %:    {sub['is_fraud'].mean()*100:.2f}%")

print(f"\n{'='*25} 2. STANDARD vs FIRST CLASS {'='*25}")

std = df[df["Shipping Mode"] == "Standard Class"]
first = df[df["Shipping Mode"] == "First Class"]

late_std = std["is_late"].mean()
late_first = first["is_late"].mean()
diff_late = late_first - late_std
n_std, n_first = len(std), len(first)

p_pool = df[df["Shipping Mode"].isin(["Standard Class", "First Class"])]["is_late"].mean()
se = np.sqrt(p_pool*(1-p_pool)*(1/n_std + 1/n_first))
z = diff_late / se
p_z = stats.norm.sf(abs(z)) * 2
ci_lo = diff_late - 1.96*se
ci_hi = diff_late + 1.96*se
h = 2*(np.arcsin(np.sqrt(late_first)) - np.arcsin(np.sqrt(late_std)))

print(f"\n  Late Delivery Rate:")
print(f"    Standard: {late_std*100:.2f}% (n={n_std:,})")
print(f"    First:    {late_first*100:.2f}% (n={n_first:,})")
print(f"    Diff:     {diff_late*100:+.2f} p.p.")
print(f"    Z={z:.4f}, p={p_z:.6f}")
print(f"    95% CI:   [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]")
print(f"    Cohen's h: {h:.4f}")

t_profit, p_profit = stats.ttest_ind(first["Benefit per order"], std["Benefit per order"], equal_var=False)
d_profit = (first["Benefit per order"].mean() - std["Benefit per order"].mean()) / \
           np.sqrt((first["Benefit per order"].std()**2 + std["Benefit per order"].std()**2)/2)

print(f"\n  Profit per Order:")
print(f"    Standard: ${std['Benefit per order'].mean():.2f}")
print(f"    First:    ${first['Benefit per order'].mean():.2f}")
print(f"    t={t_profit:.4f}, p={p_profit:.6f}")
print(f"    Cohen's d: {d_profit:.4f}")

t_delay, p_delay = stats.ttest_ind(first["delay"], std["delay"], equal_var=False)
print(f"\n  Delay (days):")
print(f"    Standard: {std['delay'].mean():+.2f} days")
print(f"    First:    {first['delay'].mean():+.2f} days")
print(f"    t={t_delay:.4f}, p={p_delay:.6f}")

print(f"\n{'='*25} 3. PAIRWISE COMPARISONS (Late %) {'='*25}")
alpha_bonf = 0.05 / 6

pw_results = []
for m1, m2 in combinations(modes, 2):
    g1 = df[df["Shipping Mode"]==m1]["is_late"]
    g2 = df[df["Shipping Mode"]==m2]["is_late"]
    diff = g1.mean() - g2.mean()
    pp = (g1.sum()+g2.sum()) / (len(g1)+len(g2))
    se_p = np.sqrt(pp*(1-pp)*(1/len(g1)+1/len(g2)))
    z_p = diff / max(se_p, 1e-10)
    p_p = stats.norm.sf(abs(z_p)) * 2
    sig = "Yes" if p_p < alpha_bonf else "No"
    pw_results.append({"Pair": f"{m1} vs {m2}", "Diff": f"{diff*100:+.1f}%",
                       "p-value": round(p_p, 6), "Sig (Bonf)": sig})
    print(f"  {m1:16s} vs {m2:16s}: {diff*100:+.2f} p.p., p={p_p:.6f} {sig}")

groups = [df[df["Shipping Mode"]==m]["is_late"] for m in modes]
f_stat, p_anova = stats.f_oneway(*groups)
kw_stat, p_kw = stats.kruskal(*groups)
print(f"\n  ANOVA: F={f_stat:.2f}, p={p_anova:.6f}")
print(f"  Kruskal-Wallis: H={kw_stat:.2f}, p={p_kw:.6f}")

print(f"\n{'='*25} 4. BOOTSTRAP (Standard vs First, 10K) {'='*25}")
n_boot = 10000
std_vals = std["is_late"].values
first_vals = first["is_late"].values

boot_diffs = []
for _ in range(n_boot):
    b_s = np.random.choice(std_vals, n_std, replace=True).mean()
    b_f = np.random.choice(first_vals, n_first, replace=True).mean()
    boot_diffs.append(b_f - b_s)
boot_diffs = np.array(boot_diffs)
bci = np.percentile(boot_diffs, [2.5, 97.5])
print(f"  Mean diff: {np.mean(boot_diffs)*100:+.2f} p.p.")
print(f"  95% CI: [{bci[0]*100:.2f}%, {bci[1]*100:.2f}%]")
print(f"  Contains 0: {'Yes' if bci[0]<=0<=bci[1] else 'No'}")

print(f"\n{'='*25} 5. SUBGROUP: Standard vs First BY MARKET {'='*25}")
for mkt in df["Market"].unique():
    s = df[(df["Shipping Mode"]=="Standard Class") & (df["Market"]==mkt)]["is_late"]
    f = df[(df["Shipping Mode"]=="First Class") & (df["Market"]==mkt)]["is_late"]
    if len(s) > 50 and len(f) > 50:
        diff = f.mean() - s.mean()
        _, p = stats.ttest_ind(s, f)
        print(f"  {mkt:15s}: Std={s.mean()*100:.1f}% | First={f.mean()*100:.1f}% | "
              f"delta={diff*100:+.1f} p.p. | p={p:.4f}")

print(f"\n{'='*25} 6. FRAUD BY PAYMENT TYPE {'='*25}")
for ptype in df["Type"].unique():
    sub = df[df["Type"]==ptype]
    print(f"  {ptype:10s}: Fraud={sub['is_fraud'].mean()*100:.2f}% (n={len(sub):,})")

ct = pd.crosstab(df["Type"], df["is_fraud"])
chi2_f, p_f, _, _ = stats.chi2_contingency(ct)
print(f"\n  Chi2: {chi2_f:.2f}, p={p_f:.6f} {'Significant' if p_f<0.05 else 'Not significant'}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Shipping Mode Comparison: Late Deliveries, Profit, Fraud", fontsize=16, fontweight="bold")

ax = axes[0, 0]
late_modes = df.groupby("Shipping Mode")["is_late"].mean().sort_values()*100
colors = [RED if v > 55 else AMBER if v > 50 else GREEN for v in late_modes.values]
late_modes.plot(kind="barh", ax=ax, color=colors, alpha=0.85)
ax.set_xlabel("Late Delivery %"); ax.set_title("Late Delivery by Shipping Mode")
ax.axvline(df["is_late"].mean()*100, color="gray", linestyle="--", alpha=0.5)

ax = axes[0, 1]
ax.hist(boot_diffs*100, bins=50, color=BLUE, alpha=0.7, edgecolor="white")
ax.axvline(0, color=RED, linewidth=2, linestyle="--", label="No diff")
ax.axvline(diff_late*100, color=GREEN, linewidth=2, label=f"Observed: {diff_late*100:+.1f}%")
ax.axvline(bci[0]*100, color="orange", linestyle=":"); ax.axvline(bci[1]*100, color="orange", linestyle=":")
ax.set_xlabel("Late % Diff (First - Standard)"); ax.set_title("Bootstrap: Standard vs First"); ax.legend(fontsize=9)

ax = axes[0, 2]
profit_modes = df.groupby("Shipping Mode")["Benefit per order"].mean().sort_values()
profit_modes.plot(kind="barh", ax=ax, color=[GREEN if v>0 else RED for v in profit_modes.values], alpha=0.85)
ax.set_xlabel("Avg Profit ($)"); ax.set_title("Profit by Shipping Mode")

ax = axes[1, 0]
hm = df[df["Shipping Mode"].isin(["Standard Class","First Class"])].groupby(
    ["Market","Shipping Mode"])["is_late"].mean().unstack()*100
sns.heatmap(hm, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("Late % by Market x Mode")

ax = axes[1, 1]
fraud_pay = df.groupby("Type")["is_fraud"].mean().sort_values()*100
fraud_pay.plot(kind="bar", ax=ax, color=[RED if v>2 else AMBER for v in fraud_pay.values], alpha=0.85)
ax.set_ylabel("Fraud %"); ax.set_title("Fraud by Payment Type")
plt.sca(ax); plt.xticks(rotation=0)

ax = axes[1, 2]
for mode, color in [("Standard Class", BLUE), ("First Class", GREEN), ("Same Day", AMBER)]:
    sub = df[df["Shipping Mode"]==mode]["delay"]
    ax.hist(sub, bins=range(-5,6), alpha=0.4, color=color, label=mode, density=True)
ax.set_xlabel("Delay (days)"); ax.set_title("Delay Distribution"); ax.legend(fontsize=9)
ax.axvline(0, color=RED, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"{P}/03_ab_test.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"\n{P}/03_ab_test.png")

print(f"\n{'='*70}")
print("BUSINESS RECOMMENDATIONS")
print("=" * 70)
sla = df.groupby("Shipping Mode").agg(
    scheduled=("Days for shipment (scheduled)", "mean"),
    real=("Days for shipping (real)", "mean"),
    late=("is_late", "mean")).loc[modes]
best_mode = sla["late"].idxmin()
fraud_by_type = df.groupby("Type")["is_fraud"].mean()
fraud_by_market = df.groupby("Market")["is_fraud"].mean()

print(f"\n  NOTE: this is an observational comparison, not a randomized A/B test —")
print(f"  customers chose the shipping mode themselves.")

print(f"\n  KEY FINDING: 'late' means the order arrived after the PROMISED date.")
print(f"  Express modes miss their promise; Standard keeps it:")
for m, r in sla.iterrows():
    print(f"     {m:16s} promised {r['scheduled']:.1f} d, delivered {r['real']:.2f} d, late {r['late']*100:.1f}%")

print(f"\n  RECOMMENDATIONS:")
print(f"     1. Do NOT migrate orders to First Class: its late rate is "
      f"{diff_late*100:+.1f} p.p. vs Standard ({late_first*100:.1f}% vs {late_std*100:.1f}%).")
print(f"     2. Fix the delivery promise for express modes: First Class promises "
      f"{sla.loc['First Class','scheduled']:.0f} d but takes {sla.loc['First Class','real']:.0f} d, "
      f"Second Class promises {sla.loc['Second Class','scheduled']:.0f} d but takes "
      f"{sla.loc['Second Class','real']:.0f} d. Either show realistic dates or fix operations.")
print(f"     3. Lowest late rate: {best_mode} ({sla.loc[best_mode,'late']*100:.1f}%).")
print(f"     4. Fraud is concentrated in {fraud_by_type.idxmax()} payments "
      f"({fraud_by_type.max()*100:.2f}% vs {fraud_by_type.drop(fraud_by_type.idxmax()).max()*100:.2f}% elsewhere) — add verification there.")
print(f"     5. Fraud by market is flat ({fraud_by_market.min()*100:.2f}–{fraud_by_market.max()*100:.2f}%): "
      f"no single market needs special monitoring.")

pd.DataFrame(pw_results).to_csv("output/pairwise_shipping.csv", index=False)
print(f"\noutput/pairwise_shipping.csv")
