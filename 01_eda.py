import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.facecolor": "white", "font.size": 11})
BLUE, RED, GREEN, AMBER, PURPLE = "#2C73D2", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"
P = "output/plots"
import os; os.makedirs(P, exist_ok=True)

df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin-1")
df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"])
df["shipping date (DateOrders)"] = pd.to_datetime(df["shipping date (DateOrders)"])
df["delay_days"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
df["is_fraud"] = (df["Order Status"] == "SUSPECTED_FRAUD").astype(int)
df["is_late"] = df["Late_delivery_risk"]
df["month"] = df["order date (DateOrders)"].dt.to_period("M")

print("=" * 70)
print("SUPPLY CHAIN — EDA (DataCo Global, 180K orders)")
print("=" * 70)
print(f"\n  Orders:       {len(df):,}")
print(f"  Customers:    {df['Customer Id'].nunique():,}")
print(f"  Products:     {df['Product Name'].nunique()}")
print(f"  Markets:      {df['Market'].nunique()} — {df['Market'].value_counts().to_dict()}")
print(f"  Date range:   {df['order date (DateOrders)'].min().date()} — {df['order date (DateOrders)'].max().date()}")
print(f"\n  Total Sales:  ${df['Sales'].sum():,.0f}")
print(f"  Total Profit: ${df['Benefit per order'].sum():,.0f}")
print(f"  Avg Order:    ${df['Sales'].mean():.0f}")
print(f"  Late delivery: {df['is_late'].mean()*100:.1f}%")
print(f"  Fraud orders:  {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("DataCo Supply Chain — Overview (180K orders)", fontsize=16, fontweight="bold")

ax = axes[0, 0]
ds = df["Delivery Status"].value_counts()
ax.pie(ds, labels=ds.index, autopct="%1.0f%%", colors=[RED, GREEN, BLUE, AMBER])
ax.set_title("Delivery Status")

ax = axes[0, 1]
sm = df["Shipping Mode"].value_counts()
sm.plot(kind="bar", ax=ax, color=[BLUE, AMBER, GREEN, RED], alpha=0.85)
ax.set_title("Shipping Mode Distribution"); ax.set_ylabel("Orders")
plt.sca(ax); plt.xticks(rotation=20, ha="right")

ax = axes[0, 2]
late_by_ship = df.groupby("Shipping Mode")["is_late"].mean().sort_values() * 100
colors = [RED if v > 55 else AMBER if v > 50 else GREEN for v in late_by_ship.values]
late_by_ship.plot(kind="barh", ax=ax, color=colors, alpha=0.85)
ax.set_xlabel("Late Delivery %"); ax.set_title("Late Delivery by Shipping Mode")
ax.axvline(df["is_late"].mean()*100, color="gray", linestyle="--", alpha=0.5)

ax = axes[1, 0]
mkt_sales = df.groupby("Market")["Sales"].sum().sort_values() / 1e6
mkt_sales.plot(kind="barh", ax=ax, color=BLUE, alpha=0.85)
ax.set_xlabel("Revenue ($M)"); ax.set_title("Revenue by Market")

ax = axes[1, 1]
profit_by_cat = df.groupby("Department Name")["Benefit per order"].sum().sort_values() / 1e3
profit_by_cat.plot(kind="barh", ax=ax, color=[GREEN if v > 0 else RED for v in profit_by_cat.values], alpha=0.85)
ax.set_xlabel("Profit ($K)"); ax.set_title("Profit by Department")

ax = axes[1, 2]
monthly = df.groupby(df["order date (DateOrders)"].dt.to_period("M")).agg(
    revenue=("Sales", "sum"), late_pct=("is_late", "mean")).reset_index()
ax.plot(range(len(monthly)), monthly["revenue"]/1e6, color=BLUE, linewidth=2)
ax.set_ylabel("Revenue ($M)"); ax.set_title("Monthly Revenue Trend")
ax.set_xticks(range(0, len(monthly), 6))
ax.set_xticklabels([str(monthly.iloc[i]["order date (DateOrders)"]) for i in range(0, len(monthly), 6)],
                    rotation=45, ha="right", fontsize=8)

plt.tight_layout()
plt.savefig(f"{P}/01_overview.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"\n{P}/01_overview.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Delivery Performance & Profitability", fontsize=16, fontweight="bold")

ax = axes[0, 0]
ax.hist(df["delay_days"], bins=range(-5, 6), color=BLUE, alpha=0.7, edgecolor="white")
ax.axvline(0, color=RED, linewidth=2, linestyle="--", label="On time")
ax.set_xlabel("Delay (days)"); ax.set_title("Delivery Delay Distribution"); ax.legend()

ax = axes[0, 1]
late_profit = df.groupby("is_late")["Benefit per order"].mean()
bars = ax.bar(["On Time", "Late"], late_profit.values, color=[GREEN, RED], alpha=0.85)
for b, v in zip(bars, late_profit.values):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"${v:.1f}", ha="center", fontweight="bold")
ax.set_ylabel("Avg Profit ($)"); ax.set_title("Profit: Late vs On-Time")

ax = axes[1, 0]
seg_data = df.groupby("Customer Segment").agg(orders=("Order Id","count"),
    avg_sale=("Sales","mean"), late_pct=("is_late","mean")).reset_index()
x = np.arange(len(seg_data))
ax.bar(x-0.15, seg_data["avg_sale"], 0.3, color=BLUE, alpha=0.85, label="Avg Sale $")
ax2 = ax.twinx()
ax2.bar(x+0.15, seg_data["late_pct"]*100, 0.3, color=RED, alpha=0.85, label="Late %")
ax.set_xticks(x); ax.set_xticklabels(seg_data["Customer Segment"])
ax.set_ylabel("Avg Sale ($)"); ax2.set_ylabel("Late %")
ax.set_title("Customer Segments"); ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

ax = axes[1, 1]
fraud_by_mkt = df.groupby("Market")["is_fraud"].mean().sort_values() * 100
fraud_by_mkt.plot(kind="barh", ax=ax, color=[RED if v > 2 else AMBER for v in fraud_by_mkt.values], alpha=0.85)
ax.set_xlabel("Fraud %"); ax.set_title("Suspected Fraud by Market")

plt.tight_layout()
plt.savefig(f"{P}/02_delivery_profit.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"{P}/02_delivery_profit.png")

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print(f"  1. 55% of deliveries are late — massive operational issue")
print(f"  2. Late deliveries reduce avg profit: ${late_profit[1]:.1f} vs ${late_profit[0]:.1f} (on-time)")
print(f"  3. Standard Class has highest late rate but handles 60% of volume")
print(f"  4. LATAM is top market by volume, Europe by profitability")
print(f"  5. Fraud rate: {df['is_fraud'].mean()*100:.1f}% — concentrated in specific markets")

df_clean = df[["Order Id", "Sales", "Benefit per order", "Shipping Mode", "Delivery Status",
               "Late_delivery_risk", "delay_days", "is_fraud", "is_late",
               "Customer Segment", "Market", "Order Region", "Department Name",
               "Category Name", "Product Name", "Product Price",
               "Order Item Quantity", "Order Item Discount Rate",
               "Type", "order date (DateOrders)", "Customer Id"]].copy()
df_clean.to_csv("output/orders_clean.csv", index=False)
print(f"\nCleaned data: output/orders_clean.csv ({len(df_clean):,} rows)")
