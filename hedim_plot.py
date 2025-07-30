import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import hedim_prepare_data

# Fill with your info:
N, M = 10, 10
vector_size = 128
compression_range = list(range(vector_size, 0, -16)) + [8, 4, 2, 1]
# Fill with the measured average time for each compression point
average_times = np.array([38560, 33308, 29193, 25149, 18999, 14255, 10962, 5569, 2207, 1831, 433, 135])

# Start plotting
results = pd.read_csv("output/normal/results.csv")
plt.rcdefaults()
fig, ((axes1, axes2), (axes3, axes4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# ------------------------- PLOT 1 -------------------------
compression_range = [(128, 'D', "black"), (16,'o', "lightgray"), (2, 's', "dimgray")]
for c, l, b in compression_range: 
    rates_df = pd.read_csv(f"output/normal/comp_{c}_rates.csv")
    axes1.plot(rates_df.threshold.apply(lambda x: (x-rates_df.threshold.min())/ rates_df.threshold.max()), rates_df.far, label=f"FAR (n'={c})", color=b, markersize=9, markeredgecolor='white', markeredgewidth=1, markevery= 40)
for c, l, b in compression_range: 
    rates_df = pd.read_csv(f"output/normal/comp_{c}_rates.csv")
    axes1.plot(rates_df.threshold.apply(lambda x: (x-rates_df.threshold.min())/ rates_df.threshold.max()), rates_df.frr, label=f"FRR (n'={c})", color=b, linestyle="--", markersize=9, markeredgecolor='white', markeredgewidth=1, markevery= 40)
    
for c, l, b in compression_range: 
    df_scores = pd.read_csv(f"output/normal/comp_{c}_scores.csv")
    rates_df, eer, eer_thr = hedim_prepare_data.compute_tar_far_frr_eer(df_scores, (M-1)*M*N, (N-1)*M*M*N, step=20)
    eer_norm = (eer_thr-rates_df.threshold.min())/ rates_df.threshold.max()
    axes1.plot([eer_norm], [eer], marker='o', label={"EERs" if c == 2 else ""}, linestyle="", color=b, markersize=9, markeredgecolor='white', markeredgewidth=1)

axes1.legend(fontsize=9, framealpha=1)
axes1.spines.top.set_visible(False)
axes1.spines.right.set_visible(False)
axes1.spines.left.set_position(('axes',0 ))
axes1.spines.bottom.set_position(('axes',0 ))
axes1.set_yticklabels(["","0%", "20%", "40%", "60%", "80%", "100%"])
axes1.plot(1, 0, "|" , transform=axes1.transAxes, clip_on=False, color='black')
axes1.plot(0, 1, "_" , transform=axes1.transAxes, clip_on=False, color='black')
axes1.grid(linestyle='--', linewidth=0.7, alpha=0.5)
axes1.set_title("(2.a) FAR and FRR for some dimensions (n')", fontweight="bold")
axes1.set_xlabel("Maximum distance for match (threshold)")
axes1.set_ylabel("False Rejection and False Acceptance (%)")

# ------------------------- PLOT 2 -------------------------
axes2.plot(results['x'], results['e'], label='Error Increase: 2×EER (%)', marker='o', color="dimgray", markersize=9, markeredgecolor='white', markeredgewidth=1)
axes2.spines.top.set_visible(False)
axes2.spines.right.set_visible(False)
axes2.spines.left.set_position(('axes',0 ))
axes2.spines.bottom.set_position(('axes',0 ))
axes2.plot(1, 0, "|" , transform=axes2.transAxes, clip_on=False, color='black')
axes2.plot(0, 1, "_" , transform=axes2.transAxes, clip_on=False, color='black')
axes2.set_yticklabels(["","0%", "10%", "20%", "30%", "40%", "50%"])
axes2.set_xticks([1, 32, 64, 96, 128])
axes2.xaxis.set_inverted(True)
axes2.set_xlabel("Feature vector dimension (n')")
axes2.set_ylabel('EER (%)')
axes2.set_title("(2.b) EER (%) versus dimension (n')", fontweight="bold")
axes2.grid(linestyle='--', linewidth=0.7, alpha=0.5)

# ------------------------- PLOT 3 -------------------------
z = np.polynomial.polynomial.polyfit(results['x'], average_times, 1)
trendline = np.poly1d(z)(results['x'])
axes3.plot(results['x'], trendline, marker='s', linestyle='-', color='black', markeredgecolor='white', markersize=0, markeredgewidth=1, label="Time Decrease: t'/t (%)")
axes3.plot(results['x'], average_times, marker='s', color='black', markeredgecolor='white', markersize=9, markeredgewidth=1, linestyle="")
axes3.spines.top.set_visible(False)
axes3.spines.right.set_visible(False)
axes3.spines.left.set_position(('axes',0 ))
axes3.spines.bottom.set_position(('axes',0 ))
axes3.plot(1, 0, "|" , transform=axes3.transAxes, clip_on=False, color='black')
axes3.plot(0, 1, "^k" , transform=axes3.transAxes, clip_on=False, color='black')
axes3.set_xticks([128, 96, 64, 32, 1])
axes3.set_yticks([268,  5000, 10000, 15000, 20000, 25000, 30000, 35000])
axes3.xaxis.set_inverted(True)
axes3.set_xlabel("Feature Vector dimension (n')")
axes3.set_ylabel('Runtime (s)')
axes3.grid(linestyle='--', linewidth=0.7, alpha=0.5)
axes3.set_title("(2.c) Runtime (s) versus dimension (n')", fontweight="bold")

# ------------------------- PLOT 4 -------------------------
x_trans = (1- np.array(results['x'])/vector_size)*100
axes4.plot(x_trans, results['e']*2, label='Error Increase: 2×EER (%)', marker='o', color="dimgray", markersize=9, markeredgecolor='white', markeredgewidth=1)
axes4.plot(x_trans, trendline/max(average_times), marker='s', linestyle='-', color='black', markeredgecolor='white', markersize=0, markeredgewidth=1, label="Time Decrease: t'/t (%)")
axes4.plot(x_trans, average_times/max(average_times), marker='s', color='black', markeredgecolor='white', markersize=9, markeredgewidth=1, linestyle="")
axes4.set_title("(2.d) Dimensionality Trade-Off", fontweight="bold")
axes4.set_xlabel("Dimensionality Reduction: n'/n (%)")
axes4.set_yticklabels(["","0%", "20%", "40%", "60%", "80%", "100%"])
axes4.set_xticklabels(["","0%\n(n' = 128)", "20%", "40%", "60%", "80%", "100%\n(n' → 0)"])
axes4.set_ylabel('Error Increase / Runtime Reduction (%)')
axes4.grid(linestyle='--', linewidth=0.7, alpha=0.5)
axes4.spines.top.set_visible(False)
axes4.spines.right.set_visible(False)
axes4.spines.left.set_position(('axes',0 ))
axes4.spines.bottom.set_position(('axes',0 ))
axes4.plot(1, 0, "|" , transform=axes4.transAxes, clip_on=False, color='black')
axes4.plot(0, 1, "_" , transform=axes4.transAxes, clip_on=False, color='black')
axes4.legend(fontsize=9, framealpha=1)

# Save to PDF
fig.tight_layout(h_pad=2, w_pad=2)
fig.savefig("results.pdf")