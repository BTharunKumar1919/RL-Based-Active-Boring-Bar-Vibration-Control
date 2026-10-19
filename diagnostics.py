import pandas as pd
import numpy as np

df = pd.read_csv("results/controller_timeseries.csv")
sac = df[(df["controller"] == "SAC") &
         (df["condition"] == "nominal")]

current = sac["control_current_a"].to_numpy()
displacement = sac["displacement_um"].to_numpy()
time = sac["time_s"].to_numpy()

print("Maximum absolute current:", np.max(np.abs(current)), "A")
print("Maximum absolute displacement:", np.max(np.abs(displacement)), "um")
print("Samples beyond 5 A:", np.sum(np.abs(current) > 5.000001))
print("Fraction at current limit:",
      np.mean(np.abs(current) >= 4.95))

# Show the first point where displacement exceeds 10 um
idx = np.flatnonzero(np.abs(displacement) > 3)
if len(idx):
    i = idx[0]
    print("First |displacement| > 3 um:")
    print("time =", time[i], "s")
    print("displacement =", displacement[i], "um")
    print("current =", current[i], "A")
else:
    print("Displacement never exceeds 10 um.")