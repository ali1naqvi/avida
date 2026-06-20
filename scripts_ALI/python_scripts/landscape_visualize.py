import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


GRID_SIZE       = 50    # Grid dimensions (N x N), no hard boundaries (toroidal)
HEIGHT          = 80    # Peak concentration value
SPREAD          = 35    # Distance from peak at which value reaches zero
PLATEAU         = -1    # Flat plateau radius around peak (-1 = no plateau)
DECAY           = 1.0   # Decay multiplier (scales the falloff rate)
MOVE_A_SCALER   = 1.0   # Movement scalar (for agent-based use, kept for reference)
PEAK_X          = 25    # X-coordinate of the concentration peak
PEAK_Y          = 25    # Y-coordinate of the concentration peak
MAX_X           = 25    # Right edge of plateau region (unused when PLATEAU == -1)
MIN_X           = 25    # Left edge of plateau region (unused when PLATEAU == -1)
MAX_Y           = 25    # Top edge of plateau region (unused when PLATEAU == -1)
MIN_Y           = 1    # Bottom edge of plateau region (unused when PLATEAU == -1)
PLATEAU_INFLOW  = 0.0   # Inflow rate inside plateau region

# ==============================================================
# Build the grid
# ==============================================================

xs = np.arange(GRID_SIZE)
ys = np.arange(GRID_SIZE)
X, Y = np.meshgrid(xs, ys)

# Toroidal (wrap-around) distance from each cell to the peak
dx = np.minimum(np.abs(X - PEAK_X), GRID_SIZE - np.abs(X - PEAK_X))
dy = np.minimum(np.abs(Y - PEAK_Y), GRID_SIZE - np.abs(Y - PEAK_Y))
dist = np.sqrt(dx**2 + dy**2)

# Linear decay away from peak, scaled by DECAY and clamped to [0, HEIGHT]
Z = np.clip(HEIGHT - DECAY * (HEIGHT / SPREAD) * dist, 0, HEIGHT)

# Apply plateau: if PLATEAU >= 0, cells within that radius are held at HEIGHT
if PLATEAU >= 0:
    Z = np.where(dist <= PLATEAU, HEIGHT, Z)

# Apply rectangular plateau region with PLATEAU_INFLOW if specified
if PLATEAU_INFLOW > 0:
    in_rect = (X >= MIN_X) & (X <= MAX_X) & (Y >= MIN_Y) & (Y <= MAX_Y)
    Z = np.where(in_rect, np.maximum(Z, PLATEAU_INFLOW), Z)

# ==============================================================
# Plot
# ==============================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Y, Z,
    cmap="Blues",
    edgecolor="navy",
    linewidth=0.08,
    alpha=0.92,
    rstride=1,
    cstride=1,
)

ax.set_xlabel("X", labelpad=8)
ax.set_ylabel("Y", labelpad=8)
ax.set_zlabel("Concentration", labelpad=8)
ax.set_title(
    'Distance as an Idealized Gradient\n'
    f'Peak at ({PEAK_X}, {PEAK_Y})  |  Height={HEIGHT}  |  Spread={SPREAD}',
    pad=14,
)

# Match the viewing angle of the reference figure
ax.view_init(elev=30, azim=-60)

ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_zlim(0, HEIGHT * 1.1)

plt.tight_layout()
plt.show()

plt.savefig("food_gradient.pdf", format="pdf")