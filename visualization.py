import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Default link length — all five bars equal length in this design
L_DEFAULT = 5.0


def _cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _segments_cross(a1, a2, b1, b2):
    """True if open segments a1-a2 and b1-b2 strictly intersect."""
    r, s = a2 - a1, b2 - b1
    denom = _cross2d(r, s)
    if abs(denom) < 1e-12:
        return False
    t = _cross2d(b1 - a1, s) / denom
    u = _cross2d(b1 - a1, r) / denom
    return 0 < t < 1 and 0 < u < 1


def _circle_intersect(c1, r1, c2, r2):
    """
    Intersection of two circles. Returns (pt_a, pt_b) or None if they don't
    intersect. pt_a has the smaller y-value (higher on screen, y-down frame).
    """
    d = np.linalg.norm(c2 - c1)
    if d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9 or d < 1e-12:
        return None
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(r1 ** 2 - a ** 2, 0.0))
    mid = c1 + a * (c2 - c1) / d
    perp = h * np.array([-(c2[1] - c1[1]), c2[0] - c1[0]]) / d
    p0, p1 = mid + perp, mid - perp
    # sort: smaller y first
    return (p0, p1) if p0[1] <= p1[1] else (p1, p0)


def compute_joints(x, y, L):
    """
    Forward geometry: given end-effector position (x, y) return the five joint
    positions (P1, P2, P3, P4, P5) or None if the position is unreachable.

    Coordinate system: +x right, +y down.
    P1 = (0, 0)   — left motor anchor
    P2 = (L, 0)   — right motor anchor
    P3             — left elbow  (on circle P1 and circle P5, radius L each)
    P4             — right elbow (on circle P2 and circle P5, radius L each)
    P5 = (x, y)   — end effector

    prev: previous frame's joint tuple — when supplied, the branch that keeps
          the elbows closest to their last positions is chosen, giving
          physically continuous motion. On the first frame (prev=None) the
          highest-elbows non-crossing branch is used as the starting pose.
    """
    P1 = np.array([0.0, 0.0])
    P2 = np.array([L, 0.0])
    P5 = np.array([x, y])

    pts3 = _circle_intersect(P1, L, P5, L)
    if pts3 is None:
        return None

    pts4 = _circle_intersect(P2, L, P5, L)
    if pts4 is None:
        return None

    # Left arm always points left  → pick the intersection with the smaller x.
    # Right arm always points right → pick the intersection with the larger x.
    # This is the only physically valid branch for a downward-hanging 5-bar
    # linkage, and it is continuous throughout the reachable workspace because
    # the two circle-intersection points never swap their x-ordering.
    P3 = pts3[0] if pts3[0][0] <= pts3[1][0] else pts3[1]
    P4 = pts4[0] if pts4[0][0] >= pts4[1][0] else pts4[1]

    if _segments_cross(P1, P3, P2, P4):
        return None

    return P1, P2, P3, P4, P5


def _motor_angles(P1, P2, P3, P4, L):
    """Return (theta1_deg, theta2_deg) measured from +y toward the link."""
    theta1 = np.degrees(np.arctan2(-P3[0] + P1[0], P3[1] - P1[1]))
    theta2 = np.degrees(np.arctan2(P4[0] - P2[0], P4[1] - P2[1]))
    return theta1, theta2


def visualize(xs, ys, L=L_DEFAULT, interval=80, title="5-Bar Linkage"):
    """
    Animate the 5-bar linkage moving through the given end-effector trajectory.

    Parameters
    ----------
    xs, ys   : array-like, end-effector X and Y positions (+y down)
    L        : link length (all links assumed equal)
    interval : ms between animation frames
    title    : window/plot title
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    print("Computing joint positions...")
    frames = []
    for x, y in zip(xs, ys):
        j = compute_joints(x, y, L)
        if j is not None:
            frames.append(j)
        else:
            print(f"  ({x:.3f}, {y:.3f}) unreachable — skipped")

    if not frames:
        print("No reachable positions; nothing to animate.")
        return None

    # ── figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X  (m)")
    ax.set_ylabel("Y  (m, positive ↓)")
    ax.grid(True, alpha=0.25, linestyle="--")

    all_pts = np.vstack([np.vstack(f) for f in frames])
    pad = L * 0.45
    xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()   # +y downward on screen

    # ── static elements ───────────────────────────────────────────────────────
    P1_s, P2_s = frames[0][0], frames[0][1]

    # hatching to show fixed wall
    for mx in np.linspace(P1_s[0] - L * 0.15, P2_s[0] + L * 0.15, 12):
        ax.plot([mx, mx - L * 0.08], [P1_s[1], P1_s[1] - L * 0.12],
                color="gray", lw=1, zorder=1)
    ax.plot([P1_s[0] - L * 0.2, P2_s[0] + L * 0.2], [P1_s[1], P1_s[1]],
            color="gray", lw=2, zorder=1)

    ax.plot([P1_s[0], P2_s[0]], [P1_s[1], P2_s[1]],
            color="#333333", lw=6, solid_capstyle="round", zorder=3,
            label="Fixed bar")

    motor1 = ax.plot(*P1_s, "s", color="#d62728", markersize=13, zorder=5,
                     label="Motor 1 (P1)")[0]
    motor2 = ax.plot(*P2_s, "s", color="#1f77b4", markersize=13, zorder=5,
                     label="Motor 2 (P2)")[0]

    # ghost trajectory
    traj_x = [f[4][0] for f in frames]
    traj_y = [f[4][1] for f in frames]
    ax.plot(traj_x, traj_y, "--", color="gray", lw=1, alpha=0.35, zorder=2)

    # ── animated link artists ─────────────────────────────────────────────────
    arm_L, = ax.plot([], [], "-o", color="#d62728", lw=3.5, markersize=7,
                     solid_capstyle="round", zorder=4, label="Left arm (L)")
    arm_R, = ax.plot([], [], "-o", color="#1f77b4", lw=3.5, markersize=7,
                     solid_capstyle="round", zorder=4, label="Right arm (R)")
    rod_L, = ax.plot([], [], "-o", color="#ff7f0e", lw=2.5, markersize=7,
                     solid_capstyle="round", zorder=4, label="Left rod")
    rod_R, = ax.plot([], [], "-o", color="#2ca02c", lw=2.5, markersize=7,
                     solid_capstyle="round", zorder=4, label="Right rod")
    ee_dot, = ax.plot([], [], "o", color="black", markersize=11, zorder=6,
                      label="End effector")
    trail, = ax.plot([], [], "-", color="black", lw=1, alpha=0.55, zorder=2)

    info = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                   va="top", ha="left", fontsize=9,
                   fontfamily="monospace",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    # ── animation callbacks ───────────────────────────────────────────────────
    def init():
        for artist in (arm_L, arm_R, rod_L, rod_R, ee_dot, trail):
            artist.set_data([], [])
        info.set_text("")
        return arm_L, arm_R, rod_L, rod_R, ee_dot, trail, info

    def update(i):
        P1, P2, P3, P4, P5 = frames[i]

        arm_L.set_data([P1[0], P3[0]], [P1[1], P3[1]])
        arm_R.set_data([P2[0], P4[0]], [P2[1], P4[1]])
        rod_L.set_data([P3[0], P5[0]], [P3[1], P5[1]])
        rod_R.set_data([P4[0], P5[0]], [P4[1], P5[1]])
        ee_dot.set_data([P5[0]], [P5[1]])
        trail.set_data(traj_x[: i + 1], traj_y[: i + 1])

        t1, t2 = _motor_angles(P1, P2, P3, P4, L)
        info.set_text(
            f"Frame {i+1}/{len(frames)}\n"
            f"θ₁ = {t1:+.1f}°\n"
            f"θ₂ = {t2:+.1f}°\n"
            f"X  = {P5[0]:.3f}\n"
            f"Y  = {P5[1]:.3f}"
        )
        return arm_L, arm_R, rod_L, rod_R, ee_dot, trail, info

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(frames), interval=interval,
        blit=False, repeat=True
    )

    plt.tight_layout()
    plt.show()
    return anim


# ── demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    L = L_DEFAULT          # 5.0

    # Circular trajectory centred in the reachable workspace
    t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    cx, cy = L / 2, L * 1.4     # centre: (2.5, 7.0)
    r = 1.8
    xs = cx + r * np.cos(t)
    ys = cy + r * np.sin(t)

    visualize(xs, ys, L=L, interval=60, title="5-Bar Linkage — Circle Demo")
