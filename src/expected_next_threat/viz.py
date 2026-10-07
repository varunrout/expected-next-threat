"""Pictures of the engine: one frame, one possession, and an animation of a possession.

    live = engine.predict_match(match)
    plot_frame(engine, match, live, frame=45602)
    plot_possession(match, live, possession=199)
    animate_possession(engine, match, live, possession=199, path="possession.gif")

The attacking team always plays left to right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FPS

INK = "#1f2937"; MUTED = "#9ca3af"; LINE = "#a8b0ac"; GRASS = "#f3f5f1"
DANGER = "#c2410c"; CARRY = "#0f6b5c"; BALL = "#111111"


def draw_pitch(ax, L=105.0, W=68.0):
    ax.set_facecolor(GRASS)
    kw = dict(color=LINE, lw=1, zorder=1)
    ax.plot([-L / 2, L / 2, L / 2, -L / 2, -L / 2], [-W / 2, -W / 2, W / 2, W / 2, -W / 2], **kw)
    ax.plot([0, 0], [-W / 2, W / 2], **kw)
    for sx in (-1, 1):
        ax.plot([sx * L / 2, sx * (L / 2 - 16.5), sx * (L / 2 - 16.5), sx * L / 2], [-20.16, -20.16, 20.16, 20.16], **kw)
        ax.plot([sx * L / 2, sx * (L / 2 - 5.5), sx * (L / 2 - 5.5), sx * L / 2], [-9.16, -9.16, 9.16, 9.16], **kw)
    import matplotlib.pyplot as plt
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, **kw))
    ax.set_xlim(-L / 2 - 2, L / 2 + 2); ax.set_ylim(-W / 2 - 2, W / 2 + 2); ax.set_aspect("equal"); ax.axis("off")
    return ax


def _row(live: pd.DataFrame, frame: int) -> pd.Series:
    r = live[live.frame == frame]
    if len(r) == 0:   # nearest on-ball frame
        r = live.iloc[[int(np.argmin(np.abs(live.frame.values - frame)))]]
    return r.iloc[0]


def draw_frame(ax, engine, match, row: pd.Series, show_text=True, vmax=None):
    """Players, ball, receiver probabilities (circle area, colour = xT at the reception point) and the carry."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    f, s, cs = int(row.frame), float(row.sign), int(row.carrier_slot)
    L, W = match.pitch_length, match.pitch_width
    X, Y = match.X[f] * s, match.Y[f] * s
    att = match.team == row.team_id
    ok = ~np.isnan(X)
    draw_pitch(ax, L, W)
    _, cand = engine.pass_layer(match, pd.DataFrame([row])[["frame", "carrier_slot", "team_id", "sign", "t_on_ball"]], return_candidates=True)
    vmax = vmax or max(0.05, float(cand.xt_target.max()) if len(cand) else 0.05)
    sc = ax.scatter(cand.x, cand.y, s=3500 * cand.p_receiver, c=cand.xt_target, cmap="Oranges", vmin=0, vmax=vmax,
                    edgecolor=DANGER, linewidth=1, alpha=0.7, zorder=3)
    ax.scatter(X[~att & ok], Y[~att & ok], s=55, color=INK, zorder=4, label="defenders")
    ax.scatter(X[att & ok], Y[att & ok], s=38, color=MUTED, edgecolor="white", linewidth=0.5, zorder=4, label="attackers")
    sg = np.sign(row.cy + 1e-6)
    ex, ey = row.cx + row.carry_dx, row.cy + row.carry_dy_out * sg
    ax.add_patch(Ellipse((ex, ey), 2 * row.carry_sd_dx, 2 * row.carry_sd_dy, facecolor=CARRY, alpha=0.18, edgecolor=CARRY, zorder=2))
    ax.annotate("", xy=(ex, ey), xytext=(row.cx, row.cy), arrowprops=dict(arrowstyle="->", color=CARRY, lw=1.5), zorder=5)
    ax.scatter([row.cx], [row.cy], s=110, color=CARRY, edgecolor="white", linewidth=1, zorder=6, label="carrier")
    bx, by = match.BX[f] * s, match.BY[f] * s
    if not np.isnan(bx):
        ax.scatter([bx], [by], s=18, color=BALL, zorder=7)
    if show_text:
        txt = (f"ENT {row.ent_full:.3f}   keep {row.p_keep:.0%}  pass {row.p_pass:.0%}  shot {row.p_shot:.0%}"
               + ("" if pd.isna(row.get("p_danger5", np.nan)) else f"   danger in 5 s {row.p_danger5:.0%}"))
        ax.text(-L / 2, W / 2 + 3.5, txt, fontsize=9.5, color=INK, va="bottom")
    return sc, cand


def plot_frame(engine, match, live: pd.DataFrame, frame: int | None = None, clock: str | None = None, ax=None):
    """One frame: who gets the next pass, where the carry goes, and what it is worth."""
    import matplotlib.pyplot as plt
    if frame is None:
        frame = match.frame_at(clock)
    row = _row(live, frame)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6.8))
    sc, _ = draw_frame(ax, engine, match, row)
    plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.01, label="xT where he would receive")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5, ncol=3, bbox_to_anchor=(0, -0.06))
    ax.set_title(f"{match.clock_at(int(row.frame))}  |  circle area = P(next receiver)", loc="left", fontsize=10, color=INK)
    return ax


def _with_gaps(p: pd.DataFrame, cols):
    """Insert NaN rows where nobody holds the ball, so a line plot shows gaps instead of joining them."""
    gap = p.clock.diff() > 0.25
    filler = p[gap].copy(); filler["clock"] = filler.clock - 0.1
    for c in cols:
        filler[c] = np.nan
    return pd.concat([p, filler]).sort_values("clock")


def draw_timeline(ax_top, ax_bot, poss: pd.DataFrame, cursor=None):
    p = poss.copy(); p["clock"] = p.t - p.t.iloc[0]
    q = _with_gaps(p, ["ent_full", "xt_here", "p_danger5"])
    ax_top.plot(q.clock, q.ent_full, color=DANGER, lw=2, label="ENT, next 1 s")
    ax_top.plot(q.clock, q.xt_here, color="#6b7280", lw=1.4, ls="--", label="xT at the ball")
    ax_top.set_ylabel("expected threat", fontsize=9); ax_top.legend(frameon=False, fontsize=8.5, loc="upper left")
    if ax_bot is not None and "p_danger5" in q:
        ax_bot.plot(q.clock, q.p_danger5, color=CARRY, lw=2)
        ax_bot.set_ylim(0, 1); ax_bot.set_ylabel("P(box entry or\nshot in 5 s)", fontsize=9)
        ax_bot.set_xlabel("seconds into the possession (gaps: ball in flight)", fontsize=9)
    for a in (ax_top, ax_bot):
        if a is None:
            continue
        a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=8)
        if cursor is not None:
            a.axvline(cursor, color=INK, lw=0.8, alpha=0.6)
    return p


def plot_possession(match, live: pd.DataFrame, possession: int, axes=None):
    """Danger timeline of one team possession."""
    import matplotlib.pyplot as plt
    poss = live[live.tp == possession]
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    draw_timeline(axes[0], axes[1], poss)
    axes[0].set_title(f"Possession {possession}, from {match.clock_at(int(poss.frame.iloc[0]))}", loc="left", fontsize=10)
    return axes


def plot_moment(engine, match, live: pd.DataFrame, possession: int, frame: int | None = None, title: str | None = None):
    """Frame plus timeline in one figure (README Figure 1). Defaults to the frame where ENT peaks."""
    import matplotlib.pyplot as plt
    poss = live[live.tp == possession]
    row = poss.loc[poss.ent_full.idxmax()] if frame is None else _row(poss, frame)
    fig = plt.figure(figsize=(10, 9.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[6.2, 2.0, 1.4], hspace=0.12)
    ax = fig.add_subplot(gs[0]); sc, _ = draw_frame(ax, engine, match, row)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, ncol=1, bbox_to_anchor=(0.01, 0.99))
    ax.text(-match.pitch_length / 2, -match.pitch_width / 2 - 3.5,
            "circle area: P(next receiver)   colour: xT where he would receive   arrow and cloud: expected 1 s carry",
            fontsize=8.5, color="#6b7280", va="top")
    a1 = fig.add_subplot(gs[1]); a2 = fig.add_subplot(gs[2], sharex=a1)
    p = draw_timeline(a1, a2, poss, cursor=float(row.t - poss.t.iloc[0]))
    plt.setp(a1.get_xticklabels(), visible=False)
    if title:
        fig.suptitle(title, x=0.125, ha="left", fontsize=12, y=0.93)
    return fig


def animate_possession(engine, match, live: pd.DataFrame, possession: int, path: str, step: int = 2, fps: int = 5, lead_in_s: float = 0.0):
    """Animate a possession at 0.2 s steps (real time at fps=5). Writes .gif (Pillow) or .mp4 (needs ffmpeg)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    poss = live[live.tp == possession]
    f0, f1 = int(poss.frame.min() - lead_in_s * FPS), int(poss.frame.max())
    frames = np.arange(f0, f1 + 1, step)
    vmax = max(0.05, float(poss.ent_pass.max()) * 1.5)
    fig = plt.figure(figsize=(9, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[6, 2.2], hspace=0.08)
    ax = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1])
    on = poss.set_index("frame")
    last = {"row": poss.iloc[0]}

    def draw(f):
        ax.clear(); a1.clear()
        if f in on.index:
            last["row"] = on.loc[f].copy(); last["row"]["frame"] = f
            draw_frame(ax, engine, match, last["row"], vmax=vmax)
        else:   # ball in flight: players and ball only
            r = last["row"]; s = float(r.sign); X, Y = match.X[f] * s, match.Y[f] * s
            att = match.team == r.team_id; ok = ~np.isnan(X)
            draw_pitch(ax, match.pitch_length, match.pitch_width)
            ax.scatter(X[~att & ok], Y[~att & ok], s=55, color=INK, zorder=4)
            ax.scatter(X[att & ok], Y[att & ok], s=38, color=MUTED, edgecolor="white", linewidth=0.5, zorder=4)
            if not np.isnan(match.BX[f]):
                ax.scatter([match.BX[f] * s], [match.BY[f] * s], s=18, color=BALL, zorder=7)
            ax.text(-match.pitch_length / 2, match.pitch_width / 2 + 3.5, "ball in flight", fontsize=9.5, color=MUTED, va="bottom")
        draw_timeline(a1, None, poss, cursor=(f - poss.frame.iloc[0]) / FPS)
        a1.set_xlabel("seconds into the possession", fontsize=9)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2400) if str(path).endswith(".mp4") else PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=110)
    plt.close(fig)
    return path
