"""
WriteWrongDemo — a five-screen Tkinter app:

    1. Introduction — clown-nose prompt; Start recording / Plot circle demo
    2. Camera       — webcam preview with red-dot nose tracking,
                      3-2-1 countdown, then 10-second nose recording
    3. Complete     — XY plot of the recorded nose trajectory + Next/Retake
    4. Plot         — embedded 5-bar linkage animation tracing the path
                      (uses visualization.compute_joints; no servo I/O)
    5. Finish       — snapshot of the final linkage pose +
                      Plot Again / Restart / Quit

Run from the silly-bot directory:

    python write_wrong_demo.py
"""

import time
import tkinter as tk
from tkinter import font as tkfont, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from visualization import compute_joints, L_DEFAULT, visualize
from servo_control import ServoController, list_available_ports, DEFAULT_PORT


mp_holistic = mp.solutions.holistic


COUNTDOWN_SECONDS = 3
RECORD_SECONDS = 10

L_LINK = L_DEFAULT
WS_CENTER = (L_LINK / 2.0, L_LINK * 1.4)
WS_RADIUS = L_LINK * 0.36


def circle_demo_trajectory(L=L_DEFAULT):
    """Same circular path as ``visualization.py`` __main__ demo."""
    t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    cx, cy = L / 2, L * 1.4
    r = 1.8
    xs = cx + r * np.cos(t)
    ys = cy + r * np.sin(t)
    return xs, ys


def get_nose_position(face_landmarks):
    if face_landmarks:
        nose = face_landmarks.landmark[1]
        return nose.x, nose.y
    return None


def map_nose_to_linkage(xs, ys):
    """
    Map normalized nose positions (each in [0, 1]) into the 5-bar linkage's
    reachable workspace, autoscaling the captured motion to fill ~85% of the
    target radius. The selfie-flip is applied so the linkage traces the
    same shape the user sees on screen.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) == 0:
        return xs, ys

    # selfie flip on x, then center both axes on 0
    xs_c = (1.0 - xs) - 0.5
    ys_c = ys - 0.5

    half_x = max(np.max(np.abs(xs_c)), 1e-3)
    half_y = max(np.max(np.abs(ys_c)), 1e-3)

    target_r = WS_RADIUS * 0.85
    s = min(target_r / half_x, target_r / half_y)

    cx, cy = WS_CENTER
    return cx + xs_c * s, cy + ys_c * s


class WriteWrongDemo:
    BG = "#fafafa"
    FG = "#1f1f1f"

    COLOR_PRIMARY = "#1f77b4"
    COLOR_ACCENT = "#d62728"
    COLOR_SUCCESS = "#2ca02c"
    COLOR_NEUTRAL = "#888888"
    BUTTON_FG = "#333333"

    def __init__(self, root):
        self.root = root
        self.root.title("Write Wrong Demo")
        self.root.geometry("960x720")
        self.root.configure(bg=self.BG)

        self.title_font = tkfont.Font(family="Helvetica", size=28, weight="bold")
        self.body_font = tkfont.Font(family="Helvetica", size=16)
        self.button_font = tkfont.Font(family="Helvetica", size=14, weight="bold")

        self.container = tk.Frame(root, bg=self.BG)
        self.container.pack(fill="both", expand=True)

        # (t_seconds, nose_x in [0,1], nose_y in [0,1])
        self.nose_positions = []
        self._cleanup_callbacks = []
        self._final_frame_image = None

        # Optional Arduino link — the linkage IK in servo_control.py uses the
        # same L as the on-screen visualization so the simulated linkage and
        # the real one trace identical paths. The actual port is chosen on
        # the intro screen; until then the controller is a silent no-op.
        self.servo = ServoController(L=L_LINK, port=DEFAULT_PORT)
        self._port_options = {}      # combobox label -> device path
        self._port_status_label = None
        self._auto_connect_done = False

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_intro()

    # ------------------------------ helpers --------------------------------

    def _make_button(self, parent, text, command, color=None):
        color = color or self.COLOR_PRIMARY
        return tk.Button(
            parent, text=text, command=command,
            font=self.button_font,
            bg=color, fg=self.BUTTON_FG,
            activebackground=color, activeforeground=self.BUTTON_FG,
            relief="flat", bd=0,
            padx=28, pady=12, cursor="hand2",
            highlightthickness=0,
        )

    def _clear(self):
        for cb in self._cleanup_callbacks:
            try:
                cb()
            except Exception:
                pass
        self._cleanup_callbacks = []
        for child in self.container.winfo_children():
            child.destroy()

    @staticmethod
    def _fit_image(pil_img, w, h):
        iw, ih = pil_img.size
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        return pil_img.resize((nw, nh), Image.LANCZOS)

    @staticmethod
    def _draw_big_text(rgb_img, text):
        h, w, _ = rgb_img.shape
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 8.0
        thickness = 14
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(rgb_img, text, (x, y), font, scale, (0, 0, 0),
                    thickness + 6, cv2.LINE_AA)
        cv2.putText(rgb_img, text, (x, y), font, scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_small_timer(rgb_img, text):
        h, w, _ = rgb_img.shape
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.6
        thickness = 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        margin = 24
        x = w - tw - margin
        y = th + margin
        cv2.putText(rgb_img, text, (x, y), font, scale, (0, 0, 0),
                    thickness + 4, cv2.LINE_AA)
        cv2.putText(rgb_img, text, (x, y), font, scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)

    # -------------------------- Screen 1: Intro ----------------------------

    def show_intro(self):
        self._clear()
        frame = tk.Frame(self.container, bg=self.BG)
        frame.pack(fill="both", expand=True)

        center = tk.Frame(frame, bg=self.BG)
        center.place(relx=0.5, rely=0.5, anchor="center")

        nose = tk.Label(center, text="\u25CF", font=tkfont.Font(size=72),
                        fg=self.COLOR_ACCENT, bg=self.BG)
        nose.pack(pady=(0, 12))

        title = tk.Label(
            center,
            text="Put on the red clown nose\nand get ready to write!",
            font=self.title_font, bg=self.BG, fg=self.FG,
            justify="center",
        )
        title.pack(pady=(0, 24))

        port_block = tk.Frame(center, bg=self.BG)
        port_block.pack(pady=(0, 24))

        port_row = tk.Frame(port_block, bg=self.BG)
        port_row.pack()

        tk.Label(port_row, text="Servo port:", font=self.body_font,
                 bg=self.BG, fg=self.FG).pack(side="left", padx=(0, 10))

        self._port_var = tk.StringVar()
        self._port_combo = ttk.Combobox(
            port_row, textvariable=self._port_var,
            state="readonly", width=42, font=self.body_font,
        )
        self._port_combo.pack(side="left", padx=(0, 10))
        self._port_combo.bind("<<ComboboxSelected>>", self._on_port_selected)

        self._make_button(port_row, "Refresh", self._refresh_ports,
                          color=self.COLOR_NEUTRAL).pack(side="left")

        self._port_status_label = tk.Label(
            port_block, text="", font=self.body_font, bg=self.BG,
            fg=self.COLOR_NEUTRAL,
        )
        self._port_status_label.pack(pady=(10, 0))

        self._refresh_ports()

        btn_row = tk.Frame(center, bg=self.BG)
        btn_row.pack()
        self._make_button(btn_row, "Start recording", self.show_camera,
                          color=self.COLOR_ACCENT).pack(side="left", padx=8)
        self._make_button(btn_row, "Plot circle demo", self.run_circle_demo,
                          color=self.COLOR_PRIMARY).pack(side="left", padx=8)

    NO_SERVO_LABEL = "(no servo)"

    def _refresh_ports(self):
        """Rescan available serial ports and repopulate the combobox."""
        ports = list_available_ports()
        self._port_options = {}
        labels = [self.NO_SERVO_LABEL]
        for device, desc in ports:
            label = f"{device} — {desc}" if desc else device
            self._port_options[label] = device
            labels.append(label)
        self._port_combo["values"] = labels

        # Pick a sensible default selection in this priority:
        #   1. whatever is currently connected
        #   2. the controller's last-used / default port if visible
        #   3. "(no servo)"
        selected = self.NO_SERVO_LABEL
        target = self.servo.port if self.servo.is_connected else self.servo.port
        if target:
            for label, device in self._port_options.items():
                if device == target:
                    selected = label
                    break
        self._port_var.set(selected)

        # First time we see the intro, try to auto-connect to the default port
        # if it's actually present, so users with the standard wiring don't
        # have to click anything. Only run once so "Restart" doesn't reconnect.
        if (not self._auto_connect_done
                and not self.servo.is_connected
                and selected != self.NO_SERVO_LABEL):
            self._auto_connect_done = True
            self._connect_to_selected()
        else:
            self._update_port_status()

    def _on_port_selected(self, _event=None):
        self._connect_to_selected()

    def _connect_to_selected(self):
        label = self._port_var.get()
        if label == self.NO_SERVO_LABEL:
            self.servo.close()
            self._update_port_status("Servo: not connected (simulation only)",
                                     color=self.COLOR_NEUTRAL)
            return

        device = self._port_options.get(label)
        if device is None:
            return
        if self.servo.is_connected and self.servo.port == device:
            self._update_port_status(f"Servo: connected ({device})",
                                     color=self.COLOR_SUCCESS)
            return

        self.servo.close()
        self.servo.port = device
        self._update_port_status(f"Connecting to {device}…",
                                 color=self.COLOR_NEUTRAL)
        # Force the label to paint before the (blocking) 2-second open.
        self.root.update_idletasks()

        if self.servo.connect():
            self._update_port_status(f"Servo: connected ({device})",
                                     color=self.COLOR_SUCCESS)
        else:
            self._update_port_status(
                f"Could not open {device} — check the cable and try Refresh.",
                color=self.COLOR_ACCENT,
            )

    def _update_port_status(self, text=None, color=None):
        if text is None:
            if self.servo.is_connected:
                text = f"Servo: connected ({self.servo.port})"
                color = self.COLOR_SUCCESS
            else:
                text = "Servo: not connected (simulation only)"
                color = self.COLOR_NEUTRAL
        label = self._port_status_label
        if label is None:
            return
        try:
            if label.winfo_exists():
                label.config(text=text, fg=color or self.COLOR_NEUTRAL)
        except tk.TclError:
            # Widget was destroyed (user navigated to another screen).
            self._port_status_label = None

    def run_circle_demo(self):
        xs, ys = circle_demo_trajectory(L_DEFAULT)
        visualize(
            xs, ys,
            L=L_DEFAULT,
            interval=60,
            title="5-Bar Linkage — Circle Demo",
        )

    # -------------------------- Screen 2: Camera ---------------------------

    def show_camera(self):
        self._clear()
        self.nose_positions = []

        frame = tk.Frame(self.container, bg="black")
        frame.pack(fill="both", expand=True)

        video_label = tk.Label(frame, bg="black")
        video_label.pack(fill="both", expand=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            err = tk.Label(frame, text="Could not open camera.",
                           font=self.body_font, fg="white", bg="black")
            err.pack(pady=20)
            self._make_button(frame, "Back", self.show_intro,
                              color=self.COLOR_NEUTRAL).pack(pady=10)
            return

        holistic = mp_holistic.Holistic(min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)

        state = {
            "phase": "countdown",
            "phase_start": time.monotonic(),
            "alive": True,
        }

        def cleanup():
            state["alive"] = False
            try:
                cap.release()
            except Exception:
                pass
            try:
                holistic.close()
            except Exception:
                pass

        self._cleanup_callbacks.append(cleanup)

        def update():
            if not state["alive"]:
                return
            success, img = cap.read()
            if not success:
                self.root.after(15, update)
                return

            img.flags.writeable = False
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            nose_pos = get_nose_position(results.face_landmarks)

            display = cv2.flip(img, 1)
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

            if nose_pos:
                h, w, _ = display_rgb.shape
                cx_disp = int((1.0 - nose_pos[0]) * w)
                cy_disp = int(nose_pos[1] * h)
                cv2.circle(display_rgb, (cx_disp, cy_disp), 10,
                           (220, 30, 30), -1)
                cv2.circle(display_rgb, (cx_disp, cy_disp), 14,
                           (220, 30, 30), 2)

            now = time.monotonic()

            if state["phase"] == "countdown":
                elapsed = now - state["phase_start"]
                remaining = int(np.ceil(COUNTDOWN_SECONDS - elapsed))
                if remaining <= 0:
                    state["phase"] = "record"
                    state["phase_start"] = now
                    self.nose_positions = []
                else:
                    self._draw_big_text(display_rgb, str(remaining))

            if state["phase"] == "record":
                rec_elapsed = now - state["phase_start"]
                if rec_elapsed >= RECORD_SECONDS:
                    state["phase"] = "done"
                    cleanup()
                    self.show_complete()
                    return
                rec_remaining = max(1, int(np.ceil(RECORD_SECONDS - rec_elapsed)))
                self._draw_small_timer(display_rgb, str(rec_remaining))
                if nose_pos:
                    self.nose_positions.append(
                        (rec_elapsed, nose_pos[0], nose_pos[1])
                    )

            pil_img = Image.fromarray(display_rgb)
            ww = video_label.winfo_width()
            hh = video_label.winfo_height()
            if ww > 1 and hh > 1:
                pil_img = self._fit_image(pil_img, ww, hh)
            tk_img = ImageTk.PhotoImage(pil_img)
            video_label.configure(image=tk_img)
            video_label.image = tk_img

            self.root.after(15, update)

        self.root.after(60, update)

    # ------------------------- Screen 3: Complete --------------------------

    def show_complete(self):
        self._clear()
        frame = tk.Frame(self.container, bg=self.BG)
        frame.pack(fill="both", expand=True)

        title = tk.Label(frame, text="Here's what you wrote!",
                         font=self.title_font, bg=self.BG, fg=self.FG)
        title.pack(pady=(20, 8))

        plot_frame = tk.Frame(frame, bg=self.BG)
        plot_frame.pack(fill="both", expand=True, padx=40, pady=10)

        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # +y down to match camera frame
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Nose trajectory")
        ax.grid(True, alpha=0.3, linestyle="--")

        if self.nose_positions:
            xs = np.array([p[1] for p in self.nose_positions])
            ys = np.array([p[2] for p in self.nose_positions])
            xs_disp = 1.0 - xs  # selfie flip
            ax.plot(xs_disp, ys, "-", color=self.COLOR_ACCENT, lw=2.2)
            ax.scatter(xs_disp[:1], ys[:1], color="green",
                       zorder=5, label="start")
            ax.scatter(xs_disp[-1:], ys[-1:], color="black",
                       zorder=5, label="end")
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        else:
            ax.text(0.5, 0.5, "No nose detected during recording.",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=13, color="#666")

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        btn_frame = tk.Frame(frame, bg=self.BG)
        btn_frame.pack(pady=20)
        self._make_button(btn_frame, "Retake", self.show_camera,
                          color=self.COLOR_NEUTRAL).grid(row=0, column=0, padx=10)
        self._make_button(btn_frame, "Next", self.show_plot,
                          color=self.COLOR_PRIMARY).grid(row=0, column=1, padx=10)

    # --------------------------- Screen 4: Plot ----------------------------

    def show_plot(self):
        self._clear()
        frame = tk.Frame(self.container, bg=self.BG)
        frame.pack(fill="both", expand=True)

        title = tk.Label(frame, text="Linkage simulation",
                         font=self.title_font, bg=self.BG, fg=self.FG)
        title.pack(pady=(20, 8))

        if self.servo.is_connected:
            status_text = f"Servo: connected ({self.servo.port})"
            status_color = self.COLOR_SUCCESS
        else:
            status_text = "Servo: not connected (simulation only)"
            status_color = self.COLOR_NEUTRAL
        tk.Label(frame, text=status_text, font=self.body_font,
                 bg=self.BG, fg=status_color).pack(pady=(0, 8))

        plot_frame = tk.Frame(frame, bg=self.BG)
        plot_frame.pack(fill="both", expand=True, padx=40, pady=10)

        xs = np.array([p[1] for p in self.nose_positions])
        ys = np.array([p[2] for p in self.nose_positions])

        if len(xs) == 0:
            tk.Label(plot_frame, text="No data captured. Try again!",
                     font=self.body_font, bg=self.BG, fg="#555"
                     ).pack(pady=40)
            self._make_button(frame, "Back", self.show_intro,
                              color=self.COLOR_NEUTRAL).pack(pady=10)
            return

        link_x, link_y = map_nose_to_linkage(xs, ys)

        fig = Figure(figsize=(7, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title("5-Bar Linkage tracing your nose path", fontsize=12)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)

        anim = self._build_linkage_animation(
            fig, ax, link_x, link_y,
            on_complete=lambda: self._on_anim_done(fig),
        )

        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self._anim = anim
        self._anim_fig = fig
        self._anim_canvas = canvas

        def stop_anim():
            try:
                if anim is not None and anim.event_source is not None:
                    anim.event_source.stop()
            except Exception:
                pass

        self._cleanup_callbacks.append(stop_anim)

        if anim is None:
            # No reachable frames — let the user move on.
            self._make_button(frame, "Back", self.show_intro,
                              color=self.COLOR_NEUTRAL).pack(pady=10)

    def _build_linkage_animation(self, fig, ax, xs, ys, on_complete=None):
        L = L_LINK
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m, +down)")
        ax.grid(True, alpha=0.25, linestyle="--")

        frames = []
        for x, y in zip(xs, ys):
            j = compute_joints(x, y, L)
            if j is not None:
                frames.append(j)

        if not frames:
            ax.text(0.5, 0.5, "No reachable points in the workspace.",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=13, color="#666")
            return None

        all_pts = np.vstack([np.vstack(f) for f in frames])
        pad = L * 0.45
        ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
        ax.invert_yaxis()

        P1_s, P2_s = frames[0][0], frames[0][1]

        for mx in np.linspace(P1_s[0] - L * 0.15, P2_s[0] + L * 0.15, 12):
            ax.plot([mx, mx - L * 0.08], [P1_s[1], P1_s[1] - L * 0.12],
                    color="gray", lw=1, zorder=1)
        ax.plot([P1_s[0] - L * 0.2, P2_s[0] + L * 0.2],
                [P1_s[1], P1_s[1]], color="gray", lw=2, zorder=1)
        ax.plot([P1_s[0], P2_s[0]], [P1_s[1], P2_s[1]],
                color="#333", lw=6, solid_capstyle="round", zorder=3)
        ax.plot(*P1_s, "s", color="#d62728", markersize=12, zorder=5)
        ax.plot(*P2_s, "s", color="#1f77b4", markersize=12, zorder=5)

        traj_x = [f[4][0] for f in frames]
        traj_y = [f[4][1] for f in frames]
        ax.plot(traj_x, traj_y, "--", color="gray", lw=1, alpha=0.35, zorder=2)

        arm_L, = ax.plot([], [], "-o", color="#d62728", lw=3.5,
                         markersize=7, zorder=4)
        arm_R, = ax.plot([], [], "-o", color="#1f77b4", lw=3.5,
                         markersize=7, zorder=4)
        rod_L, = ax.plot([], [], "-o", color="#ff7f0e", lw=2.5,
                         markersize=7, zorder=4)
        rod_R, = ax.plot([], [], "-o", color="#2ca02c", lw=2.5,
                         markersize=7, zorder=4)
        ee, = ax.plot([], [], "o", color="black", markersize=10, zorder=6)
        trail, = ax.plot([], [], "-", color="black", lw=1.4,
                         alpha=0.75, zorder=2)

        done_flag = {"done": False}

        def init():
            for art in (arm_L, arm_R, rod_L, rod_R, ee, trail):
                art.set_data([], [])
            return arm_L, arm_R, rod_L, rod_R, ee, trail

        def update(i):
            P1, P2, P3, P4, P5 = frames[i]
            arm_L.set_data([P1[0], P3[0]], [P1[1], P3[1]])
            arm_R.set_data([P2[0], P4[0]], [P2[1], P4[1]])
            rod_L.set_data([P3[0], P5[0]], [P3[1], P5[1]])
            rod_R.set_data([P4[0], P5[0]], [P4[1], P5[1]])
            ee.set_data([P5[0]], [P5[1]])
            trail.set_data(traj_x[: i + 1], traj_y[: i + 1])
            # Drive the real linkage in lockstep with the on-screen one. The
            # frames list only contains reachable poses, so the IK in
            # send_xy is guaranteed to succeed for these (x, y).
            self.servo.send_xy(float(P5[0]), float(P5[1]))
            if i == len(frames) - 1 and not done_flag["done"]:
                done_flag["done"] = True
                if on_complete is not None:
                    self.root.after(900, on_complete)
            return arm_L, arm_R, rod_L, rod_R, ee, trail

        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=len(frames), interval=60,
            blit=False, repeat=False,
        )
        return anim

    def _on_anim_done(self, fig):
        try:
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            self._final_frame_image = Image.fromarray(buf).convert("RGB")
        except Exception:
            self._final_frame_image = None
        self.show_finish()

    # -------------------------- Screen 5: Finish ---------------------------

    def show_finish(self):
        self._clear()
        frame = tk.Frame(self.container, bg=self.BG)
        frame.pack(fill="both", expand=True)

        title = tk.Label(frame, text="All done!", font=self.title_font,
                         bg=self.BG, fg=self.FG)
        title.pack(pady=(20, 8))

        img_label = tk.Label(frame, bg=self.BG)
        img_label.pack(fill="both", expand=True, padx=40, pady=10)

        if self._final_frame_image is not None:
            def render():
                w = img_label.winfo_width()
                h = img_label.winfo_height()
                if w > 1 and h > 1:
                    pil = self._fit_image(self._final_frame_image, w, h)
                    tk_img = ImageTk.PhotoImage(pil)
                    img_label.configure(image=tk_img)
                    img_label.image = tk_img
            self.root.after(100, render)
        else:
            tk.Label(img_label, text="(no preview available)",
                     font=self.body_font, bg=self.BG, fg="#777").pack(pady=40)

        btn_frame = tk.Frame(frame, bg=self.BG)
        btn_frame.pack(pady=20)
        self._make_button(btn_frame, "Plot Again", self.show_plot,
                          color=self.COLOR_PRIMARY).grid(row=0, column=0, padx=8)
        self._make_button(btn_frame, "Restart", self.show_intro,
                          color=self.COLOR_SUCCESS).grid(row=0, column=1, padx=8)
        self._make_button(btn_frame, "Quit", self.on_close,
                          color=self.COLOR_ACCENT).grid(row=0, column=2, padx=8)

    # --------------------------------- exit --------------------------------

    def on_close(self):
        for cb in self._cleanup_callbacks:
            try:
                cb()
            except Exception:
                pass
        try:
            self.servo.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    WriteWrongDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
