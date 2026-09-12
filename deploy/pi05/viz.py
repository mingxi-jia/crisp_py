"""Local OpenCV window showing what π0.5 is being sent (--visualize window).

The panel (deploy/pi05/panel.py) is the richer view; this stays for a quick
look without a browser.
"""

import numpy as np

_SCALE = 2              # 224 -> 448 so the images are actually legible
_BG = (24, 22, 20)
_FG = (235, 233, 229)
_DIM = (150, 147, 142)
_ACCENT = (247, 195, 79)
_WARN = (80, 80, 240)


class Pi05Visualizer:
    """OpenCV window. Call update() each step and close() when done."""

    WINDOW = "pi0.5-DROID inputs"

    def __init__(self, external_camera: str, wrist_camera: str):
        import cv2

        self.cv2 = cv2
        self.external_camera = external_camera
        self.wrist_camera = wrist_camera
        self.available = False
        # Check for a display BEFORE calling into OpenCV. With no X11/Wayland,
        # cv2's Qt backend calls abort() rather than raising, so a try/except
        # around namedWindow does not save the process -- the whole control
        # loop would die instead of just losing the window.
        import os
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("[visualize] no DISPLAY/WAYLAND_DISPLAY; disabling the window "
                  "(the control loop continues). Over SSH, use ssh -X.")
            return
        # DISPLAY being *set* is not proof it is reachable (stale value, ssh
        # without xauth). Probe in a throwaway subprocess -- if Qt aborts, it
        # takes that process down instead of this one. Costs ~1 s, once.
        if not self._probe_display():
            print("[visualize] DISPLAY is set but no window could be opened; "
                  "disabling (the control loop continues)")
            return
        try:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
            self.available = True
        except Exception as e:
            print(f"[visualize] cannot open a window ({e}); disabling")

    @staticmethod
    def _probe_display(timeout: float = 15.0) -> bool:
        """True if a GUI window can actually be created here."""
        import subprocess
        import sys
        try:
            r = subprocess.run(
                [sys.executable, "-c",
                 "import cv2; cv2.namedWindow('p', cv2.WINDOW_AUTOSIZE); "
                 "cv2.destroyAllWindows()"],
                capture_output=True, timeout=timeout)
            return r.returncode == 0
        except Exception:
            return False

    # -- drawing helpers -------------------------------------------------
    def _text(self, img, txt, xy, color=_FG, scale=0.42, thick=1):
        self.cv2.putText(img, txt, xy, self.cv2.FONT_HERSHEY_SIMPLEX,
                         scale, color, thick, self.cv2.LINE_AA)

    def _tile(self, rgb, label):
        """One camera tile: RGB uint8 -> upscaled BGR with a caption."""
        cv2 = self.cv2
        if rgb is None:
            side = 224 * _SCALE
            tile = np.full((side, side, 3), 40, np.uint8)
            self._text(tile, "no frame", (side // 2 - 40, side // 2), _WARN, 0.6)
        else:
            bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
            tile = cv2.resize(bgr, None, fx=_SCALE, fy=_SCALE,
                              interpolation=cv2.INTER_NEAREST)
        h = tile.shape[0]
        strip = np.full((26, tile.shape[1], 3), _BG, np.uint8)
        self._text(strip, label, (8, 18), _ACCENT, 0.45)
        return np.vstack([strip, tile])

    def _panel(self, height, request, action, info):
        cv2 = self.cv2
        panel = np.full((height, _PANEL_W, 3), _BG, np.uint8)
        y = 24

        def line(txt, color=_FG, dy=17, scale=0.42):
            nonlocal y
            self._text(panel, txt, (10, y), color, scale)
            y += dy

        line("SENT TO POLICY", _ACCENT, 22, 0.46)

        prompt = (request or {}).get("prompt", "")
        line("prompt", _DIM)
        for chunk in [prompt[i:i + 34] for i in range(0, len(prompt), 34)] or ["(empty)"]:
            line(f'  "{chunk}"' if prompt else "  (empty)")
        y += 4

        jp = (request or {}).get("observation/joint_position")
        line("joint_position (rad)", _DIM)
        if jp is not None:
            for i in range(0, 7, 2):
                pair = "  ".join(f"j{k+1} {jp[k]:+.3f}" for k in range(i, min(i + 2, 7)))
                line("  " + pair)
        else:
            line("  --")
        y += 4

        gp = (request or {}).get("observation/gripper_position")
        line("gripper_position", _DIM)
        if gp is not None:
            g = float(np.asarray(gp).ravel()[0])
            line(f"  {g:.2f} DROID  ({'closed' if g > 0.5 else 'open'})")
            line(f"  {1.0 - g:.2f} crisp", _DIM)
        else:
            line("  --")
        y += 6

        line("ACTION OUT", _ACCENT, 22, 0.46)
        if action is not None:
            v = np.abs(action.raw_velocity)
            line(f"  |v| max  {v.max():.3f}   mean {v.mean():.3f}")
            line(f"  gripper  {action.gripper:.0f} crisp "
                 f"(droid {action.raw_gripper:.0f})")
        else:
            line("  --", _DIM)
        y += 6

        line("LOOP", _ACCENT, 22, 0.46)
        for k, val in info.items():
            line(f"  {k:<11s} {val}", _DIM)
        return panel

    # -- public ----------------------------------------------------------
    def update(self, request, action, info) -> bool:
        """Redraw. Returns False if the user pressed q/ESC to quit."""
        if not self.available:
            return True
        cv2 = self.cv2
        ext = (request or {}).get("observation/exterior_image_1_left")
        wri = (request or {}).get("observation/wrist_image_left")

        tiles = np.hstack([
            self._tile(ext, f"exterior_image_1_left  ({self.external_camera})"),
            self._tile(wri, f"wrist_image_left  ({self.wrist_camera})"),
        ])
        panel = self._panel(tiles.shape[0], request, action, info)
        frame = np.hstack([tiles, panel])

        try:
            cv2.imshow(self.WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
        except Exception as e:
            print(f"[visualize] window failed ({e}); disabling")
            self.available = False
            return True
        return key not in (ord("q"), 27)

    def close(self):
        if self.available:
            try:
                self.cv2.destroyWindow(self.WINDOW)
            except Exception:
                pass


# ---------------------------------------------------------------------------
