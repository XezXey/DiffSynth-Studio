import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MultiSkeleton3DAnimator:
    """
    Overlay multiple animated skeletons in ONE Plotly figure.

    Usage:
      anim = MultiSkeleton3DAnimator(fps=30)
      anim.add_sequence(X1, edges=edges, color="blue", name="A")
      anim.add_sequence(X2, edges=edges, color="red",  name="B")
      anim.fig.show()
    """

    def __init__(self, fps=30, marker_size=4, axis_pad_ratio=0.05, title="3D Multi-Skeleton Animation"):
        self.fps = fps
        self.marker_size = marker_size
        self.axis_pad_ratio = axis_pad_ratio
        self.title = title

        self.seqs = []   # list of dicts: {"X":..., "edges":..., "color":..., "name":...}
        self.fig = go.Figure(
            layout=go.Layout(
                title=title,
                scene=dict(
                    aspectmode="cube",  # equal x/y/z
                    xaxis=dict(autorange=False, title="X"),
                    yaxis=dict(autorange=False, title="Y"),
                    zaxis=dict(autorange=False, title="Z"),
                ),
                uirevision="fixed",  # keep camera stable
            )
        )
        self._init_controls()

    def _init_controls(self):
        frame_duration_ms = int(1000 / max(self.fps, 1))
        self.fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.02, y=0.02,
                    xanchor="left", yanchor="bottom",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=frame_duration_ms, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[],
        )

    @staticmethod
    def _bones_xyz(frame_xyz, edges):
        if not edges:
            return [], [], []
        xb, yb, zb = [], [], []
        for a, b in edges:
            xb += [frame_xyz[a, 0], frame_xyz[b, 0], None]
            yb += [frame_xyz[a, 1], frame_xyz[b, 1], None]
            zb += [frame_xyz[a, 2], frame_xyz[b, 2], None]
        return xb, yb, zb

    def _recompute_scene_ranges(self):
        # fixed ranges from ALL sequences combined (prevents “shrinking”)
        if not self.seqs:
            return

        all_xyz = np.concatenate([s["X"].reshape(-1, 3) for s in self.seqs], axis=0)
        mins = np.nanmin(all_xyz, axis=0)
        maxs = np.nanmax(all_xyz, axis=0)
        center = (mins + maxs) / 2.0
        spans = (maxs - mins)
        max_span = float(np.nanmax(spans))

        pad = max_span * self.axis_pad_ratio
        half = max_span / 2.0 + pad

        xr = [center[0] - half, center[0] + half]
        yr = [center[1] - half, center[1] + half]
        zr = [center[2] - half, center[2] + half]

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(range=xr, autorange=False),
                yaxis=dict(range=yr, autorange=False),
                zaxis=dict(range=zr, autorange=False),
                aspectmode="cube",
            )
        )

    def _rebuild_animation(self):
        """
        Rebuild frames so that every frame updates ALL skeleton traces.
        Trace layout per sequence:
          - bones trace
          - joints trace
        """
        if not self.seqs:
            return

        # Total number of frames = max T across sequences
        T_max = max(s["X"].shape[0] for s in self.seqs)

        frames = []
        for t in range(T_max):
            frame_data = []
            for s in self.seqs:
                X = s["X"]
                edges = s["edges"]

                t_eff = min(t, X.shape[0] - 1)  # hold last frame if shorter
                ft = X[t_eff]

                xb, yb, zb = self._bones_xyz(ft, edges)

                # IMPORTANT: must match trace order in fig.data
                frame_data.append(go.Scatter3d(x=xb, y=yb, z=zb))
                frame_data.append(go.Scatter3d(x=ft[:, 0], y=ft[:, 1], z=ft[:, 2]))

            frames.append(go.Frame(name=str(t), data=frame_data))

        self.fig.frames = tuple(frames)

        # Slider for new length
        slider = dict(
            x=0.15, y=0.02,
            xanchor="left", yanchor="bottom",
            len=0.8,
            currentvalue=dict(prefix="t="),
            steps=[
                dict(
                    method="animate",
                    label=str(t),
                    args=[
                        [str(t)],
                        dict(frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0),
                             mode="immediate"),
                    ],
                )
                for t in range(T_max)
            ],
        )
        self.fig.update_layout(sliders=[slider])

    def add_sequence(self, X, edges=None, color="blue", name=None, line_width=4, marker_size=None):
        """
        Add another motion to overlay. Does NOT remove existing sequences.
        """
        X = np.asarray(X)
        assert X.ndim == 3 and X.shape[2] == 3, "X must be shape (T, J, 3)"
        edges = edges or []
        marker_size = marker_size if marker_size is not None else self.marker_size
        name = name or f"seq{len(self.seqs)}"

        self.seqs.append(dict(X=X, edges=edges, color=color, name=name, marker_size=marker_size, line_width=line_width))

        # Add two new traces (bones + joints) with initial pose (t=0)
        f0 = X[0]
        xb0, yb0, zb0 = self._bones_xyz(f0, edges)

        self.fig.add_trace(
            go.Scatter3d(
                x=xb0, y=yb0, z=zb0,
                mode="lines",
                line=dict(width=line_width, color=color),
                name=f"{name}-bones",
                showlegend=True,
            )
        )
        self.fig.add_trace(
            go.Scatter3d(
                x=f0[:, 0], y=f0[:, 1], z=f0[:, 2],
                mode="markers",
                marker=dict(size=marker_size, color=color),
                name=f"{name}-joints",
                showlegend=True,
            )
        )

        # Update fixed axes + rebuild frames for all sequences
        self._recompute_scene_ranges()
        self._rebuild_animation()

        return self.fig

class MultiSkeleton2D3DAnimator:
    """
    Overlay multiple animated skeletons in ONE Plotly figure.
    Optionally show 2D keypoints on a side panel (animated, synced).

    Usage:
      anim = MultiSkeleton3DAnimator(fps=30, show_2d=True, y_axis_down=True)
      anim.add_sequence(X3d, edges=edges3d, color="blue", name="A", K2=K2d, edges2d=edges2d)
      anim.add_sequence(X3d_b, edges=edges3d, color="red",  name="B", K2=K2d_b)
      anim.fig.show()
    """

    def __init__(
        self,
        fps=30,
        marker_size=4,
        axis_pad_ratio=0.05,
        title="3D + 2D Multi-Skeleton Animation",
        show_2d=True,
        y_axis_down=True,   # True if 2D keypoints are in image coordinates (y down)
        two_d_aspect="equal"  # "equal" or "auto"
    ):
        self.fps = fps
        self.marker_size = marker_size
        self.axis_pad_ratio = axis_pad_ratio
        self.title = title
        self.show_2d = show_2d
        self.y_axis_down = y_axis_down
        self.two_d_aspect = two_d_aspect

        # Each seq dict can contain:
        # X (T,J,3), edges, color, name, marker_size, line_width,
        # K2 (T,J,2) optional, edges2d optional
        self.seqs = []

        # Build subplot layout: 3D left, 2D right
        if self.show_2d:
            self.fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "scene"}, {"type": "xy"}]],
                column_widths=[0.6, 0.4],
                horizontal_spacing=0.05,
                subplot_titles=("3D", "2D"),
            )
            base_layout = go.Layout(
                title=title,
                scene=dict(
                    aspectmode="cube",
                    xaxis=dict(autorange=False, title="X"),
                    yaxis=dict(autorange=False, title="Y"),
                    zaxis=dict(autorange=False, title="Z"),
                ),
                xaxis=dict(title="u"),
                yaxis=dict(title="v", autorange="reversed" if self.y_axis_down else True),
                uirevision="fixed",
                legend=dict(itemsizing="constant"),
            )
            self.fig.update_layout(base_layout)
        else:
            self.fig = go.Figure(
                layout=go.Layout(
                    title=title,
                    scene=dict(
                        aspectmode="cube",
                        xaxis=dict(autorange=False, title="X"),
                        yaxis=dict(autorange=False, title="Y"),
                        zaxis=dict(autorange=False, title="Z"),
                    ),
                    uirevision="fixed",
                    legend=dict(itemsizing="constant"),
                )
            )

        self._init_controls()

    def _init_controls(self):
        frame_duration_ms = int(1000 / max(self.fps, 1))
        self.fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.02, y=0.02,
                    xanchor="left", yanchor="bottom",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=frame_duration_ms, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[],
        )

    @staticmethod
    def _bones_xyz(frame_xyz, edges):
        if not edges:
            return [], [], []
        xb, yb, zb = [], [], []
        for a, b in edges:
            xb += [frame_xyz[a, 0], frame_xyz[b, 0], None]
            yb += [frame_xyz[a, 1], frame_xyz[b, 1], None]
            zb += [frame_xyz[a, 2], frame_xyz[b, 2], None]
        return xb, yb, zb

    @staticmethod
    def _bones_uv(frame_uv, edges):
        if not edges:
            return [], []
        ub, vb = [], []
        for a, b in edges:
            ub += [frame_uv[a, 0], frame_uv[b, 0], None]
            vb += [frame_uv[a, 1], frame_uv[b, 1], None]
        return ub, vb

    def _recompute_scene_ranges(self):
        # fixed 3D ranges from ALL sequences combined
        if not self.seqs:
            return

        all_xyz = np.concatenate([s["X"].reshape(-1, 3) for s in self.seqs], axis=0)
        mins = np.nanmin(all_xyz, axis=0)
        maxs = np.nanmax(all_xyz, axis=0)
        center = (mins + maxs) / 2.0
        spans = (maxs - mins)
        max_span = float(np.nanmax(spans))

        pad = max_span * self.axis_pad_ratio
        half = max_span / 2.0 + pad

        xr = [center[0] - half, center[0] + half]
        yr = [center[1] - half, center[1] + half]
        zr = [center[2] - half, center[2] + half]

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(range=xr, autorange=False),
                yaxis=dict(range=yr, autorange=False),
                zaxis=dict(range=zr, autorange=False),
                aspectmode="cube",
            )
        )

    def _recompute_2d_ranges(self):
        if not self.show_2d:
            return
        # fixed 2D ranges from ALL sequences that have K2
        all_2d = []
        for s in self.seqs:
            if s.get("K2") is not None:
                all_2d.append(s["K2"].reshape(-1, 2))
        if not all_2d:
            return

        all_uv = np.concatenate(all_2d, axis=0)
        mins = np.nanmin(all_uv, axis=0)
        maxs = np.nanmax(all_uv, axis=0)
        center = (mins + maxs) / 2.0
        spans = (maxs - mins)
        max_span = float(np.nanmax(spans))

        pad = max_span * self.axis_pad_ratio
        half = max_span / 2.0 + pad

        xr = [center[0] - half, center[0] + half]
        yr = [center[1] - half, center[1] + half]

        # For image coords you often want y reversed, but still a fixed numeric range.
        # Plotly's autorange="reversed" handles display direction.
        self.fig.update_xaxes(range=xr, autorange=False, row=1, col=2)
        # self.fig.update_yaxes(range=yr, autorange=False, row=1, col=2)
        self.fig.update_xaxes(range=xr, autorange=False, row=1, col=2)

        if self.y_axis_down:
            self.fig.update_yaxes(range=yr[::-1], autorange=False, row=1, col=2)
        else:
            self.fig.update_yaxes(range=yr, autorange=False, row=1, col=2)

        

        if self.two_d_aspect == "equal":
            # make 2D panel not stretched
            self.fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=2)

    def _rebuild_animation(self):
        """
        Rebuild frames so that every frame updates ALL traces.
        Per sequence trace layout:
          - 3D bones
          - 3D joints
          - 2D bones (optional if K2 provided)
          - 2D joints (optional if K2 provided)
        """
        if not self.seqs:
            return

        T_max = max(s["X"].shape[0] for s in self.seqs)

        frames = []
        for t in range(T_max):
            frame_data = []
            for s in self.seqs:
                X = s["X"]
                edges3d = s["edges"]
                K2 = s.get("K2", None)
                edges2d = s.get("edges2d", None) or edges3d  # default to same topology

                t_eff = min(t, X.shape[0] - 1)
                ft3 = X[t_eff]

                xb, yb, zb = self._bones_xyz(ft3, edges3d)

                # 3D traces
                frame_data.append(go.Scatter3d(x=xb, y=yb, z=zb))
                frame_data.append(go.Scatter3d(x=ft3[:, 0], y=ft3[:, 1], z=ft3[:, 2]))

                # 2D traces (if provided)
                if self.show_2d and (K2 is not None):
                    t_eff2 = min(t, K2.shape[0] - 1)
                    ft2 = K2[t_eff2]

                    ub, vb = self._bones_uv(ft2, edges2d)
                    frame_data.append(go.Scatter(x=ub, y=vb))
                    frame_data.append(go.Scatter(x=ft2[:, 0], y=ft2[:, 1]))

            frames.append(go.Frame(name=str(t), data=frame_data))

        self.fig.frames = tuple(frames)

        slider = dict(
            x=0.15, y=0.02,
            xanchor="left", yanchor="bottom",
            len=0.8,
            currentvalue=dict(prefix="t="),
            steps=[
                dict(
                    method="animate",
                    label=str(t),
                    args=[
                        [str(t)],
                        dict(
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                            mode="immediate",
                        ),
                    ],
                )
                for t in range(T_max)
            ],
        )
        self.fig.update_layout(sliders=[slider])

    def add_sequence(
        self,
        X,
        edges=None,
        color="blue",
        name=None,
        line_width=4,
        marker_size=None,
        # NEW: 2D
        K2=None,             # (T,J,2) or None
        edges2d=None,
        color2d=None,        # optional different color for 2D
        line_width_2d=2,
        marker_size_2d=None,
    ):
        """
        Add another motion to overlay.
        X: (T,J,3)
        K2: optional (T,J,2) to show on the right panel
        """
        X = np.asarray(X)
        assert X.ndim == 3 and X.shape[2] == 3, "X must be shape (T, J, 3)"
        edges = edges or []

        if K2 is not None:
            K2 = np.asarray(K2)
            assert K2.ndim == 3 and K2.shape[2] == 2, "K2 must be shape (T, J, 2)"
            # allow different T, but J must match
            assert K2.shape[1] == X.shape[1], "K2 and X must have same number of joints (J)"

        marker_size = marker_size if marker_size is not None else self.marker_size
        marker_size_2d = marker_size_2d if marker_size_2d is not None else marker_size
        color2d = color2d if color2d is not None else color
        name = name or f"seq{len(self.seqs)}"

        self.seqs.append(
            dict(
                X=X, edges=edges, color=color, name=name,
                marker_size=marker_size, line_width=line_width,
                K2=K2, edges2d=(edges2d or edges),
                color2d=color2d, line_width_2d=line_width_2d, marker_size_2d=marker_size_2d
            )
        )

        # ---- Add initial traces (t=0) ----
        f0 = X[0]
        xb0, yb0, zb0 = self._bones_xyz(f0, edges)

        # 3D traces (left)
        if self.show_2d:
            self.fig.add_trace(
                go.Scatter3d(
                    x=xb0, y=yb0, z=zb0,
                    mode="lines",
                    line=dict(width=line_width, color=color),
                    name=f"{name}-bones",
                    showlegend=True,
                ),
                row=1, col=1
            )
            self.fig.add_trace(
                go.Scatter3d(
                    x=f0[:, 0], y=f0[:, 1], z=f0[:, 2],
                    mode="markers",
                    marker=dict(size=marker_size, color=color),
                    name=f"{name}-joints",
                    showlegend=True,
                ),
                row=1, col=1
            )
        else:
            self.fig.add_trace(
                go.Scatter3d(
                    x=xb0, y=yb0, z=zb0,
                    mode="lines",
                    line=dict(width=line_width, color=color),
                    name=f"{name}-bones",
                    showlegend=True,
                )
            )
            self.fig.add_trace(
                go.Scatter3d(
                    x=f0[:, 0], y=f0[:, 1], z=f0[:, 2],
                    mode="markers",
                    marker=dict(size=marker_size, color=color),
                    name=f"{name}-joints",
                    showlegend=True,
                )
            )

        # 2D traces (right), only if enabled + provided
        if self.show_2d and (K2 is not None):
            f02 = K2[0]
            edges2d_use = edges2d or edges
            ub0, vb0 = self._bones_uv(f02, edges2d_use)

            self.fig.add_trace(
                go.Scatter(
                    x=ub0, y=vb0,
                    mode="lines",
                    line=dict(width=line_width_2d, color=color2d),
                    name=f"{name}-2d-bones",
                    showlegend=True,
                ),
                row=1, col=2
            )
            self.fig.add_trace(
                go.Scatter(
                    x=f02[:, 0], y=f02[:, 1],
                    mode="markers",
                    marker=dict(size=marker_size_2d, color=color2d),
                    name=f"{name}-2d-joints",
                    showlegend=True,
                ),
                row=1, col=2
            )

        # Update fixed axes + rebuild frames for all sequences
        self._recompute_scene_ranges()
        self._recompute_2d_ranges()
        self._rebuild_animation()

        return self.fig