import numpy as np
import plotly.graph_objects as go

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