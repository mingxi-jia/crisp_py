"""Read the compact recorder dataset format without robot dependencies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import av
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class Episode:
    path: Path
    meta: dict
    table: pd.DataFrame

    @property
    def n_frames(self) -> int:
        return len(self.table)

    def frames(self, camera: str):
        """Yield decoded RGB frames in row order."""
        video = self.path / "video" / f"{camera}.mp4"
        if not video.exists():
            raise KeyError(f"{self.path.name}: no video for camera {camera!r}")
        with av.open(str(video)) as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")

    def frame(self, camera: str, index: int):
        """Decode one RGB frame by index (PyAV seeking is codec-dependent)."""
        if not 0 <= index < self.n_frames:
            raise IndexError(index)
        for current, image in enumerate(self.frames(camera)):
            if current == index:
                return image
        raise RuntimeError(f"{self.path.name}: video {camera!r} ended at {current + 1} frames")


def _video_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def load_episode(path) -> Episode:
    path = Path(path)
    try:
        meta = json.loads((path / "meta.json").read_text())
        table = pq.read_table(path / "data.parquet").to_pandas()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"invalid episode {path}: missing {e.filename}") from e
    n_rows = len(table)
    for camera in meta.get("cameras", []):
        name = camera["name"]
        n_video = _video_count(path / "video" / f"{name}.mp4")
        if n_video != n_rows:
            raise ValueError(f"{path.name}: frame/row invariant violated for {name}: "
                             f"video has {n_video} frames, parquet has {n_rows} rows")
    return Episode(path, meta, table)


@dataclass
class Dataset:
    root: Path
    manifest: dict
    episodes: list[Episode]


def load_dataset(root) -> Dataset:
    root = Path(root)
    manifest = json.loads((root / "dataset.json").read_text())
    episodes = [load_episode(root / name) for name in manifest.get("episodes", [])]
    return Dataset(root, manifest, episodes)
