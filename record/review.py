"""Offline Flask review server for compact recorder datasets.

Run with ``python -m record.review --data-dir PATH``.  This module only reads
recordings (apart from a review flag stored in an episode's metadata) and does
not create robot clients or input-device connections.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_file

from record.dataset import Dataset, Episode, load_dataset


def _json_value(value):
    """Turn pandas/numpy values into JSON values without changing nulls."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _columns(episode: Episode) -> dict:
    return {name: [_json_value(value) for value in episode.table[name].tolist()]
            for name in episode.table.columns}


class ReviewStore:
    """A deliberately small in-memory index backed by the dataset loader."""

    def __init__(self, data_dir: str | Path):
        self.dataset: Dataset = load_dataset(data_dir)
        self.episodes = {episode.path.name: episode for episode in self.dataset.episodes}

    def episode(self, episode_id: str) -> Episode:
        episode = self.episodes.get(episode_id)
        if episode is None:
            abort(404, description=f"unknown episode {episode_id!r}")
        return episode

    @staticmethod
    def cameras(episode: Episode) -> list[str]:
        return [camera["name"] for camera in episode.meta.get("cameras", [])]

    def index(self) -> list[dict]:
        items = []
        for episode_id, episode in self.episodes.items():
            meta = episode.meta
            t_rel = episode.table.get("t_rel")
            duration = float(t_rel.iloc[-1]) if t_rel is not None and len(t_rel) else 0.0
            items.append({
                "id": episode_id,
                "task": meta.get("task", ""),
                "start_time": meta.get("start_wall_time", ""),
                "duration_s": duration,
                "n_frames": episode.n_frames,
                "control_space": meta.get("control_space", ""),
                "bad": bool(meta.get("bad", False)),
                "notes": meta.get("operator_notes", meta.get("notes", "")),
                "cameras": self.cameras(episode),
            })
        return items

    def save_flag(self, episode: Episode, bad: bool, notes: str) -> None:
        path = episode.path / "meta.json"
        # Read from disk here so a review operation never drops keys added by
        # another recorder/version between app startup and this POST.
        meta = json.loads(path.read_text())
        meta["bad"] = bad
        meta["operator_notes"] = notes
        path.write_text(json.dumps(meta, indent=2) + "\n")
        episode.meta = meta


def create_app(data_dir: str | Path) -> Flask:
    """Create the review app, loading and validating the dataset once."""
    store = ReviewStore(data_dir)
    app = Flask(__name__)
    app.config["review_store"] = store

    @app.get("/")
    def page():
        return send_file(Path(__file__).with_name("review.html"), mimetype="text/html")

    @app.get("/api/episodes")
    def episodes():
        return jsonify(store.index())

    @app.get("/api/episode/<episode_id>")
    def episode_data(episode_id):
        episode = store.episode(episode_id)
        return jsonify({"id": episode_id, "meta": episode.meta, **_columns(episode)})

    @app.get("/video/<episode_id>/<camera>.mp4")
    def video(episode_id, camera):
        episode = store.episode(episode_id)
        if camera not in store.cameras(episode):
            abort(404, description=f"unknown camera {camera!r}")
        path = episode.path / "video" / f"{camera}.mp4"
        if not path.is_file():
            abort(404, description=f"missing video for camera {camera!r}")
        # conditional=True is essential: it gives browser seeking a real 206
        # Content-Range response instead of silently making video unseekable.
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.get("/api/episode/<episode_id>/thumbnail.jpg")
    def thumbnail(episode_id):
        episode = store.episode(episode_id)
        path = episode.path / "thumbnail.jpg"
        if not path.exists():
            cameras = store.cameras(episode)
            if not cameras:
                abort(404, description="episode has no cameras")
            image = episode.frame(cameras[0], 0)
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not ok:
                abort(500, description="could not encode thumbnail")
            path.write_bytes(encoded.tobytes())
        return send_file(path, mimetype="image/jpeg", conditional=True)

    @app.post("/api/episode/<episode_id>/flag")
    def flag(episode_id):
        episode = store.episode(episode_id)
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("bad"), bool) or \
                not isinstance(body.get("notes"), str):
            abort(400, description="expected JSON body {'bad': bool, 'notes': str}")
        store.save_flag(episode, body["bad"], body["notes"])
        return jsonify({"bad": body["bad"], "notes": body["notes"]})

    return app


# A conventional alias makes the app easy to embed in small local scripts.
build_app = create_app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Review compact recorder trajectories locally")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=7100)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    app = create_app(args.data_dir)
    print(f"Trajectory review: http://{args.host}:{args.port}/", flush=True)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
