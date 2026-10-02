"""Synthetic checks for the local trajectory review server.

Run with ``python -m record.test_review``.  The fixture uses the recorder
test's synthetic ``sample`` helper and never contacts a robot or recorder
process.
"""

import json
import tempfile
from html.parser import HTMLParser
from pathlib import Path

from record.recorder import EpisodeWriter, RecorderConfig
from record.review import create_app
from record.test_recorder import sample


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    assert cond, name


class MarkupCheck(HTMLParser):
    """HTMLParser is enough to catch accidental malformed inline markup."""

    def error(self, message):
        raise AssertionError(message)


def make_dataset(root: Path):
    """Use record.test_recorder.sample rather than a second data generator."""
    specs = {"agentview": {"width": 32, "height": 24},
             "inhand": {"width": 20, "height": 12}}
    config = RecorderConfig(root, record_fps=15.0, min_frames=1, task="review fixture",
                            notes="initial notes", control_space="joint")
    paths = []
    for episode_id in (7, 8):
        writer = EpisodeWriter(config, episode_id, specs).start()
        for frame in range(24):
            writer.write(sample(frame, stale=(frame == 5)))
        paths.append(writer.finalize())
    return paths


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "dataset"
        paths = make_dataset(root)
        app = create_app(root)
        client = app.test_client()
        episode_id = paths[0].name

        print("=== review API ===")
        response = client.get("/api/episodes")
        items = response.get_json()
        documented = {"id", "task", "start_time", "duration_s", "n_frames",
                      "control_space", "bad", "notes", "cameras"}
        check("episode index has documented keys", response.status_code == 200 and
              len(items) == 2 and documented <= set(items[0]))
        check("episode index uses loader cameras", items[0]["cameras"] == ["agentview", "inhand"])

        response = client.get(f"/api/episode/{episode_id}")
        payload = response.get_json()
        n_frames = payload["meta"]["n_frames"]
        parquet_columns = set(app.config["review_store"].episode(episode_id).table.columns)
        check("episode response is columnar", response.status_code == 200 and
              all(len(payload[name]) == n_frames for name in parquet_columns))
        required_columns = {
            "frame_index", "timestamp", "t_rel", "joint_position", "ee_position",
            "ee_quat_xyzw", "fingertip_position", "fingertip_quat_xyzw", "gripper_value",
            "gripper_width_mm", "ft_wrench", "action_fingertip_position",
            "action_fingertip_quat_xyzw", "action_grasp", "action_gripper",
            "action_delta", "spacemouse_motion", "spacemouse_button", "action_timestamp",
            "action_joint_position", "agentview_age_s", "agentview_stale", "inhand_age_s",
            "inhand_stale",
        }
        check("all review-chart parquet columns are emitted", required_columns <= set(payload) and
              parquet_columns <= set(payload))
        check("action grasp round-trips as binary integers", set(payload["action_grasp"]) <= {0, 1} and
              all(isinstance(value, int) and not isinstance(value, bool)
                  for value in payload["action_grasp"]))

        video_path = paths[0] / "video" / "agentview.mp4"
        full = client.get(f"/video/{episode_id}/agentview.mp4")
        ranged = client.get(f"/video/{episode_id}/agentview.mp4", headers={"Range": "bytes=0-1023"})
        size = video_path.stat().st_size
        check("video full GET works", full.status_code == 200 and len(full.data) == size)
        check("video range GET has precise seek headers", ranged.status_code == 206 and
              ranged.headers.get("Accept-Ranges") == "bytes" and
              ranged.headers.get("Content-Range") == f"bytes 0-1023/{size}" and
              len(ranged.data) == 1024, ranged.headers.get("Content-Range", ""))

        thumbnail = client.get(f"/api/episode/{episode_id}/thumbnail.jpg")
        check("thumbnail is a JPEG", thumbnail.status_code == 200 and
              thumbnail.data.startswith(b"\xff\xd8\xff"))

        before = json.loads((paths[0] / "meta.json").read_text())
        flagged = client.post(f"/api/episode/{episode_id}/flag",
                              json={"bad": True, "notes": "camera blocked"})
        after = json.loads((paths[0] / "meta.json").read_text())
        reloaded = create_app(root).test_client().get(f"/api/episode/{episode_id}").get_json()["meta"]
        check("flag persists without losing metadata", flagged.status_code == 200 and
              after["bad"] and after["operator_notes"] == "camera blocked" and
              after["schema_version"] == before["schema_version"] and
              reloaded["bad"] and reloaded["operator_notes"] == "camera blocked")
        check("unknown episode returns 404", client.get("/api/episode/no_such_episode").status_code == 404)

        print("\n=== offline page ===")
        html = (Path(__file__).with_name("review.html")).read_text()
        parser = MarkupCheck()
        parser.feed(html)
        parser.close()
        check("review page is offline self-contained markup", "http://" not in html and
              "https://" not in html and "<script" in html and "<style" in html)

    print("\nAll review tests passed.")


if __name__ == "__main__":
    main()
