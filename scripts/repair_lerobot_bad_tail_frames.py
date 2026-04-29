#!/usr/bin/env python

"""Repair LeRobot datasets where an episode tail image frame leaks from the next episode.

The repair drops the last timestep of affected episodes, then rewrites the frame parquet,
episode metadata, and affected chunked video files into a new dataset directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import av
import datasets
import numpy as np
import pandas as pd
from PIL import Image
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.utils import get_hf_features_from_features, write_stats


def load_info(root: Path) -> dict:
    with (root / "meta" / "info.json").open() as f:
        info = json.load(f)
    for feature in info["features"].values():
        if "shape" in feature:
            feature["shape"] = tuple(feature["shape"])
    return info


def video_keys(info: dict) -> list[str]:
    return [key for key, ft in info["features"].items() if ft.get("dtype") == "video"]


def load_episodes(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(path)
        df["__meta_path"] = str(path.relative_to(root))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No episode metadata parquet files under {root / 'meta/episodes'}")
    return pd.concat(frames, ignore_index=True)


def load_frame_data(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No frame parquet files under {root / 'data'}")
    return pd.concat(frames, ignore_index=True)


def infer_bad_tail_episodes(
    root: Path,
    episodes: pd.DataFrame,
    info: dict,
    same_threshold: float,
    jump_threshold: float,
) -> dict[int, list[str]]:
    fps = int(info["fps"])
    bad: dict[int, list[str]] = {}

    for ep_idx in tqdm(range(len(episodes) - 1), desc="Scan episode tail frames"):
        row = episodes.loc[ep_idx]
        next_row = episodes.loc[ep_idx + 1]
        bad_keys = []

        for key in video_keys(info):
            chunk_idx = int(row[f"videos/{key}/chunk_index"])
            file_idx = int(row[f"videos/{key}/file_index"])
            next_chunk_idx = int(next_row[f"videos/{key}/chunk_index"])
            next_file_idx = int(next_row[f"videos/{key}/file_index"])
            video_path = root / info["video_path"].format(
                video_key=key, chunk_index=chunk_idx, file_index=file_idx
            )
            next_video_path = root / info["video_path"].format(
                video_key=key, chunk_index=next_chunk_idx, file_index=next_file_idx
            )

            from_ts = float(row[f"videos/{key}/from_timestamp"])
            prev_ts = from_ts + (int(row["length"]) - 2) / fps
            last_ts = from_ts + (int(row["length"]) - 1) / fps
            next_ts = float(next_row[f"videos/{key}/from_timestamp"])

            decoder = VideoDecoder(str(video_path), seek_mode="approximate")
            prev_last = decoder.get_frames_at(
                indices=[round(prev_ts * fps), round(last_ts * fps)]
            ).data.float()
            next_decoder = decoder if video_path == next_video_path else VideoDecoder(
                str(next_video_path), seek_mode="approximate"
            )
            next_frame = next_decoder.get_frames_at(indices=[round(next_ts * fps)]).data.float()[0]

            last_frame = prev_last[1]
            prev_frame = prev_last[0]
            last_next = float((last_frame - next_frame).abs().mean() / 255.0)
            prev_last_delta = float((prev_frame - last_frame).abs().mean() / 255.0)

            if last_next < same_threshold and prev_last_delta > jump_threshold:
                bad_keys.append(key)

        if bad_keys:
            bad[int(row["episode_index"])] = bad_keys

    return bad


def vcodec_from_info(info: dict, video_key: str) -> str:
    codec = info["features"][video_key].get("info", {}).get("video.codec", "")
    if codec == "av1":
        return "libsvtav1"
    if codec in {"h264", "hevc"}:
        return codec
    return "libsvtav1"


def encode_video_batches(
    decoder: VideoDecoder,
    output_path: Path,
    frame_indices: list[int],
    fps: int,
    vcodec: str,
    batch_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    if tmp_path.exists():
        tmp_path.unlink()

    container = av.open(str(tmp_path), "w")
    stream = None
    try:
        for start in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[start : start + batch_size]
            frames = decoder.get_frames_at(indices=batch_indices).data
            np_frames = frames.detach().cpu().numpy()
            if np_frames.shape[1] in (1, 3):
                np_frames = np.transpose(np_frames, (0, 2, 3, 1))
            np_frames = np.clip(np_frames, 0, 255).astype(np.uint8)

            if stream is None:
                options = {"g": "2", "crf": "30"}
                if vcodec == "libsvtav1":
                    options["preset"] = "12"
                stream = container.add_stream(vcodec, fps, options=options)
                stream.pix_fmt = "yuv420p"
                stream.height, stream.width = int(np_frames.shape[1]), int(np_frames.shape[2])

            for np_frame in np_frames:
                frame = av.VideoFrame.from_image(Image.fromarray(np_frame).convert("RGB"))
                for packet in stream.encode(frame):
                    container.mux(packet)

        if stream is None:
            raise RuntimeError(f"No frames to encode for {output_path}")
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    tmp_path.replace(output_path)


def rewrite_data_parquet(output_root: Path, info: dict, data: pd.DataFrame) -> None:
    data_path = output_root / info["data_path"].format(chunk_index=0, file_index=0)
    if data_path.exists():
        data_path.unlink()
    data_path.parent.mkdir(parents=True, exist_ok=True)

    features = get_hf_features_from_features(info["features"])
    ds = datasets.Dataset.from_dict(data.to_dict(orient="list"), features=features, split="train")
    table = ds.with_format("arrow")[:]
    import pyarrow.parquet as pq

    writer = pq.ParquetWriter(data_path, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def recompute_numeric_stats(output_root: Path, info: dict, data: pd.DataFrame, old_stats: dict) -> None:
    stats = old_stats.copy()
    for key, ft in info["features"].items():
        if ft["dtype"] in {"video", "image", "string"} or key not in data.columns:
            continue
        values = np.asarray(data[key].tolist() if data[key].dtype == object else data[key].to_numpy())
        if values.ndim == 1:
            keepdims = True
            axis = 0
        else:
            keepdims = False
            axis = 0
        stats[key] = get_feature_stats(values, axis=axis, keepdims=keepdims)

    total_frames = len(data)
    for key, ft in info["features"].items():
        if ft["dtype"] in {"video", "image"} and key in stats:
            stats[key]["count"] = np.array([total_frames], dtype=np.int64)
    write_stats(stats, output_root)


def repair_dataset(
    source_root: Path,
    output_root: Path,
    bad_episodes: set[int] | None,
    overwrite: bool,
    batch_size: int,
    same_threshold: float,
    jump_threshold: float,
    drop_last_all: bool,
) -> dict[int, list[str]]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root}")
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)

    info = load_info(source_root)
    episodes = load_episodes(source_root)
    data = load_frame_data(source_root)
    original_episodes = episodes.copy()
    fps = int(info["fps"])

    if drop_last_all:
        bad_episodes = {int(ep_idx) for ep_idx in episodes["episode_index"].tolist()}
        bad_map = {ep: ["drop_last_all"] for ep in sorted(bad_episodes)}
    elif bad_episodes is None:
        bad_map = infer_bad_tail_episodes(source_root, episodes, info, same_threshold, jump_threshold)
        bad_episodes = set(bad_map)
    else:
        bad_map = {ep: ["manual"] for ep in sorted(bad_episodes)}

    lengths = {
        int(row["episode_index"]): int(row["length"]) - (1 if int(row["episode_index"]) in bad_episodes else 0)
        for _, row in episodes.iterrows()
    }

    drop_mask = np.zeros(len(data), dtype=bool)
    for ep_idx in bad_episodes:
        old_length = int(episodes.loc[episodes["episode_index"] == ep_idx, "length"].iloc[0])
        drop_mask |= (data["episode_index"] == ep_idx) & (data["frame_index"] == old_length - 1)
    data = data.loc[~drop_mask].copy().reset_index(drop=True)
    data["index"] = np.arange(len(data), dtype=np.int64)

    cumulative = 0
    for idx, row in episodes.sort_values("episode_index").iterrows():
        ep_idx = int(row["episode_index"])
        new_length = lengths[ep_idx]
        episodes.at[idx, "length"] = new_length
        episodes.at[idx, "dataset_from_index"] = cumulative
        episodes.at[idx, "dataset_to_index"] = cumulative + new_length
        cumulative += new_length

    for key in video_keys(info):
        chunk_col = f"videos/{key}/chunk_index"
        file_col = f"videos/{key}/file_index"
        from_col = f"videos/{key}/from_timestamp"
        to_col = f"videos/{key}/to_timestamp"

        for (chunk_idx, file_idx), group in tqdm(
            episodes.groupby([chunk_col, file_col], sort=True),
            desc=f"Rewrite {key}",
        ):
            ep_indices = {int(ep) for ep in group["episode_index"].tolist()}
            if not ep_indices.intersection(bad_episodes):
                continue

            original_group = original_episodes[
                (original_episodes[chunk_col] == chunk_idx) & (original_episodes[file_col] == file_idx)
            ].sort_values("episode_index")
            source_video = source_root / info["video_path"].format(
                video_key=key, chunk_index=int(chunk_idx), file_index=int(file_idx)
            )
            output_video = output_root / info["video_path"].format(
                video_key=key, chunk_index=int(chunk_idx), file_index=int(file_idx)
            )
            decoder = VideoDecoder(str(source_video), seek_mode="approximate")

            frame_indices: list[int] = []
            file_frame_cursor = 0
            for _, original_row in original_group.iterrows():
                ep_idx = int(original_row["episode_index"])
                old_start = round(float(original_row[from_col]) * fps)
                new_length = lengths[ep_idx]
                frame_indices.extend(range(old_start, old_start + new_length))
                repair_idx = episodes.index[episodes["episode_index"] == ep_idx][0]
                episodes.at[repair_idx, from_col] = file_frame_cursor / fps
                episodes.at[repair_idx, to_col] = (file_frame_cursor + new_length) / fps
                file_frame_cursor += new_length

            encode_video_batches(
                decoder=decoder,
                output_path=output_video,
                frame_indices=frame_indices,
                fps=fps,
                vcodec=vcodec_from_info(info, key),
                batch_size=batch_size,
            )

    rewrite_data_parquet(output_root, info, data)

    for meta_path, group in episodes.groupby("__meta_path", sort=False):
        out_path = output_root / meta_path
        group.drop(columns=["__meta_path"]).to_parquet(out_path, index=False)

    info["total_frames"] = int(len(data))
    with (output_root / "meta" / "info.json").open("w") as f:
        json.dump(info, f, indent=2)

    with (source_root / "meta" / "stats.json").open() as f:
        old_stats = json.load(f)
    recompute_numeric_stats(output_root, info, data, old_stats)
    return bad_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--same-threshold", type=float, default=0.001)
    parser.add_argument("--jump-threshold", type=float, default=0.005)
    parser.add_argument(
        "--drop-last-all",
        action="store_true",
        help="Drop the final timestep from every episode instead of detecting or listing bad episodes.",
    )
    parser.add_argument(
        "--bad-episodes",
        type=str,
        default=None,
        help="Comma-separated episode indices. If omitted, scan videos and infer them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bad_episodes = None
    if args.bad_episodes:
        bad_episodes = {int(part) for part in args.bad_episodes.split(",") if part.strip()}

    bad_map = repair_dataset(
        source_root=args.source_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        bad_episodes=bad_episodes,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        same_threshold=args.same_threshold,
        jump_threshold=args.jump_threshold,
        drop_last_all=args.drop_last_all,
    )
    print(f"Repaired {len(bad_map)} episodes: {sorted(bad_map)}")


if __name__ == "__main__":
    main()
