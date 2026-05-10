"""
Export qualitative comparison sets from R2, ranked by PSNR spread.

The script scans metric files like:

    blender/gt_geometry/black-forest-labs--FLUX.1-Fill-dev/n_<n>_t_<t>/<index>/seed_<seed>_metrics_projected_image.json

For each shared (index, seed), it collects the psnr_constrained value across
all discovered (n, t) combinations, ranks groups by the standard deviation
across those combinations, and downloads the corresponding seed images.

Output layout:

    {output_dir}/00000_index_<index>_seed_<seed>/
        n_<n>_t_<t>.png
        manifest.json

Examples
--------
Export every complete comparison set:

    python scripts/export_qualitative_psnr_std_sets.py \\
        --output-dir outputs/qualitative_psnr_std

Export only the 20 highest-spread sets:

    python scripts/export_qualitative_psnr_std_sets.py \\
        --output-dir outputs/qualitative_psnr_std \\
        --top-k 20
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

from fill_my_mirror.storage import R2Client


DEFAULT_DATASET = "blender"
DEFAULT_GEOM_SUBDIR = "gt_geometry"
DEFAULT_MODEL_SLUG = "black-forest-labs--FLUX.1-Fill-dev"
DEFAULT_METRIC = "psnr_constrained"

METRICS_RE = re.compile(
    r"^"
    r"(?P<dataset>[^/]+)/"
    r"(?P<geom_subdir>[^/]+)/"
    r"(?P<model_slug>[^/]+)/"
    r"n_(?P<n>[^/]+)_t_(?P<t>[^/]+)/"
    r"(?P<index>\d+)/"
    r"seed_(?P<seed>\d+)_metrics_projected_image\.json"
    r"$"
)


@dataclass(frozen=True)
class MetricEntry:
    n: str
    t: str
    index: int
    seed: str
    metric_key: str
    image_key: str
    value: float


@dataclass(frozen=True)
class RankedGroup:
    index: int
    seed: str
    std: float
    mean: float
    entries: tuple[MetricEntry, ...]


def _nt_sort_key(entry: MetricEntry | tuple[str, str]) -> tuple[float, float, str, str]:
    if isinstance(entry, MetricEntry):
        n, t = entry.n, entry.t
    else:
        n, t = entry

    try:
        n_float = float(n)
    except ValueError:
        n_float = math.inf
    try:
        t_float = float(t)
    except ValueError:
        t_float = math.inf
    return n_float, t_float, n, t


def _read_metric_value(r2: R2Client, key: str, metric_name: str) -> float | None:
    with tempfile.NamedTemporaryFile(suffix=".json") as tf:
        r2.download_file(key, Path(tf.name))
        data = json.loads(Path(tf.name).read_text())

    value = data.get(metric_name)
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if math.isfinite(value) else None


def _discover_entries(
    r2: R2Client,
    prefix: str,
    metric_name: str,
) -> tuple[list[MetricEntry], set[tuple[str, str]]]:
    print(f"Listing projected-image metric JSONs under R2:{prefix} ...")
    all_keys = r2.list_keys(prefix)
    metric_keys = [k for k in all_keys if k.endswith("_metrics_projected_image.json")]
    print(f"Found {len(metric_keys)} projected-image metric JSONs")

    entries: list[MetricEntry] = []
    nt_combos: set[tuple[str, str]] = set()

    for i, key in enumerate(sorted(metric_keys), start=1):
        match = METRICS_RE.match(key)
        if not match:
            continue

        value = _read_metric_value(r2, key, metric_name)
        if value is None:
            print(f"  [{i}/{len(metric_keys)}] missing/non-finite {metric_name}: {key}")
            continue

        n = match.group("n")
        t = match.group("t")
        index = int(match.group("index"))
        seed = match.group("seed")
        image_key = key.replace("_metrics_projected_image.json", ".png")

        entries.append(
            MetricEntry(
                n=n,
                t=t,
                index=index,
                seed=seed,
                metric_key=key,
                image_key=image_key,
                value=value,
            )
        )
        nt_combos.add((n, t))

        if i % 250 == 0:
            print(f"  loaded {i}/{len(metric_keys)} metric files")

    return entries, nt_combos


def _rank_groups(
    entries: list[MetricEntry],
    nt_combos: set[tuple[str, str]],
    allow_incomplete: bool,
) -> list[RankedGroup]:
    expected_nt = set(nt_combos)
    by_index_seed: dict[tuple[int, str], dict[tuple[str, str], MetricEntry]] = {}

    for entry in entries:
        key = (entry.index, entry.seed)
        by_index_seed.setdefault(key, {})[(entry.n, entry.t)] = entry

    groups: list[RankedGroup] = []
    skipped_incomplete = 0

    for (index, seed), by_nt in by_index_seed.items():
        missing = expected_nt - set(by_nt)
        if missing and not allow_incomplete:
            skipped_incomplete += 1
            continue

        ordered_entries = tuple(sorted(by_nt.values(), key=_nt_sort_key))
        values = [entry.value for entry in ordered_entries]
        if len(values) < 2:
            continue

        groups.append(
            RankedGroup(
                index=index,
                seed=seed,
                std=float(statistics.pstdev(values)),
                mean=float(statistics.fmean(values)),
                entries=ordered_entries,
            )
        )

    groups.sort(key=lambda g: (-g.std, g.index, int(g.seed)))

    if skipped_incomplete:
        print(
            f"Skipped {skipped_incomplete} incomplete (index, seed) groups "
            f"missing one or more discovered (n, t) combinations"
        )

    return groups


def _write_manifest(group: RankedGroup, out_dir: Path, rank: int) -> None:
    manifest = {
        "rank": rank,
        "index": group.index,
        "seed": group.seed,
        "std": group.std,
        "mean": group.mean,
        "metric": DEFAULT_METRIC,
        "entries": [
            {
                "n": entry.n,
                "t": entry.t,
                "value": entry.value,
                "image_file": f"n_{entry.n}_t_{entry.t}.png",
                "image_key": entry.image_key,
                "metric_key": entry.metric_key,
            }
            for entry in group.entries
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _download_group(
    r2: R2Client,
    group: RankedGroup,
    out_dir: Path,
    rank: int,
    skip_existing: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in group.entries:
        filename = f"n_{entry.n}_t_{entry.t}.png"
        local_path = out_dir / filename
        if skip_existing and local_path.exists():
            continue
        r2.download_file(entry.image_key, local_path)

    _write_manifest(group, out_dir, rank)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export qualitative image sets ranked by psnr_constrained std across (n, t).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--geom-subdir", default=DEFAULT_GEOM_SUBDIR)
    parser.add_argument("--model-slug", default=DEFAULT_MODEL_SLUG)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Only export the top K groups by std. By default, export all ranked groups.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Rank groups even if an (index, seed) is missing some discovered (n, t) combinations.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-download image files that already exist locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = f"{args.dataset}/{args.geom_subdir}/{args.model_slug}/"

    r2 = R2Client()
    entries, nt_combos = _discover_entries(r2, prefix, args.metric)

    if not entries:
        print("No valid metric entries found; nothing to export.")
        return

    print(f"Discovered {len(nt_combos)} (n, t) combinations:")
    for n, t in sorted(nt_combos, key=_nt_sort_key):
        print(f"  n_{n}_t_{t}")

    groups = _rank_groups(entries, nt_combos, allow_incomplete=args.allow_incomplete)
    if args.top_k is not None:
        groups = groups[: args.top_k]

    if not groups:
        print("No rankable groups found; nothing to export.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {len(groups)} group(s) to {args.output_dir} ...")

    width = max(5, len(str(len(groups) - 1)))
    failures = 0
    for rank, group in enumerate(groups):
        group_dir = args.output_dir / f"{rank:0{width}d}_index_{group.index}_seed_{group.seed}"
        try:
            _download_group(
                r2=r2,
                group=group,
                out_dir=group_dir,
                rank=rank,
                skip_existing=args.skip_existing,
            )
            print(
                f"  [{rank + 1}/{len(groups)}] index={group.index} seed={group.seed} "
                f"std={group.std:.6f} mean={group.mean:.6f}"
            )
        except Exception:
            failures += 1
            print(f"  [{rank + 1}/{len(groups)}] FAILED index={group.index} seed={group.seed}")
            traceback.print_exc()

    if failures:
        print(f"\nDone with {failures} failed group(s).")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
