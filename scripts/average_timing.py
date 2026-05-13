"""Average projection_time_seconds across real/estimated_geometry/<index>/timing.json for indices 0–49."""

import json
import sys

from fill_my_mirror.storage import R2Client

PREFIX = "real/estimated_geometry"
INDICES = range(50)


def main():
    r2 = R2Client()
    times = []
    missing = []

    for index in INDICES:
        key = f"{PREFIX}/{index}/timing.json"
        try:
            obj = r2._s3.get_object(Bucket=r2._bucket, Key=key)
            data = json.loads(obj["Body"].read())
            if data.get("error"):
                print(f"[{index}] skipped (error=True, t={data['projection_time_seconds']:.2f}s)")
                continue
            times.append(data["projection_time_seconds"])
            print(f"[{index}] {data['projection_time_seconds']:.2f}s")
        except Exception as exc:
            print(f"[{index}] MISSING — {exc}")
            missing.append(index)

    if not times:
        print("No valid timing entries found.")
        sys.exit(1)

    avg = sum(times) / len(times)
    total = sum(times)
    print(f"\n--- Results ---")
    print(f"Valid samples : {len(times)}")
    print(f"Missing/error : {len(missing)} {missing if missing else ''}")
    print(f"Total time    : {total:.2f}s ({total/60:.1f} min)")
    print(f"Average time  : {avg:.2f}s ({avg/60:.2f} min)")
    print(f"Min / Max     : {min(times):.2f}s / {max(times):.2f}s")


if __name__ == "__main__":
    main()
