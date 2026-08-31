from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

try:
    import duckdb
except ImportError as error:
    raise SystemExit(
        "duckdb is required; run with: uv run --with duckdb python scripts/download_thumbs.py"
    ) from error


REVISION = "c1fb5062957329a8a6bf615966c73ac9f58b8b80"
SHARD_COUNT = 31
URL_TEMPLATE = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/"
    + REVISION
    + "/raw_meta_Clothing_Shoes_and_Jewelry/full-{shard:05d}-of-00031.parquet"
)


def catalog_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [str(json.loads(line)["parent_asin"]) for line in handle if line.strip()]


def fetch_shard(connection: duckdb.DuckDBPyConnection, shard: int) -> list[dict]:
    for attempt in range(6):
        try:
            return [
                {"parent_asin": item_id, "thumb": thumb}
                for item_id, thumb in connection.execute(
                    """
                    SELECT m.parent_asin,
                           COALESCE(
                               list_extract(m.images.thumb, list_position(m.images.variant, 'MAIN')),
                               list_extract(m.images.thumb, 1)
                           ) AS thumb
                    FROM read_parquet(?) m
                    JOIN wanted w USING (parent_asin)
                    """,
                    [URL_TEMPLATE.format(shard=shard)],
                ).fetchall()
            ]
        except duckdb.HTTPException:
            if attempt == 5:
                raise
            delay = min(120, 5 * 2**attempt)
            print(f"metadata {shard + 1}/{SHARD_COUNT}: rate limited, retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def extract_urls(ids: list[str], output: Path) -> dict[str, str]:
    cache = output / ".metadata"
    cache.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("CREATE TABLE wanted(parent_asin VARCHAR PRIMARY KEY)")
    connection.executemany("INSERT INTO wanted VALUES (?)", [(item_id,) for item_id in ids])

    urls: dict[str, str] = {}
    for shard in range(SHARD_COUNT):
        cache_path = cache / f"{shard:05d}.jsonl"
        if cache_path.is_file():
            rows = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
        else:
            rows = fetch_shard(connection, shard)
            cache_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
        urls.update(
            (row["parent_asin"], row["thumb"])
            for row in rows
            if row.get("thumb")
        )
        print(f"metadata {shard + 1}/{SHARD_COUNT}: {len(rows)} matched, {len(urls)} URLs", flush=True)
    return urls


def download(item_id: str, url: str, output: Path) -> tuple[str, str | None]:
    destination = output / f"{item_id}.jpg"
    if destination.is_file() and destination.stat().st_size > 0:
        return item_id, None
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        temporary = destination.with_suffix(".jpg.part")
        try:
            with urlopen(request, timeout=30) as response:
                content_type = response.headers.get_content_type()
                if not content_type.startswith("image/"):
                    raise ValueError(f"unexpected content type {content_type}")
                temporary.write_bytes(response.read())
            temporary.replace(destination)
            return item_id, None
        except Exception as error:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                return item_id, str(error)
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Amazon thumbnail images for the catalog.")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/thumbs"))
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    ids = catalog_ids(args.catalog)
    args.output.mkdir(parents=True, exist_ok=True)
    urls = extract_urls(ids, args.output)
    missing = sorted(set(ids) - urls.keys())
    (args.output / ".missing.jsonl").write_text(
        "".join(json.dumps({"parent_asin": item_id}) + "\n" for item_id in missing),
        encoding="utf-8",
    )
    failures: list[dict[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download, item_id, url, args.output): item_id
            for item_id, url in urls.items()
        }
        for future in as_completed(futures):
            item_id, error = future.result()
            completed += 1
            if error:
                failures.append({"parent_asin": item_id, "error": error})
            if completed % 500 == 0 or completed == len(futures):
                print(f"images {completed}/{len(futures)}: {len(failures)} failed", flush=True)

    failure_path = args.output / ".failures.jsonl"
    failure_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    print(
        f"catalog={len(ids)} URLs={len(urls)} downloaded={len(urls) - len(failures)} "
        f"missing_metadata_or_thumb={len(missing)} failed={len(failures)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
