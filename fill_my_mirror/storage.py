import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


class R2Client:
    """Cloudflare R2 client using the S3-compatible API.

    Credentials are read from environment variables:
        CF_R2_ACCOUNT_ID, CF_R2_ACCESS_KEY_ID,
        CF_R2_SECRET_ACCESS_KEY, CF_R2_BUCKET_NAME
    """

    def __init__(self):
        account_id = os.environ["CF_R2_ACCOUNT_ID"]
        self._bucket = os.environ["CF_R2_BUCKET_NAME"]
        self._s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["CF_R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["CF_R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )

    def upload_file(self, local_path: Path, r2_key: str) -> None:
        self._s3.upload_file(str(local_path), self._bucket, r2_key)

    def upload_dir(self, local_dir: Path, r2_prefix: str) -> None:
        for file in local_dir.rglob("*"):
            if file.is_file():
                relative = file.relative_to(local_dir)
                key = f"{r2_prefix.rstrip('/')}/{relative.as_posix()}"
                self._s3.upload_file(str(file), self._bucket, key)

    def download_file(self, r2_key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self._bucket, r2_key, str(local_path))

    def download_dir(self, r2_prefix: str, local_dir: Path) -> None:
        """Download all keys under r2_prefix into local_dir, preserving relative structure."""
        for key in self.list_keys(r2_prefix):
            relative = key[len(r2_prefix.rstrip("/")) + 1:]
            local_path = local_dir / relative
            self.download_file(key, local_path)

    def download_dirs(self, r2_prefixes: list[str], local_dir: Path) -> None:
        """Download multiple R2 prefixes into local_dir."""
        for prefix in r2_prefixes:
            self.download_dir(prefix, local_dir)

    def list_keys(self, prefix: str) -> list[str]:
        keys = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def key_exists(self, r2_key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=r2_key)
            return True
        except ClientError:
            return False
