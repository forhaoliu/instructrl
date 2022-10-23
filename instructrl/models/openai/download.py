import hashlib
import os
import urllib
from typing import Optional

import tqdm
from absl import logging
from tensorflow.io import gfile

DEFAULT_DOWNLOAD_DIR = os.path.expanduser("./cache/")


def hash_file(path):
    return hashlib.sha256(gfile.GFile(path, "rb").read()).hexdigest()


def download(
    url: str, root: str = DEFAULT_DOWNLOAD_DIR, expected_sha256: Optional[str] = None
):
    """Download a file if it does not exist, with a progress bar.

    Based on https://github.com/openai/CLIP/blob/main/clip/clip.py#L4

    Args:
      url (str): URL of file to download.
      root (str): Directory to place the downloaded file.
      expected_sha256: Optional sha256 sum. If provided, checks downloaded file.
    Raises:
      RuntimeError: Downloaded file existed as a directory, or sha256 of dowload
                    does not match expected_sha256.
    Returns:
      download_target (str): path to downloaded file
    """
    os.makedirs(root, exist_ok=True)
    gfile.makedirs(root)
    filename = os.path.basename(url)
    if "?" in filename:
        # strip trailing HTTP GET arguments
        filename = filename[: filename.rindex("?")]

    download_target = os.path.join(root, filename)

    if gfile.exists(download_target):
        if gfile.isdir(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")
        elif expected_sha256:
            if hash_file(download_target) == expected_sha256:
                return download_target
            logging.warning(
                "%s exists, but the SHA256 checksum does not match;"
                "re-downloading the file",
                download_target,
            )

    with gfile.GFile(download_target, "wb") as output:
        with urllib.request.urlopen(url) as source:
            loop = tqdm.tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            )
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and hash_file(download_target) != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target
