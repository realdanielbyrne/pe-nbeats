import os
import tempfile
import urllib.request

CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "pe_nbeats")
# Caches written by releases published under the old `lightningnbeats` name.
LEGACY_CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "lightningnbeats")


def cached_file(subdir, filename, url, local_dir=None):
    """Return a local path for a dataset file, downloading it on first use.

    Lookup order: ``local_dir`` (files shipped alongside a source checkout),
    the pe_nbeats cache, the legacy lightningnbeats cache. When none has the
    file it is downloaded into the pe_nbeats cache. The download goes to a
    temporary file that is renamed into place, so an interrupted transfer
    never leaves a truncated CSV behind.
    """
    candidates = []
    if local_dir is not None:
        candidates.append(os.path.join(local_dir, filename))
    target = os.path.join(CACHE_ROOT, subdir, filename)
    candidates += [target, os.path.join(LEGACY_CACHE_ROOT, subdir, filename)]
    for path in candidates:
        if os.path.exists(path):
            return path

    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f"Downloading {filename} to {target} ...")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print("Download complete.")
    return target
