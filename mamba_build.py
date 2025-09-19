# mamba_build.py
import os, sys, shutil, tempfile, urllib.request, urllib.error
from pathlib import Path
from packaging.version import parse, Version

# Delegate all other hooks to setuptools' backend
from setuptools import build_meta as _orig  # type: ignore
from setuptools.build_meta import *  # re-export PEP 517 hooks

PACKAGE_NAME = "mamba_ssm"
BASE_WHEEL_URL = "https://github.com/state-spaces/mamba/releases/download/{tag_name}/{wheel_name}"

# Env toggles (analogous to setup.py)
FORCE_BUILD = os.getenv("MAMBA_FORCE_BUILD", "FALSE") == "TRUE"

def _safe_import_torch():
    try:
        import torch  # noqa
        return torch
    except Exception:
        return None

def _platform_tag():
    import platform
    m = platform.machine().lower()
    if sys.platform.startswith("linux"):
        if m in ("x86_64", "amd64"):
            return "manylinux2014_x86_64"
        if m in ("aarch64", "arm64"):
            return "manylinux2014_aarch64"
        return "manylinux2014_x86_64"
    elif sys.platform == "darwin":
        return "macosx_11_0_arm64" if m in ("arm64", "aarch64") else "macosx_10_13_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    raise RuntimeError(f"Unsupported platform: {sys.platform} {m}")

def _read_version():
    here = Path(__file__).resolve().parent
    init_py = here / PACKAGE_NAME / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    import re, ast
    m = re.search(r"^__version__\s*=\s*(.*)$", text, re.MULTILINE)
    ver = ast.literal_eval(m.group(1))
    local = os.environ.get("MAMBA_LOCAL_VERSION")
    return f"{ver}+{local}" if local else str(ver)

def _wheel_filename(torch, version_str):
    """
    Matches your setup.py logic:
    mamba_ssm-{ver}+{cuX|hipXY}torch{MM.mm}cxx11abi{TRUE|FALSE}-{cpXY}-{cpXY}-{plat}.whl
    """
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    plat = _platform_tag()
    tver = parse(torch.__version__)
    torch_mm = f"{tver.major}.{tver.minor}"

    if torch.version.hip:
        hip_v = parse(torch.version.hip.split()[-1].replace("-", "+"))
        gpu = f"{hip_v.major}{hip_v.minor}"
        kind = "hip"
    else:
        tcuda = parse(torch.version.cuda) if torch.version.cuda else None
        if not tcuda:
            return None  # CPU-only torch: no upstream GPU wheel to fetch
        # normalize to 11.8/12.3 family like your code
        norm = parse("11.8") if tcuda.major == 11 else parse("12.3")
        gpu = f"{norm.major}"
        kind = "cu"

    try:
        cxx11 = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    except Exception:
        cxx11 = "FALSE"

    return (
        f"{PACKAGE_NAME}-{version_str}"
        f"+{kind}{gpu}torch{torch_mm}cxx11abi{cxx11}-{py}-{py}-{plat}.whl"
    )

def _try_download(version_str, out_dir):
    torch = _safe_import_torch()
    if torch is None:
        return None

    wheel_name = _wheel_filename(torch, version_str)
    if not wheel_name:
        return None

    url = os.getenv(
        "MAMBA_WHEEL_URL",
        BASE_WHEEL_URL.format(tag_name=f"v{version_str}", wheel_name=wheel_name),
    )

    tmpdir = tempfile.mkdtemp()
    tmp_path = Path(tmpdir) / wheel_name
    try:
        print(f"[mamba_build] Trying upstream wheel: {url}")
        urllib.request.urlretrieve(url, tmp_path)
        dest = Path(out_dir) / wheel_name
        shutil.move(str(tmp_path), str(dest))
        print(f"[mamba_build] Downloaded: {dest.name}")
        return dest.name
    except urllib.error.HTTPError as e:
        print(f"[mamba_build] Upstream wheel not found ({e.code}). Falling back.")
        return None
    except Exception as e:
        print(f"[mamba_build] Failed to fetch upstream wheel: {e}. Falling back.")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ---- PEP 517 overrides ----
def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    if FORCE_BUILD:
        print("[mamba_build] FORCE_BUILD=TRUE -> building from source.")
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)
    ver = _read_version()
    got = _try_download(ver, wheel_directory)
    if got:
        return got
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)

def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    # For editable installs we usually don’t want to fetch a wheel;
    # just delegate to setuptools.
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)
