# Copyright (c) 2023, Albert Gu, Tri Dao.
import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform
import shutil
from setuptools import setup, find_packages
import subprocess
import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# ---- Optional Torch import to support dependency resolution / non-CUDA systems ----
try:
    import torch
    from torch.utils.cpp_extension import (
        BuildExtension,
        CUDAExtension,
        CUDA_HOME,
        HIP_HOME,
    )
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    CUDA_HOME = None
    HIP_HOME = None
    BuildExtension = CUDAExtension = None  # type: ignore

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "mamba_ssm"
BASE_WHEEL_URL = "https://github.com/state-spaces/mamba/releases/download/{tag_name}/{wheel_name}"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("MAMBA_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("MAMBA_SKIP_CUDA_BUILD", "FALSE") == "TRUE"

# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("MAMBA_FORCE_CXX11_ABI", "FALSE") == "TRUE"

# ---------------- Dependency-resolution detection (kept from your fork) ----------------
def is_dependency_resolution():
    """
    Detect if we're being called for dependency resolution rather than actual building.
    Allows this package to be listed as an optional dependency without forcing CUDA/ROCm.
    """
    import traceback
    stack = traceback.extract_stack()
    # Heuristics for setuptools.build_meta during resolution
    for frame in stack:
        if "setuptools/build_meta.py" in frame.filename:
            if any(
                method in frame.name
                for method in ["get_requires_for_build", "_get_build_requires"]
            ):
                return True
    # If torch/toolchains are missing and we didn't explicitly skip CUDA build, assume resolution
    if not TORCH_AVAILABLE:
        return True
    if (CUDA_HOME is None and HIP_HOME is None) and not SKIP_CUDA_BUILD:
        try:
            subprocess.check_output(["nvcc", "--version"], stderr=subprocess.DEVNULL)
            return False
        except Exception:
            return True
    return False

def get_platform():
    """Returns the platform name as used in wheel filenames."""
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError(f"Unsupported platform: {sys.platform}")

def get_cuda_bare_metal_version(cuda_dir):
    if not cuda_dir:
        return None, None
    try:
        raw_output = subprocess.check_output(
            [os.path.join(cuda_dir, "bin", "nvcc"), "-V"], universal_newlines=True
        )
        output = raw_output.split()
        release_idx = output.index("release") + 1
        bare_metal_version = parse(output[release_idx].split(",")[0])
        return raw_output, bare_metal_version
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None, None

def get_hip_version(rocm_dir):
    hipcc_bin = "hipcc" if rocm_dir is None else os.path.join(rocm_dir, "bin", "hipcc")
    try:
        raw_output = subprocess.check_output([hipcc_bin, "--version"], universal_newlines=True)
    except Exception as e:
        print(f"hip installation not found: {e} ROCM_PATH={os.environ.get('ROCM_PATH')}")
        return None, None
    for line in raw_output.split("\n"):
        if "HIP version" in line:
            rocm_version = parse(line.split()[-1].rstrip("-").replace("-", "+"))
            return line, rocm_version
    return None, None

def get_torch_hip_version():
    if not TORCH_AVAILABLE or not torch.version.hip:
        return None
    return parse(torch.version.hip.split()[-1].rstrip("-").replace("-", "+"))

def check_if_hip_home_none(global_option: str) -> None:
    if HIP_HOME is not None:
        return
    warnings.warn(
        f"{global_option} was requested, but hipcc was not found.  Are you sure your environment has hipcc available?"
    )

def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )

def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]

cmdclass = {}
ext_modules = []

# ---------------- Build extensions only if truly building (not during resolution) ----------------
if TORCH_AVAILABLE and not is_dependency_resolution() and not SKIP_CUDA_BUILD:
    HIP_BUILD = bool(torch.version.hip)
    print(f"\n\ntorch.__version__  = {torch.__version__}\n\n")
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])
    cc_flag = []
    bare_metal_version = None  # robust default

    if HIP_BUILD:
        check_if_hip_home_none(PACKAGE_NAME)
        rocm_home = os.getenv("ROCM_PATH")
        _, hip_version = get_hip_version(rocm_home)
        if HIP_HOME is not None and hip_version:
            if hip_version < Version("6.0"):
                raise RuntimeError(
                    f"{PACKAGE_NAME} is only supported on ROCm 6.0 and above.  "
                    "Note: make sure HIP has a supported version by running hipcc --version."
                )
            if hip_version == Version("6.0"):
                warnings.warn(
                    f"{PACKAGE_NAME} requires a patch to be applied when running on ROCm 6.0. "
                    "Refer to the README.md for detailed instructions.",
                    UserWarning,
                )
        cc_flag.append("-DBUILD_PYTHON_PACKAGE")
    else:
        check_if_cuda_home_none(PACKAGE_NAME)

        if CUDA_HOME is not None:
            _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
            if bare_metal_version and bare_metal_version < Version("11.6"):
                raise RuntimeError(
                    f"{PACKAGE_NAME} is only supported on CUDA 11.6 and above.  "
                    "Note: make sure nvcc has a supported version by running nvcc -V."
                )

        # ---- Upstream arch matrix (merged), with safe guards ----
        # Historically supported:
        if bare_metal_version and bare_metal_version <= Version("12.9"):
            cc_flag += ["-gencode", "arch=compute_53,code=sm_53"]
            cc_flag += ["-gencode", "arch=compute_62,code=sm_62"]
            cc_flag += ["-gencode", "arch=compute_70,code=sm_70"]
            cc_flag += ["-gencode", "arch=compute_72,code=sm_72"]
        else:
            # Conservative baseline when we can't probe: modern arch set
            cc_flag += ["-gencode", "arch=compute_62,code=sm_62"]
            cc_flag += ["-gencode", "arch=compute_70,code=sm_70"]
            cc_flag += ["-gencode", "arch=compute_72,code=sm_72"]

        # Always add these (upstream)
        cc_flag += ["-gencode", "arch=compute_75,code=sm_75"]
        cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]
        cc_flag += ["-gencode", "arch=compute_87,code=sm_87"]

        # Newer arch support conditionally
        if bare_metal_version and bare_metal_version >= Version("11.8"):
            cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]
        if bare_metal_version and bare_metal_version >= Version("12.8"):
            cc_flag += ["-gencode", "arch=compute_100,code=sm_100"]
            cc_flag += ["-gencode", "arch=compute_120,code=sm_120"]
        if bare_metal_version and bare_metal_version >= Version("13.0"):
            cc_flag += ["-gencode", "arch=compute_103,code=sm_103"]
            cc_flag += ["-gencode", "arch=compute_110,code=sm_110"]
            cc_flag += ["-gencode", "arch=compute_121,code=sm_121"]

    # Match upstream’s CXX11 ABI toggle
    if FORCE_CXX11_ABI and TORCH_AVAILABLE:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    if TORCH_AVAILABLE and HIP_BUILD:
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                f"--offload-arch={os.getenv('HIP_ARCHITECTURES', 'native')}",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-fgpu-flush-denormals-to-zero",
            ]
            + cc_flag,
        }
    else:
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ]
                + cc_flag
            ),
        }
    ext_modules.append(
        CUDAExtension(
            name="selective_scan_cuda",
            sources=[
                "csrc/selective_scan/selective_scan.cpp",
                "csrc/selective_scan/selective_scan_fwd_fp32.cu",
                "csrc/selective_scan/selective_scan_fwd_fp16.cu",
                "csrc/selective_scan/selective_scan_fwd_bf16.cu",
                "csrc/selective_scan/selective_scan_bwd_fp32_real.cu",
                "csrc/selective_scan/selective_scan_bwd_fp32_complex.cu",
                "csrc/selective_scan/selective_scan_bwd_fp16_real.cu",
                "csrc/selective_scan/selective_scan_bwd_fp16_complex.cu",
                "csrc/selective_scan/selective_scan_bwd_bf16_real.cu",
                "csrc/selective_scan/selective_scan_bwd_bf16_complex.cu",
            ],
            extra_compile_args=extra_compile_args,
            include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
        )
    )

def get_package_version():
    with open(Path(this_dir) / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("MAMBA_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)

def get_wheel_url():
    if not TORCH_AVAILABLE:
        return None, None
    # Determine the version numbers that will be used to determine the correct wheel
    torch_version_raw = parse(torch.__version__)
    HIP_BUILD = bool(torch.version.hip)
    if HIP_BUILD:
        torch_hip_version = get_torch_hip_version()
        if not torch_hip_version:
            return None, None
        hip_ver = f"{torch_hip_version.major}{torch_hip_version.minor}"
    else:
        torch_cuda_version = parse(torch.version.cuda) if torch.version.cuda else None
        if not torch_cuda_version:
            return None, None
        # For CUDA 11, compile for 11.8; for CUDA 12, for 12.3 (compat with minor versions)
        torch_cuda_version = parse("11.8") if torch_cuda_version.major == 11 else parse("12.3")
        cuda_version = f"{torch_cuda_version.major}"
    gpu_compute_version = hip_ver if HIP_BUILD else cuda_version
    cuda_or_hip = "hip" if HIP_BUILD else "cu"
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    mamba_ssm_version = get_package_version()
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    wheel_filename = f"{PACKAGE_NAME}-{mamba_ssm_version}+{cuda_or_hip}{gpu_compute_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    wheel_url = BASE_WHEEL_URL.format(
        tag_name=f"v{mamba_ssm_version}", wheel_name=wheel_filename
    )
    return wheel_url, wheel_filename

class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is run by pip when it cannot
    find an existing wheel. We try to download a matching prebuilt wheel; otherwise build from source.
    """
    def run(self):
        if FORCE_BUILD or not TORCH_AVAILABLE:
            return super().run()
        wheel_url, wheel_filename = get_wheel_url()
        if not wheel_url:
            print("Cannot determine wheel URL, building from source...")
            return super().run()
        print("Guessing wheel URL: ", wheel_url)
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)
            # Make the archive (lifted from wheel)
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)
            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"
            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            shutil.move(wheel_filename, wheel_path)
        except urllib.error.HTTPError:
            print("Precompiled wheel not found. Building from source...")
            super().run()

# Set up cmdclass based on available extensions (matches upstream structure, keeps your fallback)
if ext_modules and TORCH_AVAILABLE:
    cmdclass = {"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
else:
    cmdclass = {"bdist_wheel": CachedWheelsCommand}

setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "mamba_ssm.egg-info",
        )
    ),
    author="Tri Dao, Albert Gu",
    author_email="tri@tridao.me, agu@cs.cmu.edu",
    description="Mamba state-space model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/state-spaces/mamba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
        "einops",
        "triton",
        "transformers",
        # "causal_conv1d>=1.4.0",
    ],
)