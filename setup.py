from setuptools import find_packages, setup

# Package metadata
NAME = "SAM 2"
VERSION = "1.0"
DESCRIPTION = "SAM 2: Segment Anything in Images and Videos"
URL = "https://github.com/facebookresearch/segment-anything-2"
AUTHOR = "Meta AI"
AUTHOR_EMAIL = "segment-anything@meta.com"
LICENSE = "Apache 2.0"

# Read the contents of README file
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies
REQUIRED_PACKAGES = [
    "torch>=2.3.1",
    "torchvision>=0.18.1",
    "numpy>=1.24.4",
    "tqdm>=4.66.1",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=9.4.0",
    "opencv-python>=4.7.0",
]

EXTRA_PACKAGES = {
    "demo": ["matplotlib>=3.9.1", "jupyter>=1.0.0", "opencv-python>=4.7.0"],
    "dev": ["black==24.2.0", "usort==1.0.2", "ufmt==2.0.0b2"],
}

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(exclude="notebooks"),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    python_requires=">=3.10.0",
)
