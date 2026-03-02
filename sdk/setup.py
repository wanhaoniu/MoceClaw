from setuptools import setup, find_packages

setup(
    name="soarmMoce-sdk",
    version="0.1.0",
    description="SoarmMoce Python SDK (minimal)",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=["numpy>=1.21", "pyyaml>=6.0"],
    extras_require={
        "sim": ["pybullet"],
        "dev": ["pytest", "black", "ruff"],
    },
)
