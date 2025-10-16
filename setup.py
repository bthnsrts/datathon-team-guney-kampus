from setuptools import setup, find_packages

setup(
    name="ing_datathon",  # use underscores, not dashes, for package names
    version="0.1.0",
    packages=find_packages(exclude=("venv*", "notebooks*", "logs*")),
    install_requires=[
        "pandas",
        # Add other dependencies here (e.g. numpy, matplotlib, seaborn)
    ],
    python_requires=">=3.8",
)