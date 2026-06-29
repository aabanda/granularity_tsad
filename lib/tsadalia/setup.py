import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

requirements = (HERE / "requirements.txt").read_text()

setup(
    name="tsdalia",
    version="0.0.1",
    description="tsdalia",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://git.code.tecnalia.com/core-ia-apps/tsdalia",
    author="Amaia Abanda, Miguel Esteras, et al.",
    author_email="amaia.abanda@tecnalia.com, miguel.esteras@tecnalia.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Optimisation :: Machine Learning",
    ],
    keywords="tecnalia data analysis",
    packages=find_packages(exclude=("tests",)),
    install_requires=["scikit-learn", "pandas", "scipy", "numpy"],
    include_package_data=True,
)
