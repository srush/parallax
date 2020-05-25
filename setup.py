from setuptools import setup

setup(
    name="parallax",
    version="0.1",
    author="Alexander Rush",
    author_email="arush@cornell.edu",
    packages=[
        "parallax",
    ],
    package_data={"parallax": []},
    url="https://github.com/srus/parallax",
    install_requires=["jax", "jaxlib", "frozendict"]
)
