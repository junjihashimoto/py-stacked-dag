from setuptools import setup, find_packages


setup(
    name = "stackeddag",
    version = "0.1",
    packages = find_packages(),
    install_requires = [
        'pydot'
    ]
)
