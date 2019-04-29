from setuptools import setup
from setuptools import find_packages


setup(
    name                 = "ml_common",
    version              = "1.0",
    keywords             = ("machine learning"),
    description          = "common libs for machine learning",
    long_description     = "common libs for machine learning",
    author               = "ZS",
    author_email         = "",
    packages             = find_packages(),
    include_package_data = True,
    platforms            = "any",
    install_requires     = [],
    scripts              = [],
    zip_safe             = False,
)
