#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.md") as history_file:
    history = history_file.read()

requirements = [
    "numpy",
    "python-glmnet>=2.6.1",
    "extendanything",
    "scikit-learn",
    "pandas",
    "genetools",
    "multiclass-metrics",
    "typing-extensions",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Maxim Zaslavsky",
    author_email="maxim@maximz.com",
    name="wrap_glmnet",
    description="Wrap Glmnet",
    packages=find_packages(include=["wrap_glmnet", "wrap_glmnet.*"]),
    python_requires=">=3.10",
    version="0.1.1",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/maximz/wrap-glmnet",
    zip_safe=False,
)
