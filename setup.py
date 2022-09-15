#!/usr/bin/env python
# Thanks and credits to https://github.com/navdeep-G/setup.py for setup.py format
import io
import os
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup

from coomm.version import VERSION

# Package meta-data.
NAME = "COOMM"
DESCRIPTION = "Control Oriented Muscle Model of an octopus arm"
URL = "https://github.com/hanson-hschang/COOMM"
EMAIL = "hschang2@illinois.edu"
AUTHOR = "Heng-Sheng Chang'"
REQUIRES_PYTHON = ">=3.7.0"

# What packages are required for this module to be executed?
REQUIRED = ["matplotlib>=0.2.2.post2"]

# What packages are optional?
EXTRAS = {
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    #packages=[""],
    package_dir={"coomm": "./coomm"},
    packages=find_packages(),
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
    ],
    download_url=f"https://github.com/hanson-hschang/COOMM/archive/refs/tags/{VERSION}.tar.gz",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    cmdclass={
        'upload': UploadCommand,
    },
)
