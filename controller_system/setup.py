#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    scripts=["scripts/PID_controller", "scripts/DDQL_controller"],
    packages=["controller_system"],
    package_dir={'': 'src'},
)

setup(**setup_args)
