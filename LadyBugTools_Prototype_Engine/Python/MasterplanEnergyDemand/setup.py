#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.readlines()

test_requirements = ['pytest>=3', ]

setup(
    author="Tristan Gerrish",
    author_email='tristan.gerrish@burohappold.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    description="A utility to calculate the annual energy demand of a masterplan.",
    entry_points={
        'console_scripts': [
            'masterplanenergydemand=masterplanenergydemand.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='masterplanenergydemand',
    name='masterplanenergydemand',
    packages=find_packages(include=['masterplanenergydemand', 'masterplanenergydemand.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/tg359/masterplanenergydemand',
    version='0.1.0',
    zip_safe=False,
)
