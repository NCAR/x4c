from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='x4c',  # required
    version='2025.6.22',
    description='x4c: Xarray for CESM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Feng Zhu',
    author_email='fengzhu@ucar.edu',
    url='https://github.com/NCAR/x4c',
    packages=find_packages(),
    include_package_data=True,
    license='Apache-2.0',
    zip_safe=False,
    keywords='Xarray, CESM, Climate Data Analysis and Visualization',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    install_requires=[
        'netCDF4',
        'xarray',
        'dask',
        'nc-time-axis',
        'colorama',
        'tqdm',
    ],
)
