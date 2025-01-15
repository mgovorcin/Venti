from setuptools import setup, find_packages

setup(
    name='Venti',
    version='0.0.1',
    description='A Python library for vertical land motion with InSAR and GNSS',
    author='Marin Govorcin',
    author_email='marin.govorcin@jpl.nasa.gov',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Add required libraries here, e.g., 'numpy', 'pandas'
    ],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)