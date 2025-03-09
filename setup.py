from setuptools import setup, find_packages

setup(
    name='mc_mitigation',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'jax',
        'jaxlib',
        'jaxopt',
        'numpy',
        'scipy',
    ],
    author='Your Name',
    author_email='tyler.a.cox@berkeley.edu',
    description='A package for mutual coupling mitigation using the JAX ecosystem',
    url='https://github.com/tyleracox/mc_mitigation',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)