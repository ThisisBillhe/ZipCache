from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "ZIPCACHE"
LONG_DESCRIPTION = "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification"

# Setting up
setup(
    name="zipcache",
    version=VERSION,
    author="Yefei He",
    author_email="billhe@zju.edu.cn",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    keywords=["python", "AI"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
)