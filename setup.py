from setuptools import setup, find_packages

setup(
    name="TransBnEn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "transbn=TransBnEn.__main__:main",
        ],
    },
    author="Asif Iqbal",
    author_email="asif.iqbal@example.com",
    description="Bengali-English transliteration tool",
    keywords="bengali, transliteration, bangla, translation",
    python_requires=">=3.6",
)
