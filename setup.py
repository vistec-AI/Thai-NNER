from setuptools import find_packages, setup

with open("README.md","r",encoding="utf-8-sig") as f:
    readme = f.read()

with open("requirements.txt","r",encoding="utf-8-sig") as f:
    requirements = [i.strip() for i in f.readlines()]

setup(
    name="thai_nner",
    version="0.1",
    description="Thai Nested Named Entity Recognition",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Weerayut Buaphet",
    author_email="weerayut.fame@gmail.com",
    url="https://github.com/vistec-AI/Thai-NNER",
    packages=find_packages(),
    python_requires=">=3.6",
    package_data={
        "thai_nner": [
            "*",
        ]
    },
    install_requires=requirements,
    license="MIT License ",
    zip_safe=False,
    keywords=[
        "Thai",
        "NLP",
        "natural language processing",
        "text analytics",
        "text processing",
        "localization",
        "computational linguistics",
        "Thai language",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stabl",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: General",
        "Topic :: Text Processing :: Linguistic",
    ],
    project_urls={
        "Source": "https://github.com/vistec-AI/Thai-NNER",
        "Bug Reports": "https://github.com/vistec-AI/Thai-NNER/issues",
    },
)