from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="str2speech",
    version="0.1.4",
    author="Ashraff Hathibelagal",
    description="A powerful, Transformer-based text-to-speech (TTS) tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hathibelagal-dev/str2speech",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "transformers",
        "torch",
        "torchvision",
        "torchaudio",
        "scipy",
        "accelerate"
    ],
    entry_points={
        "console_scripts": [
            "str2speech=str2speech.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="ai text-to-speech speech-synthesis nlp transformer voice",
    project_urls={
        "Source": "https://github.com/hathibelagal-dev/str2speech",
        "Tracker": "https://github.com/hathibelagal-dev/str2speech/issues",
    },
)
