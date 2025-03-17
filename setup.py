from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="str2speech",
    version="0.2.4",
    author="Ashraff Hathibelagal",
    description="A powerful, Transformer-based text-to-speech (TTS) tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hathibelagal-dev/str2speech",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "transformers>=4.49.0",
        "torch",
        "torchvision",
        "torchaudio",
        "tokenizers",
        "scipy>=1.13.1",
        "accelerate",
        "numpy==1.26.4",
        "kokoro>=0.8.4",
        "soundfile",
        "gitpython",
        "moshi",
        "torchtune",
        "torchao",
        "huggingface_hub",
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
