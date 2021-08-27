import setuptools

setuptools.setup(
    name="causal_caption_setup",
    version="0.0.1",
    author="yejin",
    author_email="zinyizhen8@gmail.com",
    description="setup.py for causal_captions",
    url="https://github.com/ajamjoom/Image-Captions",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    #package_dir={"": "src"},
    packages=setuptools.find_packages(where="/home/muser/Context/Image-Captions/Cause_Caption/"),
    python_requires=">=3.6",
)
