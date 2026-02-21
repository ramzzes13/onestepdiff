from setuptools import setup, find_packages

setup(
    name="adadmd",
    version="0.1.0",
    description="AdaDMD: Adaptive Distribution Matching Distillation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "diffusers>=0.20",
        "transformers>=4.30",
        "peft>=0.5",
        "numpy",
        "scipy",
    ],
)
