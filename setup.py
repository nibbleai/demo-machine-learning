import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Machine Learning Demo Project",
    version="1.0.1",
    author="Edouard Theron, Florent Pietot",
    author_email="edouard@nibble.ai, florent@nibble.ai",
    description="A sample project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)
