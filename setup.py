import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kosmoss",
    version="0.0.1",
    author="alxyok",
    author_email="alxyok@naiama.com",
    url=None,
    license='MIT',
    platforms=['Unix'],
    description="Bootcamps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.7",
    entry_points= {
        "distutils.commands": [
            "download = kosmoss.dataproc.download:Download",
        ]
    },
    package_data={
        "kosmoss": ["config/*"],
    }
)