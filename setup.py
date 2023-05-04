from setuptools import setup,find_packages,find_namespace_packages
from mb_pytorch.version import version

setup(
    name="mb_pytorch",
    version=version,
    description="Pytorch functions functions package",
    author=["Malav Bateriwala"],
    packages=find_namespace_packages(include=["mb_pytorch.*"]),
    #packages=find_packages(),
    scripts=['scripts/embeddings/emb.py','scripts/extra_utils/dataload_results.py'],
    install_requires=[
        "numpy",
        "mb_pandas",
        "mb_utils",
        "torch",
        "torchvision",
        "tqdm",
        "torchsummary",
        "cv2"],
    python_requires='>=3.8',)
