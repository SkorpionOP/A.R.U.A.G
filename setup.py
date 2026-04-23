from setuptools import setup, find_packages

setup(
    name="rag_memory_extension",
    version="0.1.0",
    description="A drop-in memory extension for LLMs featuring Failure-Driven Learning and GraphRAG.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sentence-transformers",
        "PyPDF2",
        "requests",
        "networkx",
        "streamlit"
    ],
    python_requires=">=3.8",
)
