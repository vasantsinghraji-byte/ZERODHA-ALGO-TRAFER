from setuptools import setup, find_packages

setup(
    name="zerodha-algo-trader",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "PyYAML>=6.0",
        "psycopg2-binary>=2.9.6",
        "redis>=4.6.0",
        "kiteconnect>=4.2.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "python-telegram-bot>=20.3",
        "websocket-client>=1.6.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.4.1",
        ]
    },
    python_requires=">=3.11",
)
