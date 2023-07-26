# council-financial-analyst-agent

A Financial Analyst Agent that has access to company financial documents, real-time search through Google and company historical trading data, implemented in [council](https://github.com/chain-ml/council).

## Tutorial

For a set of notebooks building components of the Financial Analyst Agent with explanations, see the `notebooks` directory.

## Requirements

Rename the file `.env.example` to `.env` and fill in your OpenAI API key.

You can optionally fill in the `GOOGLE_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID` keys to allow the Agent to use Google Search. Instructions for generating those keys can be found in `/notebooks/2_google_search.ipynb`.

To install dependencies, run the command
```
pip install -r requirements.txt
```

## Initializing the Agent

For this demo, the Agent will be a financial analyst for Microsoft.

The Agent needs a company's documentation and historical price data to use in answering user questions.

The company documentation is stored as a pdf file in the `document_data` directory and the company market data (such as stock prices) is stored in the `market_data` directory. When the Agent is initialized, it will load the data from those directories. We provide Microsoft's 2022 10-K financial report in `/document_data/msft-10K-2022.pdf` and historical stock price and volume data starting from 2021 in `/market_data/MSFT.csv`. 

We also set the `COMPANY_NAME` and `COMPANY_TICKER` variables to *Microsoft* and *MSFT* in `constants.py`.

For document retrieval, the Agent extracts the text from the pdf, builds a vector index and persists it in the `storage` directory. Upon initialization, the Agent checks the `storage` directory if the vector index can be loaded from disk instead.

See `constants.py` for additional configurable parameters, such as version of OpenAI LLM models.

## Running the Agent

The Agent can be run using the `/notebooks/4_financial_analyst_agent.ipynb` notebook that recreates the project's source code in a step-by-step process.

Running `example.py` showcases how to load an Agent from the script and ask it a question. 

