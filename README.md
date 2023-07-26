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

Some key dependencies outside of council include:
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) for using a local model to create text embeddings 
- [LlamaIndex](https://github.com/jerryjliu/llama_index) in creating a vector index for document retrieval
- [PandasAI](https://github.com/gventuri/pandas-ai) for analyzing stock prices by making pandas dataframes conversational 

## Initializing the Agent

For this demo, the Agent will be a financial analyst for Microsoft.

The Agent needs a company's documentation and historical price data to use in answering user questions.

The company documentation is stored as a pdf file in the `document_data` directory and the company market data (such as stock prices) is stored in the `market_data` directory in either a csv or json. When the Agent is initialized, it will load the data from those directories. We provide Microsoft's 2022 10-K financial report in `/document_data/msft-10K-2022.pdf` and historical stock price and volume data starting from 2021 in `/market_data/MSFT.csv`. 

We also set the `COMPANY_NAME` and `COMPANY_TICKER` variables to *Microsoft* and *MSFT* in `constants.py`.

For document retrieval, the Agent extracts the text from the pdf, builds a vector index and persists it in the `storage` directory. Upon initialization, the Agent checks the `storage` directory if the vector index can be loaded from disk instead.

See `constants.py` for additional configurable parameters, such as version of OpenAI LLM models.

The steps to initialize the agent and its tools are in `config.py` and `agent_config.py`.

## Running the Agent

The Agent can be run using the `/notebooks/4_financial_analyst_agent.ipynb` notebook that recreates the project's source code in a step-by-step process.

Running `example.py` showcases how to load an Agent from the script and ask it a question. 

We can ask the Agent the following example question: 
```
What is the financial performance of Microsoft?
```

Here is a sample response produced by the Agent:
```
Microsoft's financial performance in fiscal year 2022 has been quite strong, as evidenced by data from their 10-K report, Google search results, and historical stock price and trading data.

According to the 10-K report, Microsoft saw significant growth in various areas. Enterprise Services revenue increased by 7%, driven by growth in Enterprise Support Services. Operating income increased by 25%, and gross margin increased by 22%, driven by growth in Azure and other cloud services. More Personal Computing revenue increased by 10%, driven by growth in Windows OEM, Windows Commercial, Search and news advertising, and Windows. Revenue from Microsoft Cloud increased by 32% to $91.2 billion. Revenue from Office Commercial products and cloud services increased by 13%, driven by Office 365 Commercial growth of 18%. Revenue from LinkedIn increased by 34%, and revenue from Server products and cloud services increased by 28%, driven by Azure and other cloud services growth of 45%.

Google search results corroborate this strong performance, reporting record results in fiscal year 2022 with $198 billion in revenue and $83 billion in operating income. Microsoft's cloud business was highlighted as a strong performer in its fourth-quarter and fiscal year results.

Historical stock price and trading data also indicate a positive financial performance. The opening price of Microsoft stock on January 4, 2021, was $222.53, and it reached a high of $223.00. The lowest point for the day was $214.81, and the closing price was $217.69. The adjusted closing price, which takes into account factors such as dividends and stock splits, was $212.88. The trading volume for that day was 37,130,100 shares. While there have been fluctuations in the stock price, Microsoft has generally performed well.

In conclusion, Microsoft's financial performance in fiscal year 2022 has been strong, with significant growth in revenue, operating income, and gross margin, driven largely by its cloud services. This is reflected in the company's stock price, which has generally trended upwards despite some fluctuations.
```