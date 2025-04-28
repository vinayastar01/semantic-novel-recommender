
# Semantic Book Recommender with LLMs

Welcome to the **Semantic Book Recommender** project! This system uses Large Language Models (LLMs) to provide book recommendations based on natural language descriptions. Whether you're looking for a book about a specific theme or genre, this project allows users to get the most relevant books from a given query.

## Project Overview

The project is built around the concept of semantic (vector) search and integrates several components:

1. **Text Data Cleaning**  
   - Clean and preprocess the raw data for effective use in semantic search.  
   - Code located in: `data-exploration.ipynb`

2. **Semantic (Vector) Search**  
   - Create a vector database to perform similarity searches. Users can input natural language queries to find the most similar books.  
   - Code located in: `vector-search.ipynb`

3. **Text Classification (Zero-shot Classification)**  
   - Classify books into categories such as "fiction" or "non-fiction" using zero-shot classification.  
   - Code located in: `text-classification.ipynb`

4. **Sentiment Analysis**  
   - Extract emotions (e.g., joyful, suspenseful, sad) from book descriptions and sort the books accordingly.  
   - Code located in: `sentiment-analysis.ipynb`

5. **Web Application with Gradio**  
   - A Gradio dashboard for users to easily interact with the recommender system and get personalized book recommendations.  
   - Code located in: `gradio-dashboard.py`

## Requirements

This project was developed using Python 3.11, and the following dependencies are required:

- `kagglehub`
- `pandas`
- `matplotlib`
- `seaborn`
- `python-dotenv`
- `langchain-community`
- `langchain-opencv`
- `langchain-chroma`
- `transformers`
- `gradio`
- `notebook`
- `ipywidgets`

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/semantic-book-recommender.git
   cd semantic-book-recommender
   ```

2. **Create a `.env` file** in the root directory with your OpenAI API key:

   ```text
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Download the Data** from Kaggle. Detailed instructions on how to download the dataset can be found in the repository.

4. **Run the Notebooks** in sequence to:
   - Clean the data (`data-exploration.ipynb`)
   - Build the vector database (`vector-search.ipynb`)
   - Classify books (`text-classification.ipynb`)
   - Perform sentiment analysis (`sentiment-analysis.ipynb`)

5. **Run the Gradio Dashboard**:

   Once all components are set up, run the Gradio web app:

   ```bash
   python gradio-dashboard.py
   ```

   The dashboard will be available at `http://localhost:7860`.

## Adding Images to Your README

You can add images to your README by placing them in a directory (e.g., `images/`) and then linking to them as follows:

```markdown
![Image Alt Text](images/your_image.png)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [freeCodeCamp](https://www.freecodecamp.org/)
- [OpenAI](https://openai.com/)
- [Kaggle](https://www.kaggle.com/)

## Contributing

Contributions are welcome! If you have improvements or suggestions, feel free to fork the repo, create issues, and submit pull requests.
