# DiNaM: Disinformation Narrative Mining with Large Language Models

This repository provides tools for mining disinformation narratives based on fact-checking articles. The implementation is described in the paper *"DiNaM: Disinformation Narrative Mining with Large Language Models"*. Fact-checking articles used in the paper are available under a license from EDMO.

## Installation

1. Ensure Python 3.9 or higher is installed on your machine.
2. Install CUDA 12.2 or higher if you're planning to run the code on a GPU:
   - Make sure the version of CUDA is compatible with your GPU driver and the version of PyTorch or other libraries you're using.
   
2. Clone this repository to your local machine:
   ```bash
   git clone <repository_url>
   ```
3. Navigate to the project directory:
   ```bash
   cd <project_directory>
   ```
4. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Download the Dataset

For convenience, the full dataset of fact-checking articles should be downloaded into the `data/raw` directory. You can access it via the following link:  
[Download Fact-Checking Dataset](https://drive.google.com/file/d/1EQjcC2gd68w_IHxGAHQHxnv_8Zq46_eo/view?usp=sharing)

Once downloaded, move the dataset to the appropriate directory:
```bash
mkdir -p data/raw
mv <path_to_downloaded_file> data/raw/
```

### 2. Identify False Information
This step processes fact-checking articles through various stages, including filtering, extracting claims, verifying claims, and refining the information. Run the following command to execute the pipeline:

```bash
python script/identify_false_information.py --fact_checking_articles_path <path_to_articles>
```

#### Arguments:
- `--fact_checking_articles_path`: Path to the raw fact-checking articles (default: `./data/raw/fact_checking_articles.csv`).
- `--filtered_articles_path`: Path to save the filtered articles (default: `./data/processed/filtered_articles.csv`).
- `--extracted_claims_path`: Path to save the extracted claims (default: `./data/processed/extracted_claims.csv`).
- `--verified_claims_path`: Path to save the verified claims (default: `./data/processed/verified_claims.csv`).
- `--refined_claims_path`: Path to save the refined claims (default: `./data/processed/refined_claims.csv`).
- `--api_key`: API key for OpenAI generative model (optional, if using the OpenAI generative model).

This script performs the following stages:
1. **Filtering Articles**: Filters irrelevant articles.
2. **Extracting Claims**: Identifies and extracts claims from the articles.
3. **Verifying Claims**: Verifies extracted claims against reference data.
4. **Refining Claims**: Refines verified claims for further analysis.

### 3. Cluster False Information
Cluster the instances of false information by performing dimensionality reduction and clustering on refined claims. Run the following command:

```bash
python script/cluster_false_information.py --refined_claims <path_to_refined_claims>
```

#### Arguments:
- `--refined_claims`: Path to the refined claims (default: `./data/processed/refined_claims.csv`).
- `--clustered_claims_path`: Path to save the clustered claims (default: `./data/processed/clustered_claims.csv`).
- `--start_date`: Start date for filtering claims (default: `2021-07-01`).
- `--end_date`: End date for filtering claims (default: `2023-02-01`).
- `--umap_params`: Parameters for UMAP dimensionality reduction (default: `{ "n_components": 256, "n_neighbors": 15 }`).
- `--hdbscan_params`: Parameters for HDBSCAN clustering (default: `{ "min_cluster_size": 10, "min_samples": 15 }`).

This script performs the following steps:
1. **Date Filtering**: Filters refined claims within the specified date range.
2. **Dimensionality Reduction**: Reduces the dimensionality of claims data using UMAP.
3. **Clustering**: Clusters the reduced data using HDBSCAN.

### 4. Derive Disinformation Narratives
Derive disinformation narratives by analyzing clusters of false information. Run the following command:

```bash
python script/derive_disinformation_narratives.py --clustered_claims_path <path_to_clustered_claims>
```

#### Arguments:
- `--clustered_claims_path`: Path to the clustered claims CSV file (default: `./data/processed/clustered_claims.csv`).
- `--derived_narratives_path`: Path to save the derived narratives CSV file (default: `./data/processed/predicted_narratives_ds.csv`).
- `--predicted_narratives_path`: Path to save the predicted narratives text file (default: `./data/processed/predicted_narratives.txt`).
- `--api_key`: API key for the OpenAI generative model (optional, if using the OpenAI generative model).

This script performs the following steps:
1. **Analyze Clusters**: Uses clustered claims to identify patterns and groupings.
2. **Generate Narratives**: Derives disinformation narratives from clustered data using a generative model.

## Testing

The repository includes test scripts in the `tests` directory to evaluate the different components of the pipeline. Below are the available tests:

- `test_filtering.py`: Tests the filtering of irrelevant articles.
- `test_extraction.py`: Tests the extraction of claims from articles.
- `test_clustering.py`: Tests the clustering of refined claims.
- `test_narratives.py`: Tests the derivation of disinformation narratives.

### Example: Testing Filtering
Run the following command to evaluate the false claims classification:

```bash
python tests/test_filtering.py --api_key <your_openai_api_key> --claims_classification_path <path_to_ground_truth>
```

#### Arguments:
- `--api_key`: API key for OpenAI generative model (default: provided placeholder key).
- `--claims_classification_path`: Path to the ground truth claims classification CSV file (default: `./data/raw/ground_truth_false_claims_classification.csv`).

### Example: Testing Extraction
Run the following command to evaluate the claim extraction process:

```bash
python tests/test_extraction.py --api_key <your_openai_api_key> --ground_truth_fact_checking_articles_path <path_to_ground_truth>
```

#### Arguments:
- `--api_key`: API key for OpenAI generative model (default: provided placeholder key).
- `--ground_truth_fact_checking_articles_path`: Path to the ground truth fact-checking articles CSV file (default: `./data/raw/ground_truth_article_claim_extraction.csv`).
- `--filtered_articles_path`: Path to save the filtered articles (default: `./data/processed/filtered_articles.csv`).
- `--extracted_claims_path`: Path to save the extracted claims (default: `./data/processed_test/extracted_claims.csv`).
- `--verified_claims_path`: Path to save the verified claims (default: `./data/processed_test/verified_claims.csv`).
- `--refined_claims_path`: Path to save the refined claims (default: `./data/processed_test/refined_claims.csv`).

### Example: Testing Clustering
Run the following command to test the clustering process:

```bash
python tests/test_clustering.py --refined_narratives_path <path_to_refined_claims>
```

#### Arguments:
- `--refined_narratives_path`: Path to the refined narratives CSV file (default: `./data/processed/refined_claims.csv`).
- `--embedding_paths`: Path to save or load embeddings (default: `./data/embeddings/embeddings.pt`).
- `--embedding_model_name`: Name of the embedding model to use (default: `Salesforce/SFR-Embedding-2_R`).
- `--clustering_output_file`: Path to save clustering results (default: `./data/results/clustering_results.txt`).
- `--date_range_start`: Start date for filtering data (default: `2021-07-01`).
- `--date_range_end`: End date for filtering data (default: `2023-02-01`).

### Example: Testing Narrative Derivation
Run the following command to test the narrative derivation process:

```bash
python tests/test_narratives.py --clustered_claims_path <path_to_clustered_claims>
```

#### Arguments:
- `--clustered_claims_path`: Path to the clustered claims CSV file (default: `./data/processed/clustered_claims.csv`).
- `--derived_narratives_path`: Path to save the derived narratives CSV file (default: `./data/processed_test/predicted_narratives_ds.csv`).
- `--predicted_narratives_path`: Path to save the predicted narratives text file (default: `./data/processed_test/predicted_narratives.txt`).
- `--api_key`: API key for OpenAI generative model (default: provided placeholder key).

## Data Directory Structure

The `data/` directory contains the following files:

1. **raw/fact_checking_articles.csv**  
   - The main dataset containing fact-checking articles required to run DiNaM, including fact-checking details.

2. **gt/ground_truth_filtering.csv**  
   - Used to test the filtering process. This file contains annotated articles indicating whether the article includes false information (whether the debunked claim is verified as true or false).

3. **gt/ground_truth_extraction.csv**  
   - Contains pairs of articles and false information. This dataset is used to test the extraction of false information from articles.

4. **gt/ground_truth_narratives.txt**  
   - Provides ground truth disinformation narratives from the period between `2021-07-01` and `2023-02-01`.  
   - This is required for end-to-end testing of DiNaM to derive disinformation narratives. **Note**: The predicted narratives should also correspond to this same period for accurate evaluation.



## Additional Notes
- Ensure that you have an active API key for seamless integration.

Feel free to explore and contribute to the repository!