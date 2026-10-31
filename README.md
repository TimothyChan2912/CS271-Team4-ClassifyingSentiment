# Team Information
**Team Number:** Team 4

**Team Members:**
- Timothy Chan
- John Brisebois
- Shivam Shelar


# Setup
Instructions for setting up the project environment.
- Clone the repository
- Create and activate the virtual environment
- Install dependencies using requirements.txt

# Data Access
1. Download the Dataset 
- https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset 
- Download the dataset as a zip file

2. Extract the Dataset
- Unzip the downloaded file
- Move the extracted files into data/raw/ directory

3. Dataset Organization
The project uses the following directory structure:

```text
data/
├── raw/          # Original dataset downloaded from Kaggle
└── processed/    # Cleaned and transformed datasets
```

4. Dataset License and Attribution
- Source: Kaggle
- Dataset: Social Media Sentiments Analysis Dataset
- Author: Kashish 
- License: CC0 1.0 Universal (Public Domain Dedication)
- Dataset URL: https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset?resource=download 

5. Data Processing
The original dataset should remain unchanged in data/raw/. Any cleaning or processing done via the Python script should be saved separately in data/processed/.

# Baseline Run Routes