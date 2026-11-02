# Data Folder

This folder stores datasets used by the spam detection project.

Recommended dataset for the project:
- SMS Spam Collection dataset from the UCI Machine Learning Repository
- File format: tab-separated text with `label` and `text`

Quick setup:

1. Run the download script:
   ```bash
   python data/download_data.py
   ```
2. Ensure the data file is named `SMSSpamCollection` or `SMSSpamCollection.csv`.
3. The training script will automatically use the sample file if a downloaded dataset is not available.
