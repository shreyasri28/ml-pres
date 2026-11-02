# Dataset

The repository includes `sms_spam_sample.csv` as a tiny example dataset so the project structure is easy to understand.

For model training and evaluation, use the full **SMS Spam Collection** dataset from the UCI Machine Learning Repository.

Download it with:

```bash
python data/download_data.py
```

The script creates `data/SMSSpamCollection`, a tab-separated file with two fields per row: the label (`ham` or `spam`) and the message text.

The training script automatically prefers the full downloaded dataset when it is present. The bundled sample is only for a basic demonstration and should not be used to report meaningful accuracy.
