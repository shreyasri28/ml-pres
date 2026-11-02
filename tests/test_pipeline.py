import unittest

from src.preprocess import preprocess_text
from src.predict import predict_message
from src.train import load_dataset, train_model


class SpamDetectionPipelineTests(unittest.TestCase):
    def test_preprocessing_returns_text(self):
        self.assertIsInstance(preprocess_text("URGENT! Visit https://example.com now"), str)

    def test_sample_dataset_has_both_classes(self):
        dataset = load_dataset("data/sms_spam_sample.csv")
        self.assertEqual(set(dataset["label"].unique()), {0, 1})

    def test_training_and_prediction_smoke_test(self):
        result = train_model("data/sms_spam_sample.csv")
        prediction = predict_message(
            "Congratulations! You won a free prize. Claim it now.",
            model=result["model"],
        )
        self.assertIn(prediction["label"], {"Ham", "Spam"})
        self.assertGreaterEqual(prediction["confidence"], 0.0)
        self.assertLessEqual(prediction["confidence"], 1.0)
        self.assertAlmostEqual(
            prediction["probabilities"]["ham"] + prediction["probabilities"]["spam"],
            1.0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
