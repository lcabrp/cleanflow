import unittest
import pandas as pd
import numpy as np
from cleanflow.pipeline import AutomatedCleaner

class TestCleanFlow(unittest.TestCase):
    def setUp(self):
        self.cleaner = AutomatedCleaner()
        self.test_df = pd.DataFrame({
            'name': ['Alice', 'bob', np.nan],
            'age': [25, np.nan, 30],
            'salary': ['$50,000', '$60,000', np.nan]
        })

    def test_full_pipeline(self):
        cleaned_df, report = self.cleaner.clean(self.test_df)
        
        # Check if missing values were handled
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)
        
        # Check if salary was converted to numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['salary']))
        
        # Check if quality report exists
        self.assertIn('initial_status', report)
        self.assertIn('final_status', report)

if __name__ == '__main__':
    unittest.main()