import unittest
import joblib
from sklearn.cluster import KMeans
import pandas as pd
import os
import numpy as np

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.model_path = 'model/customer_segmentation.pkl'
        self.data_path = 'mall.csv'
        
    def test_data_file_exists(self):
        """Test if Mall_Customers.csv exists"""
        self.assertTrue(os.path.exists(self.data_path), 
                       "Mall_Customers.csv file not found")
    
    def test_model_file_exists(self):
        """Test if the model file exists"""
        self.assertTrue(os.path.exists(self.model_path), 
                       "Model file not found. Run train.py first.")
    
    def test_model_loading(self):
        """Test if the model can be loaded"""
        try:
            model = joblib.load(self.model_path)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Failed to load model: {str(e)}")
    
    def test_model_type(self):
        """Test if the loaded model is KMeans (not RandomForestClassifier)"""
        model = joblib.load(self.model_path)
        self.assertIsInstance(model, KMeans, 
                             "Model should be KMeans for clustering")
    
    def test_model_clusters(self):
        """Test if the model has correct number of clusters"""
        model = joblib.load(self.model_path)
        self.assertEqual(model.n_clusters, 5, 
                        "Model should have 5 clusters")
    
    def test_model_prediction(self):
        """Test if the model can make predictions on Mall Customers data"""
        model = joblib.load(self.model_path)
        
        # Load real Mall_Customers data for testing
        data = pd.read_csv(self.data_path)
        X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
        
        # Take first 5 rows for testing
        sample_data = X.head(5)
        
        predictions = model.predict(sample_data)
        self.assertEqual(len(predictions), 5, 
                        "Should predict 5 clusters for 5 samples")
        
        # Check if predictions are in valid range (0 to 4)
        for pred in predictions:
            self.assertGreaterEqual(pred, 0)
            self.assertLess(pred, 5)
    
    def test_data_structure(self):
        """Test if Mall_Customers.csv has correct structure"""
        data = pd.read_csv(self.data_path)
        
        # Check required columns
        required_columns = ['CustomerID', 'Genre', 'Age', 
                           'Annual Income (k$)', 'Spending Score (1-100)']
        
        for col in required_columns:
            self.assertIn(col, data.columns, f"Missing column: {col}")
        
        # Check data types and ranges
        self.assertTrue(data['Age'].dtype in [np.int64, np.float64], 
                       "Age should be numeric")
        self.assertTrue(data['Annual Income (k$)'].dtype in [np.int64, np.float64], 
                       "Annual Income should be numeric")
        self.assertTrue(data['Spending Score (1-100)'].dtype in [np.int64, np.float64], 
                       "Spending Score should be numeric")
    
    def test_clustered_output(self):
        """Test if clustered customers file is created"""
        clustered_file = 'data/clustered_customers.csv'
        
        if os.path.exists(clustered_file):
            clustered_data = pd.read_csv(clustered_file)
            
            # Check if Cluster column exists
            self.assertIn('Cluster', clustered_data.columns, 
                         "Clustered data should have Cluster column")
            
            # Check cluster values are in correct range
            cluster_values = clustered_data['Cluster'].unique()
            for val in cluster_values:
                self.assertGreaterEqual(val, 0)
                self.assertLess(val, 5)
    
    def test_full_prediction_pipeline(self):
        """Test the complete prediction pipeline"""
        model = joblib.load(self.model_path)
        data = pd.read_csv(self.data_path)
        
        # Use the same features as training
        X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
        
        # Predict all customers
        all_predictions = model.predict(X)
        
        # Check we have predictions for all customers
        self.assertEqual(len(all_predictions), len(data), 
                        "Should have predictions for all customers")
        
        # Check all predictions are valid
        unique_clusters = set(all_predictions)
        self.assertLessEqual(len(unique_clusters), 5, 
                            "Should not have more than 5 unique clusters")
        
        for cluster in unique_clusters:
            self.assertGreaterEqual(cluster, 0)
            self.assertLess(cluster, 5)

if __name__ == '__main__':
    unittest.main(verbosity=2)
