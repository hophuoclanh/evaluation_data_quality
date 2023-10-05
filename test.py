import pandas as pd
from data_quality_metrics import feature_relevance
from data_quality_metrics import feature_correlation
from data_quality_metrics import completeness

# Sample dataset for demonstration
# Load the data
data = pd.read_csv('train.csv')

# Drop columns that won't be used in metric calculations
data_numeric = data.drop(['STUDENT ID'], axis=1)

# Calculate metrics using the provided functions
QoD_FC_value = feature_correlation(data_numeric)
QoD_FR_value = feature_relevance(data_numeric, 'GRADE')  # Assuming 'GRADE' is the label
QoD_Com_value = completeness(data_numeric)

print("QoD^D_FC:", QoD_FC_value)
print("QoD^D_FR:", QoD_FR_value)
print("QoD^D_Com:", QoD_Com_value)
