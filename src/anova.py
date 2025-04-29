from anova.local import generate_jasp_ready_data
from anova.kaggle import generate_jasp_ready_data_kaggle

# Define input and output
input_csv = "results/summary_local.csv"
output_dir = "anova/data/local"

# Generate without variance
generate_jasp_ready_data(input_csv, output_dir, expand_variance=False)

# Generate with variance
generate_jasp_ready_data(input_csv, output_dir, expand_variance=True)

# Define input and output for Kaggle
input_csv_kaggle = "results/summary_kaggle.csv"
output_dir_kaggle = "anova/data/kaggle"

generate_jasp_ready_data_kaggle(input_csv_kaggle, output_dir_kaggle)
