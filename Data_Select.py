import pandas as pd


def extract_and_save_data(input_file_path, output_file_path, rows_per_label=300):
    # Read the CSV file
    df = pd.read_csv(input_file_path, header=None, names=['Domain', 'Label'])

    # Initialize an empty DataFrame to store the sampled data
    sampled_data = pd.DataFrame(columns=['Domain', 'Label'])

    # Iterate over unique labels
    for label in df['Label'].unique():
        # Sample 2000 rows for each label
        sampled_rows = df[df['Label'] == label].sample(n=rows_per_label, random_state=42)

        # Append the sampled rows to the result DataFrame
        sampled_data = pd.concat([sampled_data, sampled_rows])

    # Save the sampled data to a new CSV file
    sampled_data.to_csv(output_file_path, index=False)

def update_second_column(input_file_path, output_file_path):
    # Read the CSV file
    df = pd.read_csv(input_file_path, header=None, names=['Column1', 'Column2'])

    # Update the values in the second column to 2022
    df['Column2'] = 2023

    # Save the updated data to a new CSV file
    df.to_csv(output_file_path, index=False)
# Example usage:
input_file_path = './Year_Test/2023.csv'
output_file_path = './Year_Test/2023_300.csv'
extract_and_save_data(input_file_path, output_file_path)
# update_second_column(input_file_path, output_file_path)