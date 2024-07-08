import os
import pandas as pd


def load_essential_proteins(file_path):
    """Load the list of essential proteins from a text file."""
    with open(file_path, 'r') as file:
        essential_proteins = {line.strip() for line in file}
    return essential_proteins


def append_essentiality_column(csv_file, essential_file, output_file):
    """Read a CSV file, append a column "essentiality" and save the updated file."""
    # Load the essential proteins
    essential_proteins = load_essential_proteins(essential_file)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Check if "Name" column exists in the DataFrame
    if 'Name' not in df.columns:
        raise ValueError("CSV file must contain a 'Name' column")

    # Create the "essentiality" column
    df['Essentiality'] = df['Name'].apply(
        lambda x: 1 if x in essential_proteins else 0)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Updated file saved as {output_file}")


# Load the CSV files

GO_file = 'data/GO.csv'
node_embeddings_file = 'data/features/dip_node_embeddings_256.csv'
feature_file = 'data/features/dip_features.csv'
essential_protein_file = 'data/essential.txt'

if not os.path.exists(node_embeddings_file):
    raise FileNotFoundError(
        f"File {node_embeddings_file} does not exist. \n✔ Please config and run NodeEmbeddings.py to generate Node Embeddings")
# endif

df1 = pd.read_csv(GO_file)
df2 = pd.read_csv(node_embeddings_file)

# Merge the DataFrames on the 'name' column, keeping only rows with matching 'name' values
merged_df = pd.merge(df1, df2, on='Name', how='right')
merged_df = merged_df.fillna(0)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(feature_file, index=False)

# print('Merged DataFrame:')
# print(merged_df.head())

# add essentiality colum inplace
append_essentiality_column(feature_file, essential_protein_file, feature_file)
