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
    df['Essentiality'] = df['Name'].apply(lambda x: 1 if x in essential_proteins else 0)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Updated file saved as {output_file}")

# Example usage
csv_file = 'data/features/mips_node_embeddings_256.csv'           # Input CSV file path
essential_file = 'data/essential.txt'    # Essential proteins text file path
output_file = 'data/features/mips_node_embeddings_256_label.csv' # Output CSV file path

append_essentiality_column(csv_file, essential_file, output_file)
