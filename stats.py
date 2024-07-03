def count_essential_proteins(essential_file, ppi_file):
    # Load essential protein names into a set for uniqueness
    with open(essential_file, 'r') as f:
        essential_proteins = set(line.strip() for line in f.readlines())

    # Initialize a set to track unique essential proteins found in PPI file
    unique_essential_proteins = set()

    # Count unique essential proteins in PPI file
    with open(ppi_file, 'r') as f:
        for line in f:
            protein1, protein2 = line.strip().split()
            if protein1 in essential_proteins:
                unique_essential_proteins.add(protein1)
            if protein2 in essential_proteins:
                unique_essential_proteins.add(protein2)

    return len(unique_essential_proteins)


# Example usage:
essential_file = 'data/essential.txt'

count = count_essential_proteins(essential_file, 'data/ppi/dip.txt')
print(f"Unique essential proteins in DIP: {count}")

count = count_essential_proteins(essential_file, 'data/ppi/biogrid.txt')
print(f"Unique essential proteins in BioGrid: {count}")

count = count_essential_proteins(essential_file, 'data/ppi/mips.txt')
print(f"Unique essential proteins in MIPS: {count}")
