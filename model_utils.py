import pickle


def save_genome(genome, path):
    """Save a NEAT genome to a file (e.g. best_genome.pkl)."""
    with open(path, 'wb') as f:
        pickle.dump(genome, f)
    print(f"Saved genome to {path}")


def load_genome(path):
    """Load a NEAT genome from a file."""
    with open(path, 'rb') as f:
        genome = pickle.load(f)
    return genome
