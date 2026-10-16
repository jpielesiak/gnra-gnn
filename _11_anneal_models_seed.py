import os
gat = __import__('08_gat')

def getSeed(folder_path="folder1", file_name="seed.txt"):
    file_path = os.path.join(folder_path, file_name)

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # If seed.txt doesn't exist, create it with -1
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("-1")

    # Read current seed
    with open(file_path, "r") as f:
        content = f.read().strip()
        try:
            seed = int(content)
        except ValueError:
            seed = -1  # fallback if file is corrupted

    # Increment seed
    seed += 1

    # Write new seed back
    with open(file_path, "w") as f:
        f.write(str(seed))
    return seed
if __name__ == '__main__':


    for i in range(100):  # Run 5 times to get different seeds
        seed = getSeed()
        print(f"Current seed: {seed}")
        gat.setSeed(seed, "test_models_annealing",-9)
    print("Seed set to 0 for reproducibility.")