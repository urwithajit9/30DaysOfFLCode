import pandas as pd
from datasets import load_dataset
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner

# Single file
data_files = "ClaMP_Integrated-5184.csv"

df = pd.read_csv(data_files)

# Drop the 'NoPacker' column from the DataFrame
df_features = df.drop(columns=['packer_type'])

# Assuming the last column is the label
features = df_features.iloc[:, :-1].values.tolist()  # All columns except the last
labels = df_features.iloc[:, -1].values.tolist()     # The last column

# Convert to the desired format
data = {
    "features": features,
    "labels": labels
}


# Multiple Files
#data_files = [ "path-to-my-file-1.csv", "path-to-my-file-2.csv", ...]
#dataset = load_dataset("csv", data_files=data_files)
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)


partitioner = IidPartitioner(num_partitions=10)
partitioner.dataset = dataset

for p_num in range(10):
    partition = partitioner.load_partition(partition_id=p_num)
    print(partition)


def load_data_csv(partition_id: int, num_partitions: int):
    """Load , partition and load clamp data."""
    # Only initialize `FederatedDataset` once
    # Single file
    data_files = "ClaMP_Integrated-5184.csv"

    df = pd.read_csv(data_files)

    # Drop the 'NoPacker' column from the DataFrame
    df_features = df.drop(columns=['packer_type'])

    # Assuming the last column is the label
    features = df_features.iloc[:, :-1].values.tolist()  # All columns except the last
    labels = df_features.iloc[:, -1].values.tolist()     # The last column

    # Convert to the desired format
    data = {
        "features": features,
        "labels": labels
    }


    # Multiple Files
    #data_files = [ "path-to-my-file-1.csv", "path-to-my-file-2.csv", ...]
    #dataset = load_dataset("csv", data_files=data_files)
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)    
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = dataset
        partition = partitioner.load_partition(partition_id=partition_id)


    # dataset = fds.load_partition(partition_id, "train").with_format("numpy")

    X, y = partition['features'], partition['labels']

    # Split the on edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

    return X_train, X_test, y_train, y_test    