import versioning

version = "v1.0.0"

def load_data():
    data, metadata = versioning.load_data(version)
    train, val = data['train_dataset'], data['val_dataset']
    # do data augumentation here
    return train, val