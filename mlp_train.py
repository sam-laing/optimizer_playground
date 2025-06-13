from datasets import MnistDataset  

dataset = MnistDataset(root="./data")


if __name__ == "__main__":
    from models.simpleMLP import SimpleMLP
    from datasets.make_loaders import make_loaders

    train_loader, val_loader = make_loaders(dataset, batch_size=64, seed=42, val_size=0.2)

    model = SimpleMLP(
        dim_input=28*28, dim_output=10, hidden_layers=[128, 64], activation='relu', dropout=0.15
        )

    print(model)

    # Example of training loop (not implemented here)
    # for epoch in range(num_epochs):
    #     for batch in train_loader:
    #         # Training code here
    for X, Y in train_loader:
        print(f"Batch X shape: {X.shape}, Y shape: {Y.shape}")
        print(f"Batch X type: {type(X)}, Y type: {type(Y)}")
        break 