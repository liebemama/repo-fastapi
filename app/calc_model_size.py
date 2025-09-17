def model_size(in_features: int, hidden: int, out_features: int, dtype_bytes: int = 4) -> dict:
    """
    Calculate the number of weights and memory size required for a simple MLP model.

    Args:
        in_features (int): Number of input features.
        hidden (int): Number of units in the hidden layer.
        out_features (int): Number of output features.
        dtype_bytes (int, optional): Number of bytes per weight. Default is 4 (float32).

    Returns:
        dict: A dictionary containing model dimensions, total weights,
              and memory usage in MB and GB.
    """
    weights = in_features * hidden + hidden * out_features
    memory_bytes = weights * dtype_bytes

    return {
        "in_features": in_features,
        "hidden": hidden,
        "out_features": out_features,
        "total_weights": weights,
        "memory_MB": round(memory_bytes / (1024**2), 2),
        "memory_GB": round(memory_bytes / (1024**3), 2),
    }


if __name__ == "__main__":
    print("Model Size Calculator (MLP)")

    try:
        in_f = int(input("Enter number of input features (in_features): "))
        hidden = int(input("Enter number of hidden units (hidden): "))
        out_f = int(input("Enter number of output features (out_features): "))

        result = model_size(in_f, hidden, out_f)
        print("\nResult:")
        for key, value in result.items():
            print(f"{key:15}: {value}")
    except Exception as e:
        print("Error:", e)
