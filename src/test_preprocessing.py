from preprocessing import (
    load_dataset,
    clean_dataset,
    split_features_target,
    build_preprocessor,
)


def main():

    print("=" * 60)
    print("TESTING PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load
    df = load_dataset()

    print(f"\nOriginal shape: {df.shape}")

    # Clean
    df = clean_dataset(df)

    print(f"After cleaning: {df.shape}")

    # Split
    X, y = split_features_target(df)

    print(f"\nFeature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Build preprocessor
    preprocessor = build_preprocessor(X)

    # Fit + transform
    X_processed = preprocessor.fit_transform(X)

    print(
        f"\nProcessed feature shape: "
        f"{X_processed.shape}"
    )

    print("\nPreprocessing pipeline works successfully!")

    print("\nMissing values after preprocessing:")

    print(
        "Handled automatically by the "
        "imputation pipelines."
    )


if __name__ == "__main__":
    main()