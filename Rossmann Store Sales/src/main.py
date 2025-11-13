import argparse
import data
import features

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset")
    args = parser.parse_args()
    dataset = args.dataset
    
    data.clean_data(dataset)
    features.get_features(dataset)

    if dataset == "train":
        import model
        model.train_lgbm()
    else:
        import predict
        predict.test_model()

if __name__ == "__main__":
    main()