from sklearn.datasets import fetch_20newsgroups


def load_dataset():

    print("Starting dataset download...")

    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )

    print("Dataset loaded!")

    documents = data.data
    labels = data.target
    categories = data.target_names

    print("Total documents:", len(documents))
    print("Categories:", categories)

    return documents, labels, categories


if __name__ == "__main__":

    docs, labels, categories = load_dataset()