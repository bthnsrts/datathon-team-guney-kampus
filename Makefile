# Ensure data directory exists
data:
	mkdir -p data

# Download, extract and clean up Kaggle dataset
download_kaggle_dataset: data
	kaggle competitions download -c ing-hubs-turkiye-datathon
	unzip ing-hubs-turkiye-datathon.zip -d data/
	rm ing-hubs-turkiye-datathon.zip

.PHONY: download_kaggle_dataset data
