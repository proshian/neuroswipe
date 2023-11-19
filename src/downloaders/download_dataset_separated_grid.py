import gdown

if __name__ == "__main__":
    DATA_PATH = "data/data_separated_grid"
    
    url = "https://drive.google.com/drive/folders/1rRBUKUC0D6eZBJqT9qKs5fKQLl-gboej"
    
    gdown.download_folder(url,
                          output=DATA_PATH,
                          quiet=False,
                          use_cookies=False)