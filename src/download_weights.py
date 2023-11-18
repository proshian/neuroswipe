import gdown

if __name__ == "__main__":
    WEIGHTS_DIR = "data/trained_models_for_final_submit"
    
    url = "https://drive.google.com/drive/folders/1-iFPYCcRYy-tEu14Ry6xU6SMMf3eCjn6"
    
    gdown.download_folder(url,
                          output=WEIGHTS_DIR,
                          quiet=False,
                          use_cookies=False)