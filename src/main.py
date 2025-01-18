
from read import json_to_csv
from preprocesing import create_vectorstore



def main():
    api_url = 'https://magicloops.dev/api/loop/f35fe175-2e71-4fad-81be-7a6b3a9aa4dc/run'
    base_filename = './data/magic_loops_data.csv'
    json_to_csv(api_url, base_filename)
    create_vectorstore()

 
if __name__ == "__main__":
    main()