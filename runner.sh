set -e
python data_extractor.py -d ../hyperpartisan_data/by_articles/train/ -t ../hyperpartisan_data/by_articles/truth/ -o data/
python save_model.py
python new_model.py -d ../hyperpartisan_data/by_articles/train/ -o output/
python evaluator.py -d ../hyperpartisan_data/by_articles/truth/ -r output/ -o evaluated/
