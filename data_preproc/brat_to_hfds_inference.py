import json
import os
import argparse

from nltk.data import load
from nltk.tokenize import NLTKWordTokenizer

from tqdm.auto import tqdm

# ## Закомментируйте после первой загрузки пакета. 
import nltk
nltk.download('punkt')
# ##

ru_tokenizer = load("tokenizers/punkt/russian.pickle") # Загрузка токенизатора для русского языка
word_tokenizer = NLTKWordTokenizer()

brat2mrc_parser = argparse.ArgumentParser(description = "Brat to hfds-json formatter script.")
brat2mrc_parser.add_argument('--brat_dataset_path', type = str, required = True, help = "Path to brat dataset (with train, dev, test dirs).")
brat2mrc_parser.add_argument('--tags_path', type = str, required = True, help = 'Path to tags file with format ["CLASS1", "CLASS2", ...].')
brat2mrc_parser.add_argument('--hfds_output_path', type = str, default = None, help = "Path, where formatted dataset would be stored. By default, same path as in --brat_dataset_path would be used.")

args = brat2mrc_parser.parse_args()

brat_dataset_path = args.brat_dataset_path

hfds_output_path = args.hfds_output_path
if hfds_output_path is None:
    hfds_output_path = brat_dataset_path

tags_path = args.tags_path

with open(tags_path, "r") as tags_file:
    tags = json.loads(tags_file.read())

print(tags)

jsonpath = os.path.join(hfds_output_path, "test.json")
dataset_path = brat_dataset_path

jsondir = os.path.dirname(jsonpath)

if not os.path.exists(jsondir):
    os.makedirs(jsondir)

jsonfile = open(jsonpath, "w", encoding='UTF-8') 

doc_count = 0
doc_ids = []

for ad, dirs, files in os.walk(dataset_path):
    for f in tqdm(files):

        if f[-4:] == '.txt':
            try:

                if os.stat(dataset_path + '/' + f).st_size == 0:
                    continue

                txtfile = open(dataset_path + '/' + f, "r", encoding='CP1251')
                txtdata = txtfile.read()
                txtfile.close()

                offset_mapping = []

                sentence_spans = ru_tokenizer.span_tokenize(txtdata)
                
                for span in sentence_spans:

                    start, end = span
                    context = txtdata[start : end]

                    word_spans = word_tokenizer.span_tokenize(context)
                    offset_mapping.extend([(s + start, e + start) for s, e in word_spans])

                start_words, end_words = zip(*offset_mapping)

                doc_entities = {
                    'text': txtdata,
                    'entity_types': [],
                    'entity_start_chars': [],
                    'entity_end_chars': [],
                    'id': f[:-4],
                    'word_start_chars': start_words,
                    'word_end_chars': end_words
                }

                # Шаг 4. Сохранить все сущности в hfds формат.

                doc_count += 1
                doc_ids.append(f[:-4])

                jsonfile.write(json.dumps(doc_entities, ensure_ascii = False) + '\n')

            except FileNotFoundError:
                pass

print(f"Docs: {doc_count}")

jsonfile.close()