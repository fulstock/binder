import os
import json
from tqdm.auto import tqdm
from nltk.data import load
from nltk.tokenize import word_tokenize
import pymorphy2

from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")

from collections import Counter

morph = pymorphy2.MorphAnalyzer()
ru_tokenizer = load("tokenizers/punkt/russian.pickle") # Загрузка токенизатора для русского языка

train_dataset_path = "S:/HRCode/data/NEREL/train"

dataset_name = "nerel"

most_common = 5

tags = ["AGE", "AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISEASE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PERCENT", "PERSON", "PENALTY", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"]

all_entities = []

for ad, dirs, files in os.walk(train_dataset_path):
    for f in tqdm(files):

        if f[-4:] == '.ann':
            try:

                if os.stat(train_dataset_path + '/' + f).st_size == 0:
                    continue

                annfile = open(train_dataset_path + '/' + f, "r", encoding='UTF-8')
                txtfile = open(train_dataset_path + '/' + f[:-4] + ".txt", "r", encoding='UTF-8')

                txtdata = txtfile.read()
                # txtdata = txtdata.replace('\n', '.', 1) # Отделение заголовков

                # Шаг 1. Считать все именованные сущности из файла, закрыть файл.

                file_entities = []

                # Именованная сущность пока что будет представленна укороченной записью. Позже она будет приведена к выду выше.

                for line in annfile:
                    line_tokens = line.split()
                    if len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'T':
                        if line_tokens[1] in tags:
                            try:
                                file_entities.append( { 
                                                    "tag" : line_tokens[1], 
                                                    "start" : int(line_tokens[2]),
                                                    "end" : int(line_tokens[3]),
                                                    "span" : txtdata[int(line_tokens[2]) : int(line_tokens[3])]
                                                    } )
                            except ValueError:
                                pass # Все неподходящие сущности

                annfile.close()

                all_entities.extend(file_entities)

            except FileNotFoundError:
                pass

tag_to_spans = {tag : [] for tag in tags}
for entity in all_entities:
    tag_to_spans[entity["tag"]].extend(word_tokenize(entity["span"]))

for tag, entities in tag_to_spans.items():
    tag_to_spans[tag] = [morph.parse(word)[0].normal_form for word in entities if word not in russian_stopwords]
    tag_to_spans[tag] = Counter(tag_to_spans[tag]).most_common(most_common)
    tag_to_spans[tag] = [w for w, n in sorted(tag_to_spans[tag], key = lambda x : x[1], reverse = True)]

with open("nerel/entity_types_cmn-mfc5.json", "w", encoding = "UTF-8") as etfile:
    for tag in tags:
        etfile.write(json.dumps({
                "dataset" : dataset_name, 
                "name" : tag, 
                "description" : f"{tag} - это сущности, такие как " + ", ".join(tag_to_spans[tag]) + ".", 
                "description_source" : f"{most_common}-most Frequent Components Prompt"
            }, ensure_ascii = False) + "\n"
        )