import json
import os
from tqdm.auto import tqdm

from nltk.data import load
from nltk.tokenize import word_tokenize

from collections import Counter

train_dataset_path = "S:/HRCode/data/nerel-bio-v1.0/train"

ru_tokenizer = load("tokenizers/punkt/russian.pickle") # Загрузка токенизатора для русского языка

dataset_name = "nerel-bio-fix"

all_entities = []

tags = ['ACTIVITY', 'ADMINISTRATION_ROUTE', 'AGE', 'ANATOMY', 'CHEM', 'CITY', 'COUNTRY', 'DATE', 'DEVICE', 'DISO', 'DISTRICT', 'EVENT', 'FACILITY', 'FAMILY', 'FINDING', 'FOOD', 'GENE', 'HEALTH_CARE_ACTIVITY', 'INJURY_POISONING', 'LABPROC', 'LIVB', 'LOCATION', 'MEDPROC', 'MENTALPROC', 'MONEY', 'NATIONALITY', 'NUMBER', 'ORDINAL', 'ORGANIZATION', 'PERCENT', 'PERSON', 'PHYS', 'PRODUCT', 'PROFESSION', 'SCIPROC', 'STATE_OR_PROVINCE', 'TIME']

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
                                                    "txtdata" : txtdata,
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
    tag_to_spans[entity["tag"]].append(entity)

for tag, entities in tag_to_spans.items():
    ent_cont = []
    for entity in entities:
        txtdata = entity["txtdata"]
        sentence_spans = ru_tokenizer.span_tokenize(txtdata)
        for span in sentence_spans:
            start, end = span
            context = txtdata[start : end]
            if entity["span"] in context and entity["start"] >= start and entity["end"] <= end:
                # context = context[ : entity["start"] - start] + tag + context[entity["end"] - end : ]
                ent_cont.append((context, entity, start, end))
    tag_to_spans[tag] = ent_cont
    all_contexts = list(set([v[0] for v in tag_to_spans[tag]]))
    new_contexts = {}
    for context in all_contexts:
        context_entities = [(v[1], v[2], v[3]) for v in tag_to_spans[tag] if v[0] == context]
        context_entities = [(v[0], v[0]["start"] - v[1], v[0]["end"] - v[1]) for v in context_entities]

        outermost_entities = [v for v in context_entities if \
            not any([(s[1], s[2]) for s in context_entities if v is not s and v[1] >= s[1] and v[2] <= s[2]])]
        outermost_entities = sorted(outermost_entities, key = lambda x: x[1])

        lex_context = ""
        rcontext = context
        offset = 0

        for entity in outermost_entities:
            s, e = entity[1], entity[2]
            s += offset
            e += offset
            lex_context += rcontext[ : s] + tag
            rcontext = context[e - offset : ]
            offset = len(rcontext) - len(context)

        lex_context += rcontext
        
        new_contexts[context] = lex_context

    tag_to_spans[tag] = [(new_contexts[t[0]], t[1], t[2], t[3]) for t in tag_to_spans[tag]]

    span_count = [v[1]["span"] for v in ent_cont]
    span_count = Counter(span_count)
    tag_to_spans[tag] = [(v[0], v[1], span_count[v[1]["span"]], v[2], v[3]) for v in tag_to_spans[tag]]
    tag_to_spans[tag] = sorted(tag_to_spans[tag], key = lambda x : x[2], reverse = True)

    lex_context = tag_to_spans[tag][0][0]
    tag_to_spans[tag] = lex_context

with open("nerel-bio/entity_types_bio-lao.json", "w", encoding = "UTF-8") as etfile:
    for tag in tags:
        etfile.write(json.dumps({
                "dataset" : dataset_name, 
                "name" : tag, 
                "description" : tag_to_spans[tag], 
                "description_source" : "Full Lexical Outermost Prompt"
            }, ensure_ascii = False) + "\n"
        )