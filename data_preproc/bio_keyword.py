import json
import os

dataset_name = "nerel-bio-fix"

tags = ['ACTIVITY', 'ADMINISTRATION_ROUTE', 'AGE', 'ANATOMY', 'CHEM', 'CITY', 'COUNTRY', 'DATE', 'DEVICE', 'DISO', 'DISTRICT', 'EVENT', 'FACILITY', 'FAMILY', 'FINDING', 'FOOD', 'GENE', 'HEALTH_CARE_ACTIVITY', 'INJURY_POISONING', 'LABPROC', 'LIVB', 'LOCATION', 'MEDPROC', 'MENTALPROC', 'MONEY', 'NATIONALITY', 'NUMBER', 'ORDINAL', 'ORGANIZATION', 'PERCENT', 'PERSON', 'PHYS', 'PRODUCT', 'PROFESSION', 'SCIPROC', 'STATE_OR_PROVINCE', 'TIME']

with open("entity_types_bio-kw.json", "w", encoding = "UTF-8") as etfile:
    for tag in tags:
        etfile.write(json.dumps({"dataset" : dataset_name, "name" : tag, "description" : tag, "description_source" : "Keyword Prompt"}) + "\n")

