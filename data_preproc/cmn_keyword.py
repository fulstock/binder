import json
import os

dataset_name = "nerel"

tags = ["AGE", "AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISEASE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PERCENT", "PERSON", "PENALTY", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"]

with open("nerel/entity_types_cmn-kw.json", "w", encoding = "UTF-8") as etfile:
    for tag in tags:
        etfile.write(json.dumps({"dataset" : dataset_name, "name" : tag, "description" : tag, "description_source" : "Keyword Prompt"}) + "\n")

