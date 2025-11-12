import json
import os
from tqdm.auto import tqdm

outdir = "S:/HRCode/cyberattacks/data/sec_events_smallinf_binderinf_brat"
hfds_inf_file = "S:/HRCode/biencoder/binder/predict_predictions_utf8.json"
origbratdir = "S:/HRCode/cyberattacks/data/sec_events_smallinf"

entities = []
entity_count = {
    "ARG_CVE" : 0,
    "ARG_CAPABILITIES" : 0,
    "ARG_DATA" : 0,
    "ARG_DEVICE" : 0,
    "ARG_FILE" : 0,
    "ARG_GPE" : 0,
    "ARG_MALWARE" : 0,
    "ARG_MONEY" : 0,
    "ARG_NUMBER" : 0,
    "ARG_ORGANIZATION" : 0,
    "ARG_PII" : 0,
    "ARG_PATCH" : 0,
    "ARG_PAYMENTMETHOD" : 0,
    "ARG_PERSON" : 0,
    "ARG_PURPOSE" : 0,
    "ARG_SOFTWARE" : 0,
    "ARG_SYSTEM" : 0,
    "ARG_TIME" : 0,
    "ARG_VERSION" : 0,
    "ARG_VULNERABILITY" : 0,
    "ARG_WEBSITE" : 0,
    "NUG_DISCOVERVULNERABILITY" : 0,
    "NUG_PHISHING" : 0,
    "NUG_DATABREACH" : 0,
    "NUG_RANSOM" : 0,
    "NUG_PATCHVULNERABILITY" : 0
}

with open(hfds_inf_file, "r", encoding = "UTF-8") as hfdsfile:
    for line in hfdsfile:
        entities.append(json.loads(line))

for file_entities in tqdm(entities):

    filename = file_entities["id"]

    with open(os.path.join(origbratdir, filename + ".txt"), "r", encoding = "cp1251") as tf:
        text = tf.read()
        
    with open(os.path.join(outdir, filename + ".txt"), "w", encoding = "UTF-8") as textfile:
        print(text, file = textfile)

    with open(os.path.join(outdir, filename + ".ann"), "w", encoding = "UTF-8") as annfile:

        outputs = []

        for ent in file_entities["pred_ner"]:
            outputs.append((ent[2], ent[0], ent[1]))
            entity_count[ent[2]] += 1

        for e_idx, (tag, first_char, last_char) in enumerate(sorted(outputs, key = lambda x : x[1])):
            annfile.write("T" + str(e_idx + 1) + "\t" + tag + " " + str(first_char) + " " + str(last_char) + "\t" + text[first_char : last_char] + "\n")     

entity_count = list([(k, v) for k, v in entity_count.items()])
print("\n".join([str(e[1]) for e in entity_count]))