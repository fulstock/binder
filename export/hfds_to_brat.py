import json
import os
from tqdm.auto import tqdm

import argparse

parser = argparse.ArgumentParser(description = "HFDS-json to brat formatter script.")
parser.add_argument('--output_path', type = str, required = True, help = "Path to output brat dataset (with train, dev, test dirs).")
parser.add_argument('--pred_file', type = str, required = True, help = "Path to predict_predictions.json file, with binder predictions.")
parser.add_argument('--orig_path', type = str, required = True, help = "Path to original dataset in brat format.")
parser.add_argument('--tags_file', type = str, required = True, help = "Path to tags file.")

args = parser.parse_args()

outdir = args.output_path
hfds_inf_file = args.pred_file
origbratdir = args.orig_path
tagsfile = args.tags_file

os.makedirs(outdir, exist_ok=True)

entities = []
entity_count = {}

with open(hfds_inf_file, "r", encoding = "UTF-8") as hfdsfile:
    with open(os.path.join(outdir, "conv_preds.json"), "w", encoding = "utf-8", newline = "") as out:
        for line in hfdsfile:
            docdict = json.loads(line)
            print(json.dumps(docdict, ensure_ascii = False), file = out)

with open(os.path.join(outdir, "conv_preds.json"), "r", encoding = "utf-8", newline = "") as converted:
    for line in converted:
        entities.append(json.loads(line))

with open(tagsfile, "r") as tagsfile:
    tags = json.load(tagsfile)

for tag in tags:
    entity_count[tag] = 0

for f_idx, file_entities in enumerate(tqdm(entities)):

    filename = file_entities["id"]

    with open(os.path.join(origbratdir, filename + ".txt"), "r", encoding = "UTF-8", newline = "") as tf:
        text = tf.read()
        
    with open(os.path.join(outdir, filename + ".txt"), "w", encoding = "UTF-8", newline = "") as textfile:
        textfile.write(text)

    with open(os.path.join(outdir, filename + ".ann"), "w", encoding = "UTF-8", newline = "") as annfile:

        outputs = []

        for ent in file_entities["pred_ner"]:
            outputs.append((ent[2], ent[0], ent[1], ent[3]))
            entity_count[ent[2]] += 1

        for e_idx, (tag, first_char, last_char, entity) in enumerate(sorted(outputs, key = lambda x : x[1])):
            assert entity == text[first_char : last_char]
            annfile.write("T" + str(e_idx + 1) + "\t" + tag + " " + str(first_char) + " " + str(last_char) + "\t" + entity + "\n")  