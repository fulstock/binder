import json

with open("S:/HRCode/biencoder/binder/predict_predictions.json", "r") as f:
    with open("S:/HRCode/biencoder/binder/predict_predictions_utf8.json", "w", encoding = "utf-8") as out:
        for line in f:
            docdict = json.loads(line)
            # docdict["text"] = docdict["text"].decode('utf-8')
            print(json.dumps(docdict, ensure_ascii = False), file = out)