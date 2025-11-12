import os
import json

for root, dirs, files in os.walk('./nerel-parted2-neutral'):
    for file in files:
        if file == "final.json":
            # orig_root = root.replace('nerel-parted2-neutral', 'nerel-parted2')
            orig_root = root
            with open(os.path.join(orig_root, file), "r", encoding = "UTF-8") as conffile:
                config = json.load(conffile)

            # run_name = config["run_name"]
            # run_name = run_name.replace("lao", "lao-neutral") 
            # config["run_name"] = run_name

            # train_file = config["train_file"]
            # train_file = train_file.replace("outet", "outer")
            # config["train_file"] = train_file
            # validation_file = config["validation_file"]
            # validation_file = validation_file.replace("outet", "outer")
            # config["validation_file"] = validation_file
            # test_file = config["test_file"]
            # test_file = test_file.replace("outet", "outer")
            # config["test_file"] = test_file

            train_file = config["train_file"]
            train_file = train_file.replace("final", "neutral-final")
            config["train_file"] = train_file
            validation_file = config["validation_file"]
            validation_file = validation_file.replace("final", "neutral-final")
            config["validation_file"] = validation_file
            test_file = config["test_file"]
            test_file = test_file.replace("final", "neutral-final")
            config["test_file"] = test_file

            # output_dir = config["output_dir"]
            # output_dir = output_dir.replace("lao", "neutral/lao") 
            # config["output_dir"] = output_dir

            # config["do_neutral_spans"] = True

            with open(os.path.join(root, file), "w", encoding = "UTF-8") as conffile:
                json.dump(config, conffile, indent = 4, ensure_ascii = False)
