f = open('S:/HRCode/relations/other/test/raw_wiki_page.txt', 'r', encoding = 'UTF-8', newline = "")
t = f.read()
f.close()

f = open('S:/HRCode/relations/other/test/raw_wiki_page.ann', 'w', encoding = 'UTF-8', newline = "")
e = open('S:/HRCode/relations/other/test/entities.json', 'r', encoding = 'UTF-8')
import json
entities = json.load(e)

offset = 0
for e_idx, (first_char, last_char, tag, entity_span) in enumerate(sorted(entities, key = lambda x : x[0])):
    while entity_span != t[first_char + offset: last_char + offset]:
        offset += 1
    f.write('T' + str(e_idx + 1) + '\t' + tag + ' ' + str(first_char + offset) + ' ' + str(last_char + offset) + '\t' + entity_span + '\n')

f.close()
e.close()