import json
import pprint
json_file = "/bigdata/chenda/v1_0_train_nus.json"
with open(json_file, 'r') as file:
    json_content = json.load(file)
firstkey = list(json_content)[0]
firstval = json_content[firstkey]
print(firstval)
pprint.pprint(firstval, indent=4)

