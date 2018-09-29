import simplejson as json

with open('MTNavOTGY.json') as data_file:
    #for i in range(5):
    #    data_file.readline()
    data = json.load(data_file)

informables = data['informable']
requestables = data['requestable']
#place_search_keyword = data['informable']['place_search_keyword']
#pleace_search_ratings = data['informable']['place_s']

semidict = {}
for slot in data['informable']:
    semidict[slot] = [slot]
    for x in data['informable'][slot]:
        semidict[x] = [x.replace('_', ' ')]
for x in data['requestable']:
    semidict[x] = [x.replace('_', ' ')]

# copy
semidict["any"] = ["no specific","no preference","dont really care","do not care","dont care","does not matter","any"]
semidict["none"] = ["NONE"]
semidict["exist"] = ["NONE"]
semidict["place"] = ["place", "location"]
semidict["waypoints"].append("current list of destinations")
semidict["duration"].append("travel time")

semidict["place_ratings"] += ["ratings", "stars"]
semidict["place_address"].append("address")
semidict["open_now"].append("opening hours")


with open('MTNavSemiDict.json', 'w') as data_file:
    json.dump(semidict, data_file, indent=4)
