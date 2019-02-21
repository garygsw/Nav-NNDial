import simplejson as json

with open('KvretnavOTGY.json') as data_file:
    data = json.load(data_file)

informables = data['informable']
print 'inf:', informables
requestables = data['requestable']
print 'req:', requestables

with open('/Users/garygsw/SUTD/Research/dialogue-self-play/baselines/sequicity/kvret_cleaned.json') as infile:
    cleaned_data = json.load(infile)
ALL_ADDRESS = set()
ALL_POI = set()
for i, dial in enumerate(cleaned_data):
    db = dial["scenario"]["kb"]["items"]
    for x in db:
        ALL_POI.add(x['poi'].lower())
        ALL_ADDRESS.add(x['address'].lower())
data['requestable']['address'] += list(ALL_ADDRESS)
data['other']['name'] += list(ALL_POI)


semidict = {}
for slot in data['informable']:
    semidict[slot] = [slot]
    for value in data['informable'][slot]:
        semidict[value] = [value.replace('_', ' ')]
for slot in data['requestable']:
    semidict[slot] = [slot]
    for value in data['requestable'][slot]:
        semidict[value] = [value.replace('_', ' ')]
for slot in data['other']:
    semidict[slot] = [slot]
    for value in data['other'][slot]:
        semidict[value] = [value.replace('_', ' ')]

# default
semidict["any"] = ["dont really care"]
#semidict["none"] = ["NONE"]
#semidict["exist"] = ["NONE"]
semidict['poi'] += ['name']


# add to requestable slot names [SLOT_REQ]
semidict["address"] += ["position", "directions", "details of the location", "details", "located", "where is it", "where is", "where's"]
semidict["traffic_info"] += ["route details", "the traffic", "any traffic", "how's the traffic", "is the traffic",
                             "there traffic", "show me traffic", "traffic like", "how is the traffic", "contain traffic",
                             "any traffic", "details about the traffic", "details of the traffic", "is there traffic",
                             "how does the traffic", "traffic nearby"]
semidict["distance"] += ["how far", "how far away", "how close", "how long"]

# add to requestable value names [VALUE_REQ]
semidict["1 mile"] += ["one mile", "1 miles"]
semidict["2 miles"] += ["two miles", "2 mile", "two mile"]
semidict["3 miles"] += ["three miles", "3 mile", "three mile"]
semidict["4 miles"] += ["four miles", "4 mile", "four mile"]
semidict["5 miles"] += ["five miles", "5 mile", "five mile"]
semidict["6 miles"] += ["six miles", "6 mile", "six mile"]
semidict["7 miles"] += ["seven miles", "7 mile", 'seven mile']
semidict["8 miles"] += ["eight miles", "8 mile", "eight mile"]
semidict["9 miles"] += ["nine miles", "9 mile", "nine mile"]
semidict["no traffic"] += ["isn't any traffic"]
semidict["jills house"] += ["jill's house", "jill's", "jill", "jill house"]
semidict["toms house"] += ["tom's house", "tom's", "tom", "tom house"]
semidict["jacks house"] += ["jack", "jacks", "jack house", "jack's house", "jack's"]
semidict["trader joes"] += ["trader joe's", "traders joes"]
semidict["hotel keen"] += ["keen"]
semidict["stanford oval parking"] += ["stanford oval"]
semidict["palo alto cafe"] += ["palo alto"]
semidict["willows market"] += ["willows", "willow's", "willow's market", "willow"]
semidict["the westin"] += ["westin", "westin hotel"]
semidict["stanford express care"] += ["stanford express"]
semidict["hacienda market"] += ["hacienda"]
semidict["panda express"] += ["panda"]
semidict["palo alto garage r"] += ["palo alto garage"]
semidict["valero"] += ["valeros"]
semidict["town and country"] += ["town and country shopping center"]
semidict["ravenswood shopping center"] += ["ravenswood"]

# add to informable values names [VALUE_INF]
semidict["nearest"] += ["within 5 miles", "within 4 miles", "within 3 miles", "within 2 miles", "within 1 mile",
                        "around me", "around here", "near me", "near my current location", "in my area", "in the area",
                        "close", "closer", "closest", "close by", "around my current location", "around this area",
                        "nearer", "nearest", "near", "near my location", "around", "near to me",
                        "nearby", "shortest", "shorter", "short", "in the area", "less time to reach"]
semidict["least traffic"] += ["best route", "best possible route", "best available route", "less traffic", "least traffic",
                              "fastest", "faster", "fast", "quickest", "quicker", "quick", "asap",
                              "avoid the traffic", "pronto", "quickly", "fastest route",
                              "no traffic", "no heavy traffic", "avoid heavy traffic", "avoids heavy traffic",
                              "avoid all heavy traffic", "avoids all heavy traffic", "avoid the heavy traffic",
                              "avoid all traffic", "avoids all traffic", "avoiding heavy traffic",
                              "without traffic", "without heavy traffic", "without any traffic",
                              "avoid any traffic", "avoid any heavy traffic", "avoiding all heavy traffic",
                              "avoid traffic", "avoids traffic", "avoid most traffic", "avoids most traffic",
                              "avoiding most traffic", "avoid most heavy traffic", "avoids most heavy traffic",
                              "avoiding most heavy traffic", "avoids as much heavy traffic",
                              "avoid the traffic", "avoids the traffic", "avoid all traffic",
                              "avoid the heavy traffic", "avoids the heavy traffic",
                              "avoid heavy traffic", "avoids heavy traffic", "without any heavy traffic",
                              "least amount of traffic", "around the traffic", "least heavy traffic",
                              "little traffic", "few traffic", "lighter traffic",
                              "doesn't have any traffic"]

semidict["pizza restaurant"] += ["pizza", "fast food", "place to get to eat", "pizza place", "pizza places", "pizza shop", "pizza spot",
                                 "fast food restaurant", "fast food place", "fast food restaurants", "something to eat", "pizzeria", "pizza joint"]
semidict["chinese restaurant"] += ["chinese restaurants", "chinese", "chinese food", "restaurant", "restaurants", "chinese place", "chinese places",
                                   "chinese food restaurant", "chinese food place", "chinese takeout", "place to eat", "chinese spot", "go to eat", "get to eat",
                                   "get lunch", "chinese joint"]
semidict["coffee"] += ["coffee shop", "coffee shops", "barista", "coffee place", "cafe", "coffee house"]
semidict["tea"] += ["tea shop", "tea shops", "tea place", "tea house"]
semidict["home"] += ["my house", "my address", "i live", "the house"]
semidict["parking garage"] += ["parking spot", "parking", "parking lot", "parking lots", "garage", "place to park", "parking area",
                               "somewhere to park", "parking garages", "where i can park", "parking areas", "park"]
semidict["grocery store"] += ["groceries", "grocery stores", "grocery market", "produce stand", "grocery shop", "grocery shops"]
semidict["friend house"] += ["friend", "friends", "friend's", "friendss'", "friends' house", "friend's house", "friend's home", "friends house", "friend's home"]
semidict["rest stop"] += ["rest stops", "motel", "motels", "hotel", "hotels", "place to sleep"]
semidict["gas station"] += ["fill up", "gas", "gas stations"]
semidict["shopping center"] += ["shopping centers", "shopping mall", "mall", "malls", "place to shop", "shopping"]
semidict["hospital"] += ["hospitals"]


with open('Kvretnav_SemiDict.json', 'w') as data_file:
    json.dump(semidict, data_file, indent=4)
