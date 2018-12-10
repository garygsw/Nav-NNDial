import simplejson as json

with open('CamRestOTGY.json') as data_file:
    for i in range(5):
        data_file.readline()
    data = json.load(data_file)

print len(data['informable']['food'])  # 91 food types
print len(data['informable']['area'])  # 5 area types
print len(data['informable']['pricerange'])  # 3 price range  ==> 99
print len(data['requestable'])  # 3
informables = data['informable']
food_types = data['informable']['food']
area_types = data['informable']['area']
price_types = data['informable']['pricerange']
requestables = data['requestable']

with open('CamRestHDCSemiDict.json') as data_file:
    for i in range(5):
        data_file.readline()
    data = json.load(data_file)

for x in data:
    if x not in informables and x not in food_types and x not in area_types and x not in price_types and x not in requestables:
        print x

# EXTRAS:
# name
# none
# exist
# any
