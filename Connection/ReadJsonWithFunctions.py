import json
import numpy as np
from geojson import Point, Feature, FeatureCollection, dump

filename = 'json_12_2.txt'
featuresLC = []
featuresProb = []
LandCover = str()
#state = input('NC or TX?')  #The other idea is to check the lenght of the prob
    
with open(filename, 'r') as infile:
    data = infile.read()
    new_data = data.replace('}{', '},{')
    json_data = json.loads(f'[{new_data}]')


for i in json_data[0]:
    new_dict = {'coordinates' : [(i['Inputs']['data'][0]['x']), (i['Inputs']['data'][0]['y'])]}
    i['Inputs']['data'][0].update(new_dict) #create new key called "coordinates" in the "input"
    #print((i['Inputs']['data'][0]))
    #coorpoint= ((i['Inputs']['data'][0]['x']), (i['Inputs']['data'][0]['y']))
    #print(coorpoint)
    Probability = np.max(i['Outputs']['Results'])
    arr = np.array(i['Outputs']['Results'])
    #print(np.where(arr == np.amax(i['Outputs']['Results']))[1] )
    LCIndex = np.where(arr == np.amax(i['Outputs']['Results']))[1]
    LCIndexInt = int(LCIndex[0])
    #print(LCIndexInt)
    #print(type(LCIndexInt))
    #Land cover for TX: 0,1,12,2,3,4,5,6 (for NC, there is no LC 0)
    def NCLC():
        global LandCover
        if LCIndexInt ==1:
            LandCover = str(12)        
        elif LCIndexInt < 1:
            LandCover = str(LCIndexInt)-1
        else:
            LandCover = str(LCIndexInt)
    #print(LandCover)    
    def TXLC(): #LandCover for TX
        global LandCover
        if LCIndexInt == 2:
            LandCover = str(12)
        elif LCIndexInt < 2:
            LandCover = str(LCIndexInt)
        else:
            LandCover = str(LCIndexInt-1)
    if arr.size == 8:
        state = 'TX'
    elif arr.size == 7:
        state = 'NC'
    else:
        print("Check number of LandCover Classes")
            
        
    #print(LandCoverTX)
    if state == 'NC':
        NCLC()
    elif state == 'TX':
        TXLC()
    else:
        print('Invalid State')

    print(LandCover)
    #add the land cover and the associate prob (in 2dp) as new item in dict.
    new_dict_out = {'LandCover' : LandCover, 'Probability' : round(Probability, 2)}
    i['Outputs'].update(new_dict_out)
    #print(i['Outputs'])
    #Creating a geojson
    coorpoint= Point(((i['Inputs']['data'][0]['x']), (i['Inputs']['data'][0]['y'])))
    #print(coorpoint)
    featuresLC.append(Feature(geometry=coorpoint, properties={'LandCover' : LandCover}))
    #print(features)
    featuresProb.append(Feature(geometry=coorpoint, properties={'Probability' : round(Probability, 2)}))

feature_collectionLC = FeatureCollection(featuresLC)
print(feature_collectionLC)
feature_collectionProb = FeatureCollection(featuresProb)
#print(feature_collectionProb)
#create a geojson
with open('LandCover2.geojson', 'a') as f:
   dump(feature_collectionLC, f)
   f.write('\n')

with open('LandCoverProb2.geojson', 'a') as f:
   dump(feature_collectionProb, f)
   f.write('\n')
    
    
    


    
    
    
