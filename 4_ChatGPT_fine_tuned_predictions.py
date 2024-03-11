#pip3 install openai

# Importing necessary packages.
import sys
import os
import zipfile as zip


import pandas as pd
import csv
import numpy as np
from openai import OpenAI

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)

with open('Config.txt') as f:

    lines = f.readlines()
    openAI_key = lines[0].strip()
    print (openAI_key)

client = OpenAI(api_key=openAI_key)

Predicting_Patents_sample_size = 100

Desktop_Developing = True

if Desktop_Developing:
    Y_category ='Y04S'   ### ONly accept 1 element
    df_Current_Y_category_all_address = 'ResultsServer/Step2/'+Y_category+ '_ML_Candidates.csv'

else:
    Y_category = sys.argv[1]
    df_Current_Y_category_all_address = 'Results/Step2/'+Y_category+ '_ML_Candidates.csv'


### A dictionary mapping Y02 category to the abreiviation of the category.

Y02_category_dict = {'Y02A': 'ADAPTATION TO CLIMATE CHANGE',
                    'Y02B': 'BUILDINGS',
                    'Y02C': 'CAPTURE, STORAGE, SEQUESTRATION OR DISPOSAL OF GREENHOUSE GASES',
                    'Y02D': 'INFORMATION AND COMMUNICATION TECHNOLOGIES [ICT]',
                    'Y02E': 'ENERGY GENERATION, TRANSMISSION, DISTRIBUTION',
                    'Y02P': 'PRODUCTION OR PROCESSING OF GOODS',
                    'Y02T': 'TRANSPORTATION',
                    'Y02W': 'WASTEWATER TREATMENT OR WASTE MANAGEMENT',
                    'Y04S': 'SYSTEMS INTEGRATING TECHNOLOGIES RELATED TO POWER NETWORK OPERATION, COMMUNICATION OR INFORMATION TECHNOLOGIES'}

## Lowercase the contents from Y02_category_dict[Y_category]

current_Y02_category_USPTOkeywords = Y02_category_dict[Y_category].lower()

print (current_Y02_category_USPTOkeywords)

## Read the current Y_category_all

df_Current_Y_category_all = pd.read_csv(df_Current_Y_category_all_address, header=0, dtype='unicode', low_memory=False)

## I will only keep the patent_id, patent_title, patent_abstract, and label columns.

df_Current_Y_category_all = df_Current_Y_category_all[['patent_id', 'patent_title', 'patent_abstract', 'label']]

## Get the rows when label is equals to 0 or 3.
# WHen it is 0, it is a green patent from USPTO. We need to check whether it is green.
# WHen it is 3, it is a green patent candidate we want to discover.

df_Current_Y_category_all = df_Current_Y_category_all.loc[df_Current_Y_category_all['label'].isin(['0', '3'])]

## Create a new column by adding the patent_title and patent_abstract together.

df_Current_Y_category_all['content'] = 'Tell me whether this is a green patent related to '+ current_Y02_category_USPTOkeywords +' with Yes or No. ' + df_Current_Y_category_all['patent_title'] + '.' + df_Current_Y_category_all['patent_abstract']

## Transform the column "label" value into Yes or No. WHen the value is 0, it is Yes. When the value is 3, it is No.

df_Current_Y_category_all['label_USPTO_original'] = np.where(df_Current_Y_category_all['label'] == '0', 'Yes', 'No')

## save the df_Current_Y_category_all into a csv file in the tmp folder.

df_Current_Y_category_all.to_csv('Results/Step4/'+Y_category+'_df_Current_Y_category_all_with_content.csv', index=False)


# df_Current_Y_category_all['content'] = 'Tell me whether this can be considered as a green patent with Yes or No' + df_Current_Y_category_all['patent_title'] + '.' + df_Current_Y_category_all['patent_abstract']

## Drop patent_title and patent_abstract, and patent label columns.

df_Current_Y_category_all = df_Current_Y_category_all.drop(['patent_title', 'patent_abstract'], axis=1)


## get the head 100 rows from label 0; then get the head 100 rows from label 3. This is to testing the openAI API.

df_Current_Y_category_all_label_USPTO_Possible_Green = df_Current_Y_category_all.loc[df_Current_Y_category_all['label'] == '0'].head(Predicting_Patents_sample_size)

df_Current_Y_category_all_label_Outside_Possible_Green = df_Current_Y_category_all.loc[df_Current_Y_category_all['label'] == '3'].head(Predicting_Patents_sample_size)

print (len(df_Current_Y_category_all_label_USPTO_Possible_Green))
print (len(df_Current_Y_category_all_label_Outside_Possible_Green))

### FOr the data in df_Current_Y_category_all_label_Outside_Possible_Green. I will utilize my own trained GPT3 model to predict whether it is a green patent.
df_Current_Y_category_Possible_Green_GPT35 = pd.concat([df_Current_Y_category_all_label_USPTO_Possible_Green, df_Current_Y_category_all_label_Outside_Possible_Green])


##  Fine tuned GPT3.5 Turbo Predictions

print (' Fine tuned GPT3.5 Turbo Predictions')

def add_predictions_to_dataframe(df):
    # Create an empty list to store the predictions
    predictions = []

    for index, row in df.iterrows():
        patent_content = row['content']
        patent_label = row['label']
        system_content = 'You are a green patent examiner for '+ current_Y02_category_USPTOkeywords +'.'

        response = client.chat.completions.create(
            # model="ft:gpt-3.5-turbo-0613:personal:y02a:8dBmrYkb",
            # model="ft:gpt-3.5-turbo-0613:personal:y02b:8dZ5klxT",
            # model="ft:gpt-3.5-turbo-0613:personal:y02c:8cqGJJl0",
            #model="ft:gpt-3.5-turbo-0613:personal:y02d:8dagFMLn",
            # model="ft:gpt-3.5-turbo-0613:personal:y02e:8ct2RcEJ",
            # model="ft:gpt-3.5-turbo-0613:personal:y02p:8duF5NRk",
            # model="ft:gpt-3.5-turbo-0613:personal:y02t:8aI1lfgn",
            #model="ft:gpt-3.5-turbo-0613:personal:y02w:8dvinqmB",
            model="ft:gpt-3.5-turbo-0613:personal:y04s:8dxNdlqN",

            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": patent_content}
            ]
        )

        # Extract the last message from the response
        last_message = response.choices[0].message.content  # Get the content of the last message
        predictions.append(last_message)

        print (index)
        print (last_message)

    # Add the predictions as a new column to the DataFrame
    df['predictions'] = predictions

# Assuming df_Current_Y_category_all is your DataFrame
add_predictions_to_dataframe(df_Current_Y_category_Possible_Green_GPT35)

# Now df_Current_Y_category_all contains a new column 'predictions' with the AI responses. save it into a csv file.

df_Current_Y_category_Possible_Green_GPT35.to_csv('Results/Step4/'+Y_category+'_df_Current_Y_category_Possible_Green_GPT35.csv', index=False)



### not used any more
### not used any more
#
#
# response = client.chat.completions.create(
#   model="ft:gpt-3.5-turbo-0613:personal::8XS5CVag",
#   messages=[
#     {"role": "system", "content": "You are a green patent examiner for transportation innovation."},
#     #{"role": "user", "content": "Tell me whether this is a green patent. Integrated hydromethanation fuel cell power generation. The present invention relates to processes and apparatuses for generating electrical power from certain non-gaseous carbonaceous feedstocks through the integration of catalytic hydromethanation technology with fuel cell technology."}
#     #{"role": "user", "content": "Tell me whether this is a green patent. Fuel injection system for internal combustion engine. A fuel injection system for an internal combustion engine is provided which works to correct the pressure of fuel, as measured by a pressure sensor, using a pressure change corresponding to a change in quantity of the fuel in a common rail within a pressure change compensating time Tp to determine a pump discharge pressure Ptop. This compensates for an error in determining the pump discharge pressure Ptop which arises from propagation of the pressure of fuel from a pump to the pressure sensor. The pressure change compensating time Tp is the sum of a time T1 elapsed between sampling the output of the pressure sensor before a calculation start time when the pump discharge pressure is to start to be calculated and the calculation start time and a time T2 required for the pressure to transmit from the outlet of the pump to the pressure sensor."}
#     # {"role": "user",
#     #    "content": "Tell me whether this is a green patent. Connector assembly. The present invention provides a connector assembly allowing attachment of a connector to a connector receptacle with a limited attachment space. The connector assembly includes the connector having a terminal and a connector housing, the connector receptacle having a case and a terminal portion, and a fastening member. The terminal has an electrical contact portion with a hole. The connector housing has a column portion for receiving the electrical contact portion with the hole aligned in an insertion direction of the connector. The terminal portion has a connection portion to be superposed with the electrical contact portion and has a second hole to be communicated with the hole. The case has a housing portion for supporting the connection portion so that the second hole of the connection portion is aligned with the insertion direction of the connector. The electrical contact portion and the connection portion are fastened together with the fastening member passed through the communicated hole and the second hole."}
#     {"role": "user",
#        "content": "Tell me whether this is a green patent. System for heating a fuel.An embodiment of the present invention takes the form of a system that uses exhaust from the combustion turbine engine to heat the fuel gas consumed by a combustion turbine engine. The benefits of the present invention include reducing the need to use a parasitic load to heat the fuel gas."}
#
#   ]
# )
# print(response.choices[0].message.content)
#
