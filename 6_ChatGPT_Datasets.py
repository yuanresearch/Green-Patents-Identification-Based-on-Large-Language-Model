#pip3 install openai
#pip3 install --upgrade tiktoken


# Importing necessary packages.
import sys
import os
import time
import zipfile as zip


import pandas as pd
import csv
import numpy as np
from openai import OpenAI

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

print(num_tokens_from_string("Hello world, let's test tiktoken.", "gpt-3.5-turbo"))

with open('Config.txt') as f:

    lines = f.readlines()
    openAI_key = lines[0].strip()
    print (openAI_key)

client = OpenAI(api_key=openAI_key)

Predicting_Patents_sample_size = 10

Desktop_Developing = False   ### Batch processing

if Desktop_Developing:
    Y_category ='Y02E'   ### ONly accept 1 element
    df_Current_Y_category_all_address = 'ResultsServer/Step2/'+Y_category+ '_ML_Candidates.csv'

else:
    Y_category = sys.argv[1]
    df_Current_Y_category_all_address = 'ResultsServer/Step2/'+Y_category+ '_ML_Candidates.csv'

# python3 6_ChatGPT_Datasets.py Y02A &&
# python3 6_ChatGPT_Datasets.py Y02W &&
# python3 6_ChatGPT_Datasets.py Y02T &&
# python3 6_ChatGPT_Datasets.py Y02P


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


#### A dictionary mapping Y02 category to the corresponding model name.

Y02_category_model_dict = {'Y02A': 'ft:gpt-3.5-turbo-0613:personal:y02a:8dBmrYkb',
                            'Y02B': 'ft:gpt-3.5-turbo-0613:personal:y02b:8dZ5klxT',
                            'Y02C': 'ft:gpt-3.5-turbo-0613:personal:y02c:8cqGJJl0',
                            'Y02D': 'ft:gpt-3.5-turbo-0613:personal:y02d:8dagFMLn',
                            'Y02E': 'ft:gpt-3.5-turbo-0613:personal:y02e:8ct2RcEJ',
                            'Y02P': 'ft:gpt-3.5-turbo-0613:personal:y02p:8duF5NRk',
                            'Y02T': 'ft:gpt-3.5-turbo-0613:personal:y02t:8aI1lfgn',
                            'Y02W': 'ft:gpt-3.5-turbo-0613:personal:y02w:8dvinqmB',
                            'Y04S': 'ft:gpt-3.5-turbo-0613:personal:y04s:8dxNdlqN'}

current_Y02_category_model = Y02_category_model_dict[Y_category]

print (current_Y02_category_model)



## Read the current Y_category_all

df_Current_Y_category_all = pd.read_csv(df_Current_Y_category_all_address, header=0, dtype='unicode', low_memory=False)

## I will only keep the patent_id, patent_title, patent_abstract, and label columns.

df_Current_Y_category_all = df_Current_Y_category_all[['patent_id', 'patent_title', 'patent_abstract', 'label']]

## Debug code: select rows when patent_id is 6471020 or 6470985 or 6472608

# df_Current_Y_category_all = df_Current_Y_category_all.loc[df_Current_Y_category_all['patent_id'].isin(['6471020', '6470985', '6472608'])]


## I need to get rid of the rows with empty patent_title or patent_abstract. Very important.

df_Current_Y_category_all = df_Current_Y_category_all.dropna(subset=['patent_title', 'patent_abstract'])

df_Current_Y_category_all_copy = df_Current_Y_category_all.copy()


## Get the rows when label is equals to 0 or 3.
# WHen it is 0, it is a green patent from USPTO. We need to check whether it is green.
# WHen it is 3, it is a green patent candidate we want to discover.

df_Current_Y_category_all = df_Current_Y_category_all.loc[df_Current_Y_category_all['label'].isin(['0', '3'])]

df_Current_Y_category_all_absolute_green = df_Current_Y_category_all_copy.loc[df_Current_Y_category_all_copy['label'] =='1']

## Print the counts of each label.

print (df_Current_Y_category_all['label'].value_counts())
print (df_Current_Y_category_all_absolute_green['label'].value_counts())

## Create a new column by adding the patent_title and patent_abstract together.

df_Current_Y_category_all['content'] = 'Tell me whether this is a green patent related to '+ current_Y02_category_USPTOkeywords +' with only YES or NO. ' + df_Current_Y_category_all['patent_title'] + '.' + df_Current_Y_category_all['patent_abstract']

## Transform the column "label" value into Yes or No. WHen the value is 0, it is Yes. When the value is 3, it is No.

df_Current_Y_category_all['label_USPTO_original'] = np.where(df_Current_Y_category_all['label'] == '0', 'Yes', 'No')
df_Current_Y_category_all_absolute_green['label_USPTO_original'] = np.where(df_Current_Y_category_all_absolute_green['label'] == '1', 'Yes', 'No')

## Get numebr of tokens for each row in the content column.

df_Current_Y_category_all['num_tokens'] = df_Current_Y_category_all['content'].apply(lambda x: num_tokens_from_string(str(x), "gpt-3.5-turbo"))

## get the head 100 rows from label 0; then get the head 100 rows from label 3. This is to testing the openAI API.

# df_Current_Y_category_all_label_USPTO_Possible_Green = df_Current_Y_category_all.loc[df_Current_Y_category_all['label'] == '0'].head(Predicting_Patents_sample_size)
# df_Current_Y_category_all_label_Outside_Possible_Green = df_Current_Y_category_all.loc[df_Current_Y_category_all['label'] == '3'].head(Predicting_Patents_sample_size)

df_Current_Y_category_all_label_USPTO_Possible_Green = df_Current_Y_category_all.loc[df_Current_Y_category_all['label'] == '0']
df_Current_Y_category_all_label_Outside_Possible_Green = df_Current_Y_category_all.loc[df_Current_Y_category_all['label'] == '3']

print (len(df_Current_Y_category_all_label_USPTO_Possible_Green))
print (len(df_Current_Y_category_all_label_Outside_Possible_Green))

### FOr the data in df_Current_Y_category_all_label_Outside_Possible_Green. I will utilize my own trained GPT3 model to predict whether it is a green patent.
df_Current_Y_category_Possible_Green_GPT35 = pd.concat([df_Current_Y_category_all_label_USPTO_Possible_Green, df_Current_Y_category_all_label_Outside_Possible_Green])

print ("The total counts of the df_Current_Y_category_Possible_Green_GPT35 is: ", len(df_Current_Y_category_Possible_Green_GPT35))




##  Fine tuned GPT3.5 Turbo Predictions

print (' Fine tuned GPT3.5 Turbo Predictions')

def add_predictions_to_dataframe(df):
    # Create an empty list to store the predictions
    predictions = []

    counts = 0


    for index, row in df.iterrows():
        patent_content = row['content']
        patent_label = row['label']
        system_content = 'You are a green patent examiner for '+ current_Y02_category_USPTOkeywords +'.'


        ## I need to check the length of patent_content,
        # make sure the maximum context length not more than 4000 tokens as limited by OpenAI.
        ## if the tokens is indeed larger than 4000, set last_message to be too much contents.

        num_tokens = int(row['num_tokens'])

        print (patent_content)
        #print (num_tokens)

        if num_tokens > 2000:

            last_message = 'No. The content is too long. Please check the patent content manually.'

            predictions.append(last_message)

            counts = counts + 1

            print (counts)

            print (last_message)

            continue

        retries = 2

        while retries > 0:

            try:
                response = client.chat.completions.create(
                # model="ft:gpt-3.5-turbo-0613:personal:y02a:8dBmrYkb",
                # model="ft:gpt-3.5-turbo-0613:personal:y02b:8dZ5klxT",
                # model="ft:gpt-3.5-turbo-0613:personal:y02c:8cqGJJl0",
                # model="ft:gpt-3.5-turbo-0613:personal:y02d:8dagFMLn",
                # model="ft:gpt-3.5-turbo-0613:personal:y02e:8ct2RcEJ",
                # model="ft:gpt-3.5-turbo-0613:personal:y02p:8duF5NRk",
                # model="ft:gpt-3.5-turbo-0613:personal:y02t:8aI1lfgn",
                # model="ft:gpt-3.5-turbo-0613:personal:y02w:8dvinqmB",
                # model="ft:gpt-3.5-turbo-0613:personal:y04s:8dxNdlqN",

                model = current_Y02_category_model,

                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": patent_content}
                    ],

                timeout = 30,
                )

                # Extract the last message from the response
                last_message = response.choices[0].message.content  # Get the content of the last message
                predictions.append(last_message)

                counts = counts + 1

                print(counts)
                print(last_message)

                ## Break the while loop

                break

            except Exception as e:

                if e:
                    print(e)
                    print('Timeout error, retrying...')
                    retries -= 1
                    time.sleep(300)
                    ## Get back to while loop until retries is 0.
                    continue

                else:
                    raise e



    # Add the predictions as a new column to the DataFrame
    df['predictions'] = predictions

# Assuming df_Current_Y_category_all is your DataFrame
add_predictions_to_dataframe(df_Current_Y_category_Possible_Green_GPT35)

## In the predictions column, it will have the responses from the AI. I will transform the responses
## into Yes or No. for example, when the response contains non or no, it is No. Otherwise, it is Yes.

df_Current_Y_category_Possible_Green_GPT35['predictions_clean'] = np.where(df_Current_Y_category_Possible_Green_GPT35['predictions'].str.contains('no|non|not', case=False), 'No', 'Yes')

# Now df_Current_Y_category_all contains a new column 'predictions' with the AI responses. save it into a csv file.
## concate the df_Current_Y_category_Possible_Green_GPT35 with df_Current_Y_category_all_absolute_green

df_Current_Y_category_Possible_Green_GPT35 = pd.concat([df_Current_Y_category_Possible_Green_GPT35, df_Current_Y_category_all_absolute_green])

df_Current_Y_category_Possible_Green_GPT35.to_csv('Results/Step6/'+Y_category+'_Datasets_FineTuned_GPT35.csv', index=False)



