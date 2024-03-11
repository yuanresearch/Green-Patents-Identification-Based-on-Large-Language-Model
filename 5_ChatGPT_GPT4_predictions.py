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


## 'Results/Step4/'+Y_category+'_df_Current_Y_category_Possible_Green_GPT35.csv'

df_Current_Y_category_Possible_Green_GPT35 = pd.read_csv('Results/Step4/'+Y_category+'_df_Current_Y_category_Possible_Green_GPT35.csv', header=0, dtype='unicode', low_memory=False)

print (len(df_Current_Y_category_Possible_Green_GPT35))

df_Current_Y_category_Possible_Green_GPT35_USPTO_Possible_Green = df_Current_Y_category_Possible_Green_GPT35.loc[df_Current_Y_category_Possible_Green_GPT35['label'] == '0'].head(Predicting_Patents_sample_size)

df_Current_Y_category_Possible_Green_GPT35_Outside_Possible_Green = df_Current_Y_category_Possible_Green_GPT35.loc[df_Current_Y_category_Possible_Green_GPT35['label'] == '3'].head(Predicting_Patents_sample_size)


df_Current_Y_category_Possible_Green_GPT4 = pd.concat([df_Current_Y_category_Possible_Green_GPT35_USPTO_Possible_Green, df_Current_Y_category_Possible_Green_GPT35_Outside_Possible_Green])


## Step1: GPT4 Predictions

print ('Step1: GPT4 Predictions')

def add_predictions_to_dataframe(df): ## Use GPT4 to predict whether it is a green patent.

    # Create an empty list to store the predictions
    predictions = []

    for index, row in df.iterrows():

        patent_content = row['content']

        response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a green patent examiner for transportation."},
                        {"role": "user", "content": patent_content}
                    ]
        )

        # Extract the last message from the response
        last_message = response.choices[0].message.content
        predictions.append(last_message)
        print (index)
        print (last_message)

    # Add the predictions as a new column to the DataFrame

    df['predictions_GPT4'] = predictions

# Assuming df_Current_Y_category_all is your DataFrame

add_predictions_to_dataframe(df_Current_Y_category_Possible_Green_GPT4)

# Now df_Current_Y_category_all contains a new column 'predictions' with the AI responses. save it into a csv file.

df_Current_Y_category_Possible_Green_GPT4.to_csv('Results/Step5/'+Y_category+'_df_Current_Y_category_Possible_Green_GPT4.csv', index=False)


