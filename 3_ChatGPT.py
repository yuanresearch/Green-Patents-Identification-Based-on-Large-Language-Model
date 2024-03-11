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

Desktop_Developing = True

Patents_sample_size = 5000

if Desktop_Developing:
    Y_category ='Y04S'   ### ONly accept 1 element
    df_Current_Y_category_all_address = 'ResultsServer/Step2/'+Y_category+ '_ML_Candidates.csv'

else:
    Y_category = sys.argv[1]
    df_Current_Y_category_all_address = 'Results/Step2/'+Y_category+ '_ML_Candidates.csv'

## Read the current Y_category_all

df_Current_Y_category_all = pd.read_csv(df_Current_Y_category_all_address, header=0, dtype='unicode', low_memory=False)

## Get the counts of each label in the label column in current Y_category_all

df_Current_Y_category_all_counts = df_Current_Y_category_all['label'].value_counts().to_frame().reset_index()

print (len(df_Current_Y_category_all.index))

print(df_Current_Y_category_all_counts)

##Step1: I will read openAI key from Config.txt

with open('Config.txt') as f:

    lines = f.readlines()
    openAI_key = lines[0].strip()
    print (openAI_key)


## Step2: Format the database for fine-tuning.
# {"prompt": "Patent description: [Patent Text]", "completion": "green"}
# {"prompt": "Patent description: [Patent Text]", "completion": "not green"}

## Step2.1: I will read the current Y_category_all

df_Current_Y_category_all = pd.read_csv(df_Current_Y_category_all_address, header=0, dtype='unicode', low_memory=False)

## Step2.2: I will only keep the patent_id, patent_title, patent_abstract, and label columns.

df_Current_Y_category_all = df_Current_Y_category_all[['patent_id', 'patent_title', 'patent_abstract', 'label']]

## Step2.3: Each line in your JSONL file should represent a single training example. For binary classification, it might look something like this:
## {"prompt": "Patent description: [Patent Text]", "completion": "green"}
## {"prompt": "Patent description: [Patent Text]", "completion": "not green"}
## I will create a new column with the combination of patent_title and patent_abstract.

df_Current_Y_category_all['description'] = 'Patent Title: '+ df_Current_Y_category_all['patent_title'] + '. Patent Abstract: ' + df_Current_Y_category_all['patent_abstract']

## Step2.4: I will create a new column named completion.
## WHen the label is equals to 1, the completion is green;
## when the label is equals to 2, the completion is not green.
## WHen the label is equals to 0, the completion is original_green_candidate
## WHen the label is equals to 3, the completion is original_non_green_candidate

df_Current_Y_category_all.loc[df_Current_Y_category_all['label'] == '1', 'completion'] = 'This is a green patent.'

df_Current_Y_category_all.loc[df_Current_Y_category_all['label'] == '2', 'completion'] = 'This is a non-green patent.'

df_Current_Y_category_all.loc[df_Current_Y_category_all['label'] == '0', 'completion'] = 'This is a green patent from USPTO. We need to check whether it is green.'

df_Current_Y_category_all.loc[df_Current_Y_category_all['label'] == '3', 'completion'] = 'This is a green patent candidate. We need to check whether it is green.'

## save to tmp folder

print (df_Current_Y_category_all.head(5))

## Remove all duplicates. I will keep the first one.
## Get the counts of each label in the label column in current Y_category_all
df_Current_Y_category_all = df_Current_Y_category_all.drop_duplicates(subset=['patent_id'], keep='first')

df_Current_Y_category_all_counts = df_Current_Y_category_all['label'].value_counts().to_frame().reset_index()

print(df_Current_Y_category_all_counts)


## Step2.6: Keep the rows when the label is equals to 1 or 2.

df_Current_Y_category_all_training = df_Current_Y_category_all.loc[df_Current_Y_category_all['label'].isin(['1', '2'])]

df_Current_Y_category_all_counts_training = df_Current_Y_category_all_training['label'].value_counts().to_frame().reset_index()

print(df_Current_Y_category_all_counts_training)

## Step2.7. I will sample Patents_sample_size rows from label 1, then split our data set into the following parts: training (70%), validation (20%), and test (10%).

df_Current_Y_category_all_training_label_1 = df_Current_Y_category_all_training.loc[df_Current_Y_category_all_training['label'] == '1']

df_Current_Y_category_all_training_label_2 = df_Current_Y_category_all_training.loc[df_Current_Y_category_all_training['label'] == '2']

## if the row counts of df_Current_Y_category_all_training_label_1 is larger than Patents_sample_size, I will sample Patents_sample_size rows from df_Current_Y_category_all_training_label_1.
## else I will keep all rows from df_Current_Y_category_all_training_label_1.

if len(df_Current_Y_category_all_training_label_1.index) > Patents_sample_size:

    df_Current_Y_category_all_training_label_1 = df_Current_Y_category_all_training_label_1.sample(n=Patents_sample_size, random_state=1)

## if the row counts of df_Current_Y_category_all_training_label_2 is larger than Patents_sample_size, I will sample Patents_sample_size rows from df_Current_Y_category_all_training_label_2.
## else I will keep all rows from df_Current_Y_category_all_training_label_2.

if len(df_Current_Y_category_all_training_label_2.index) > Patents_sample_size:

    df_Current_Y_category_all_training_label_2 = df_Current_Y_category_all_training_label_2.sample(n=Patents_sample_size, random_state=1)

df_Current_Y_category_all_training_label_1_train, df_Current_Y_category_all_training_label_1_validate, df_Current_Y_category_all_training_label_1_test = \
              np.split(df_Current_Y_category_all_training_label_1.sample(frac=1, random_state=1),
                       [int(.7*len(df_Current_Y_category_all_training_label_1)), int(.9*len(df_Current_Y_category_all_training_label_1))])

df_Current_Y_category_all_training_label_2_train, df_Current_Y_category_all_training_label_2_validate, df_Current_Y_category_all_training_label_2_test = \
                np.split(df_Current_Y_category_all_training_label_2.sample(frac=1, random_state=1),
                            [int(.7*len(df_Current_Y_category_all_training_label_2)), int(.9*len(df_Current_Y_category_all_training_label_2))])

df_Current_Y_category_all_training_train = pd.concat([df_Current_Y_category_all_training_label_1_train, df_Current_Y_category_all_training_label_2_train])

df_Current_Y_category_all_training_validate = pd.concat([df_Current_Y_category_all_training_label_1_validate, df_Current_Y_category_all_training_label_2_validate])

df_Current_Y_category_all_training_test = pd.concat([df_Current_Y_category_all_training_label_1_test, df_Current_Y_category_all_training_label_2_test])


#### Step2.8: I will save the training, validation and testing files into the Results/Step3/
#### I only keep two columns: description and completion.

df_Current_Y_category_all_training_train = df_Current_Y_category_all_training_train[['description', 'completion']]

df_Current_Y_category_all_training_train.to_csv('Results/Step3/'+Y_category+'_3_df_Current_Y_category_all_training_train.csv', index=False)

df_Current_Y_category_all_training_validate = df_Current_Y_category_all_training_validate[['description', 'completion']]

df_Current_Y_category_all_training_validate.to_csv('Results/Step3/'+Y_category+'_3_df_Current_Y_category_all_training_validate.csv', index=False)

df_Current_Y_category_all_training_test = df_Current_Y_category_all_training_test[['description', 'completion']]

df_Current_Y_category_all_training_test.to_csv('Results/Step3/'+Y_category+'_3_df_Current_Y_category_all_training_test.csv', index=False)

## print size of each file

print (len(df_Current_Y_category_all_training_train.index))

print (len(df_Current_Y_category_all_training_validate.index))

print (len(df_Current_Y_category_all_training_test.index))


## Step3: convert the CSV file to JSONL file.

import csv
import json


def convert_csv_to_jsonl(csv_file_path, jsonl_file_path):
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file, \
            open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            # Structure the conversation correctly
            conversation = {
                "messages": [
                    {"role": "system", "content": "You are a green patent examiner."},
                    {"role": "user", "content": row['description']},
                    {"role": "assistant", "content": row['completion']}
                ],
            }

            # Write each conversation as a single JSON object to the JSONL file
            jsonl_file.write(json.dumps(conversation) + "\n")


# Convert your CSV file
convert_csv_to_jsonl('Results/Step3/'+Y_category+'_3_df_Current_Y_category_all_training_train.csv',
                     'Results/Step3/'+Y_category+'_3_df_Current_Y_category_all_training_train.jsonl')

# Convert your CSV file
convert_csv_to_jsonl('Results/Step3/'+Y_category+'_3_df_Current_Y_category_all_training_validate.csv',
                     'Results/Step3/'+Y_category+'_3_df_Current_Y_category_all_training_validate.jsonl')

# Convert your CSV file
convert_csv_to_jsonl('Results/Step3/'+Y_category+'_3_df_Current_Y_category_all_training_test.csv',
                     'Results/Step3/'+Y_category+'_3_df_Current_Y_category_all_training_test.jsonl')


## Step4: Upload Data to OpenAI

client = OpenAI(api_key=openAI_key)

print (client.files.list())

## If the list has a file named 3_df_Current_Y_category_all_counts_training.jsonl, I will delete it.

for d in client.files.list().data:

    print (d.filename)

    if      (d.filename == (Y_category+'_3_df_Current_Y_category_all_training_train.jsonl')
            or d.filename == (Y_category+'_3_df_Current_Y_category_all_training_validate.jsonl')
            or d.filename == (Y_category+'_3_df_Current_Y_category_all_training_test.jsonl')):

        print ("File exists. I will delete it.")

        print (d.id)

        client.files.delete(d.id)

## I will upload the file to OpenAI.

response_train = client.files.create(
  file=open(('Results/Step3/'+Y_category+'_3_df_Current_Y_category_all_training_train.jsonl'), "rb"),
  purpose="fine-tune"
)

print(response_train)

file_id_train = response_train.id
print("file_id_train ID:", file_id_train)


response_validate = client.files.create(
  file=open(('Results/Step3/'+Y_category+'_3_df_Current_Y_category_all_training_validate.jsonl'), "rb"),
  purpose="fine-tune"
)

print(response_validate)

file_id_validate = response_validate.id
print("file_id_validate ID:", file_id_validate)

#########################################
# ## Step5: Fine-tune the model. Expensive. I will skip this step when it is no longer use.
#########################################

fine_tune = client.fine_tuning.jobs.create(
  training_file=file_id_train,
validation_file=file_id_validate,
  model="gpt-3.5-turbo",
suffix=str(Y_category)
)

print(fine_tune)

fine_tune_job_id = fine_tune.id
print("Fine-tuning job ID:", fine_tune_job_id)

import time

while True:
    fine_tune_status = client.fine_tuning.jobs.retrieve(fine_tune_job_id)
    if fine_tune_status.status == "succeeded":
        break
    elif fine_tune_status.status == "failed":
        print("Fine-tuning failed.")
        break
    time.sleep(300)  # Wait for 300 seconds before checking again

print("Fine-tuning completed successfully.")

fine_tuned_model_id = fine_tune_status.fine_tuned_model
print("Fine-tuned model ID:", fine_tuned_model_id)

## Save the fine-tuned model ID to a txt file.

with open('Results/Step3/'+Y_category+'_3_df_Current_Y_category_all_training_fine_tuned_model_id.txt', 'w') as f:
    f.write(fine_tuned_model_id)
