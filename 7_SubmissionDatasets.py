# Importing necessary packages.
import sys
import os
import time
import zipfile as zip


import pandas as pd
import csv


print ("Python version: {}".format(sys.version))

# ## First, I will count the number of rows from each file from the USPTOResults folder. It will serve as the baseline
# ## for the number of patents defined by USPTO in the dataset.
#
# ## Loop each file in the folder and count the number of rows.
#
# # Path to the folder containing the USPTO Results
#
# df_Current_Y_category_address = 'USPTOResults'
#
# # List of files in the folder
#
# df_Current_Y_category_files = os.listdir(df_Current_Y_category_address)
#
# # Loop through each file and count the number of rows
#
# for file in df_Current_Y_category_files:
#
#     # Read the file
#
#     df = pd.read_csv(df_Current_Y_category_address + '/' + file)
#
#     # Count the number of rows
#
#     print (file, len(df))
#
#
#
# ## Now I need to
# # 1)count the YES/NO from label_USPTO_original column from each file from the GPTResults folder.
# # 2)count the YES/NO from the predictions_clean column if label_USPTO_original column is YES;
# # 3)count the YES/NO from the predictions_clean column if label_USPTO_original column is NO.
# # 4)create an empty panda dataframe and save the above results in it.
# # 5)save the above results in the same dataframe, with each csv file name as the first column.
#
#
#
#
# ## Loop each file in the folder and count the number of YES/NO.
#
# # Path to the folder containing the GPT Results
#
# df_Current_Y_category_address = 'GPTResults'
#
# # List of files ends with csv in the folder
#
# df_Current_Y_category_files = [file for file in os.listdir(df_Current_Y_category_address) if file.endswith('.csv')]
#
# # Create an empty dataframe
#
# df_step7 = pd.DataFrame(columns = ['File', 'Yes_USPTO', 'No_USPTO', 'Yes_GPT_Yes_USPTO', 'No_GPT_Yes_USPTO', 'Yes_GPT_No_USPTO', 'No_GPT_No_USPTO'])
#
# # Loop through each file ends with csv and count the number of YES/NO
#
# for file in df_Current_Y_category_files:
#
#         # Read the file
#
#         df = pd.read_csv(df_Current_Y_category_address + '/' + file)
#
#         # Count the number of YES/NO from label_USPTO_original column
#
#         print (file, df['label_USPTO_original'].value_counts())
#
#         # Count the number of YES/NO from the predictions_clean column if label_USPTO_original column is YES
#
#         print (file, df[df['label_USPTO_original'] == 'Yes']['predictions_clean'].value_counts())
#
#         # Count the number of YES/NO from the predictions_clean column if label_USPTO_original column is NO
#
#         print (file, df[df['label_USPTO_original'] == 'No']['predictions_clean'].value_counts())
#
#         # Save the above results in the dataframe
#
#         df_step7 = df_step7._append({'File': file, 'Yes_USPTO': df['label_USPTO_original'].value_counts()['Yes'], 'No_USPTO': df['label_USPTO_original'].value_counts()['No'], 'Yes_GPT_Yes_USPTO': df[df['label_USPTO_original'] == 'Yes']['predictions_clean'].value_counts()['Yes'], 'No_GPT_Yes_USPTO': df[df['label_USPTO_original'] == 'Yes']['predictions_clean'].value_counts()['No'], 'Yes_GPT_No_USPTO': df[df['label_USPTO_original'] == 'No']['predictions_clean'].value_counts()['Yes'], 'No_GPT_No_USPTO': df[df['label_USPTO_original'] == 'No']['predictions_clean'].value_counts()['No']}, ignore_index = True)
#
# # Save the dataframe in a csv file
#
# df_step7.to_csv('Nauture Sumbission/Step7_GPTResults.csv', index = False)
#
# ### The first part is done. Now I will trim the GPTResults and USPTOResults's columns, so the
# ### files are under 100MB and can be uploaded to the Github repository. All the files will be
# ### saved in the Nauture Sumbission Github folder.
#
# ## Loop each file in the folder and trim the columns.
#
# # Path to the folder containing the GPT Results
#
# df_Current_Y_category_address = 'GPTResults'
#
# # List of files ends with csv in the folder
#
# df_Current_Y_category_files = [file for file in os.listdir(df_Current_Y_category_address) if file.endswith('.csv')]
#
# # Loop through each file ends with csv and trim the columns
#
# for file in df_Current_Y_category_files:
#
#     # Read the file
#
#     df = pd.read_csv(df_Current_Y_category_address + '/' + file)
#
#     # Trim the columns
#
#     df = df[['patent_id','content', 'label_USPTO_original', 'predictions_clean']]
#
#     # Save the file
#
#     df.to_csv('Nauture Sumbission Github/' + file, index = False)
#
#
# # Path to the folder containing the USPTO Results
#
# df_Current_Y_category_address = 'USPTOResults'
#
# # List of files in the folder
#
# df_Current_Y_category_files = os.listdir(df_Current_Y_category_address)
#
# # Loop through each file and trim the columns
#
# for file in df_Current_Y_category_files:
#
#     # Read the file
#
#     df = pd.read_csv(df_Current_Y_category_address + '/' + file)
#
#     # Trim the columns
#
#     df = df[['patent_id','patent_date', 'patent_title','patent_abstract','cpc_subclass']]
#
#     # Save the file
#
#     df.to_csv('Nauture Sumbission Github/' + file, index = False)
#

### THe second part is done. Now I need to check the size of the files in the Nauture Sumbission Github folder.
### if the size of one file is greater than 100 MB, I will split it into two files and save them in the same folder.

## Loop each file in the folder and check the size of the files.

# Path to the folder containing the Nauture Sumbission Github

df_Current_Y_category_address = 'Nauture Sumbission Github'

# List of files in the folder ends with csv

df_Current_Y_category_files = [file for file in os.listdir(df_Current_Y_category_address) if file.endswith('.csv')]

# Loop through each file and check the size of the files

for file in df_Current_Y_category_files:

    # Check the size of the file

    print (file, os.path.getsize(df_Current_Y_category_address + '/' + file))

    # If the size of the file is greater than 100 MB, split it into two files and save them in the same folder

    if os.path.getsize(df_Current_Y_category_address + '/' + file) > 100000000:

        # Read the file

        df = pd.read_csv(df_Current_Y_category_address + '/' + file)

        # Split the file into two files

        df1 = df.iloc[:len(df)//2]
        df2 = df.iloc[len(df)//2:]

        # Save the files

        df1.to_csv(df_Current_Y_category_address + '/' + file[:-4] + '_1.csv', index = False)
        df2.to_csv(df_Current_Y_category_address + '/' + file[:-4] + '_2.csv', index = False)

        # Remove the original file

        os.remove(df_Current_Y_category_address + '/' + file)






