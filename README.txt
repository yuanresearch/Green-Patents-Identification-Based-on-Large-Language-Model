A Robust Green Patent Database: A New Dataset of Green Patents Through Large Language Models

Overview
This repository contains the code and dataset for the paper titled "A Robust Green Patent Database: A New Dataset of Green Patents Through Large Language Models." This research introduces a comprehensive dataset of green patents issued by the United States Patent and Trademark Office (USPTO) from 1976 to 2022, employing advanced Large Language Models (LLMs) and clustering techniques to enhance the understanding of green innovation.

Dataset Highlights
Total Patents Identified: 192,343 previously unclassified green patents.
Exclusions: 25,978 patents incorrectly classified as non-green.
Dataset Size: 31.5% larger than existing USPTO green patent datasets.
Accuracy: Our classification methodology achieved an accuracy exceeding 85%, significantly higher than the USPTO’s Y02 tagging scheme (74.5%).

Key Features
Enhanced Classification: Utilizes supervised and unsupervised machine learning techniques to classify patents as “green” or “not green.”
Clustering: Implements HDBSCAN to identify 807 distinct clusters within the green patent landscape.
Predictive Analysis: Incorporates LLMs to improve the identification process based on patent titles and abstracts.

Copy code
git clone https://github.com/yuanresearch/Green-Patents-Identification-Based-on-Large-Language-Model.git

Public Dissemination
In line with our commitment to transparency and collaboration, we encourage researchers, policymakers, and industry professionals to utilize this dataset for further analyses and discussions related to green technology innovation.

Future Directions
We aim to continuously refine this dataset by integrating advanced LLM methodologies to identify core technologies within each green category, enhancing the capacity to discern “real” green patents.