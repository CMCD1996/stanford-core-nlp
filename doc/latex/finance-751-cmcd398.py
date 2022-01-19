# Descriptions
# This script/function implements the StanfordNLP to score corporate culture, replicating the production of inputs in Option 1 as outputs
# Inputs for Option 1 include:
# 1.	Firm_score.xlsx contains five scores estimated with two different dictionaries for all calls. Scores ended with 1 (for example, integrity1) are estimated with the dictionary trained on the 258 call transcripts included in this sample.  Scores ended with 2 (for example, integrity2) are estimated with the dictionary from the original paper (Table IA3 in the Internet Appendix). Other variables include document_id (used in your coding), filename (file name used by CapitalIQ), firm_id (firm name) and call time (year and month of the call).
# 2.	Expanded_dict1.csv is the culture dictionary trained with the 258 call transcripts (the new dictionary).
# 3.	Expanded_dict2.csv is the culture dictionary from the original paper (the original dictionary).
# 4.	Word_contributin_TFIDF1.csv (Word_contributin_TFIDF2.csv) contains word contribution based on TFIDF score estimated with the new dictionary (the original dictionary). 
# 5.	The Li, Mai, Shen and Yan (2021) paper and the Internet Appendix of this paper. 
#
# Author: Connor McDowall
# Date: 25th August 2021

# Imports
# Transcript Processing Modules
import pandas as pd
from pathlib import Path
import shutil as sh
from pdfrw import PdfReader, PdfWriter
import pdfminer as pdfm
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
import io
import datefinder as dtf
# General Python Modules
import datetime
import functools
import logging
import sys
import math
import os
import pickle
import gensim
import itertools
from pprint import pprint
from collections import Counter, defaultdict, OrderedDict
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Set
from multiprocessing import Pool
from operator import itemgetter
from tqdm import tqdm as tqdm

# StanfordNLP Specific Functions
from culture import culture_models, file_util, preprocess, culture_dictionary, preprocess_parallel
from stanfordnlp.server import CoreNLPClient
import global_options
import parse
import clean_and_train
import create_dict
import score
import aggregate_firms

# Functions
def get_transcipts(firm_score_xlsx, transcript_directory,transcript_selected):
    """Locates and isolates transcripts for processing

    Args:
        firm_score_xlsx (xlsx): Excel file containing the initial list of transcripts
        transcript_directory (str): Source of all transcripts
        transcript_selected (str): Destination for transcripts of interest

    Returns:
        transcript_list (list): List of transcript filenames
        calltimes (list): List of calltimes
    """
    # Get list of filenames
    firms_df = pd.read_excel(firm_score_xlsx)
    firms_df=firms_df.dropna()
    firms_df.columns = firms_df.iloc[0]
    firms_df = firms_df.drop(2)
    firms_df = firms_df.reset_index(drop=True)
    transcript_list = firms_df['filename'].tolist()
    # Get list of calltimes for the firm ID
    calltimes = firms_df['calltime'].tolist()
    # Copy file into selection if exists
    files_found = 0
    files_to_find = len(transcript_list)
    missing_files_list = []
    for filename in transcript_list:
        transcipt_x = Path(transcript_directory +'/'+filename)
        if transcipt_x.is_file():
            transcipt_y = Path(transcript_selected +'/'+filename)
            sh.copy(transcipt_x,transcipt_y)
            files_found = files_found + 1
        else:
            missing_files_list.append(filename)
    missing_files = files_to_find - files_found
    if missing_files > 0:
        print('You are missing the following transcripts...')
        print(missing_files_list)
    else:
        print('All transcripts found')
    return transcript_list,calltimes
        
def create_ids(transcript_list,qa_num, company_ids_set, company_ids_order, documents_ids_text, calltimes):
    """Creates document identification, updates transcript list to only include transcript lists
    with Question and Answer Sections, and creates dataframe to compare results.

    Args:
        transcript_list (list): List of transcript filenames
        qa_num (list): List of page numbers denoting the start of question and answer sections
        company_ids_set (list): List of company names
        company_ids_order (list): List of numbers referencing number of file relating to one company
        documents_ids_text (str): Directory to store document id list as a text file
        calltimes (list): List of calltimes

    Returns:
        updated_transcript_list (list): List of updated filenames
        updated_document_ids (list): List of updated document ids
        updated_firm_id (list): List of updated firm ids
        output_df (dataframe): Dataframe with document information
    """
    # Initial lists
    document_ids = []
    firm_id = []
    # Updated lists
    updated_document_ids = []
    updated_firm_id = []
    updated_transcript_list = []
    updated_calltimes = []
    # Assigns document id
    idx = 0
    for i in range(len(transcript_list)):
        document_ids.append(str(i + 1)+'.F')
        if i < company_ids_order[idx]:
            firm_id.append(company_ids_set[idx])
        else:
            idx = idx +1
            firm_id.append(company_ids_set[idx])
    # Updates lists to remove entries with no question and answer sections
    for j in range(len(qa_num)):
        if qa_num[j] != 4:
            updated_document_ids.append(document_ids[j])
            updated_firm_id.append(firm_id[j])
            updated_transcript_list.append(transcript_list[j])
            updated_calltimes.append(calltimes[j])
    # Creates document_id text file
    with open(documents_ids_text, "w") as file:
        # Clear the file
        file.truncate(0)
        for element in updated_document_ids:
             file.write(element + "\n")
        file.close()
    # Creates a dataframe with updated transcript list
    output_df = pd.DataFrame(list(zip(updated_document_ids, updated_transcript_list,updated_firm_id)),
               columns =['document_id', 'filename', 'firm_id'])
    # Creates id2firsm csv
    for i in range(len(updated_calltimes)):
        val = updated_calltimes[i]
        new_val = int(str(val)[:4])
        updated_calltimes[i] =  new_val
    id2firms_df = pd.DataFrame(list(zip(updated_document_ids,updated_firm_id,updated_calltimes)),
               columns =['document_id', 'firm_id','time'])
    print(id2firms_df.head())
    id2firms_df.to_csv('data/input/id2firms.csv')
    return updated_transcript_list, updated_document_ids, updated_firm_id, output_df

def remove_transcript_metadata(transcript_list,qa_num,transcript_selected,transcript_processed):
    """Removes front matter, table of contents, call participants, and copyright disclaimer
    to process transcripts to a format suitable for combination. This is possible as the
    format is consistent for all earnings call transcripts.

    Args:
        transcript_list (list): List of transcript filenames
        qa_num (list): List of page numbers denoting the start of question and answer sections
        transcript_selected (str): String of selected transcipt directory 
        transcript_processed (str): String of processed transcript directory 
    """
    # Count for 
    i = 0
    # Create copy, remove pages, and move to processed directory
    for filename in transcript_list:
        # Defines pdfs
        input_pdf = Path(transcript_selected +'/'+filename)
        output_pdf = Path(transcript_processed +'/'+filename)
        # Defines objects
        reader_input = PdfReader(input_pdf)
        writer_output = PdfWriter()
        for page_x in range(len(reader_input.pages)):
            # Adds pages excluding sections prior to Q&A section and legal disclaimer
            if page_x >= qa_num[i]-1 and page_x < (len(reader_input.pages)-1):
                writer_output.addpage(reader_input.pages[page_x])
        writer_output.write(output_pdf)
        i = i + 1 
    return

def create_documents_text(transcript_list,transcript_processed, text_processed,documents_text):
    """Creates documents.txt file for the Stanford NLP

    Args:
        transcript_list(str): List of processed transcipts
        transcript_processed (str): String of processed transcript directory 
        text_processed (str): Directory to store text file
        documents_text (str): Directory for documents.txt file 

    Returns:
        documents_test_list (list): Returns a list of processed transcript document strings 
    """
    # Adapted from https://towardsdatascience.com/pdf-text-extraction-in-python-5b6ab9e92dd
    # Erase object contents to reset the textfile
    with open(documents_text, "r+") as file:
        file.truncate(0)
        file.close()
    # Creates empty list
    documents_test_list = []
    # Begin extracting files 
    for file_name in transcript_list:
        file_pdf = Path(transcript_processed +'/'+file_name)
        file_text = io.StringIO()
        with open(file_pdf, 'rb') as in_file:
            parser = PDFParser(in_file)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, file_text, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
        # Extract text to and remove characters
        textname = Path(text_processed +'/output.txt')
        with open(textname, "w") as file:
            file.write(file_text.getvalue())
            file.close()
        # Print the lines
        with open(textname, "r+") as file:
            line = file.read().replace("\n", " ")
            file.truncate(0)
            file.close()
        # Write line to the documents file
        with open(documents_text, "a") as file:
            file.write(line)
            if file_name != transcript_list[-1]:
                file.write("\n")
            file.close()
        # Create list of texts and dates
        documents_test_list.append(line)
    return documents_test_list

def prepare_documents(firm_score_xlsx, transcript_directory, transcript_selected, transcript_processed, text_processed, documents_text, documents_ids_text, qa_num, company_ids_set,company_ids_order):
    """ Isolate transcripts of interest, process Q&A sections, and create document files

    Args:
        firm_score_xlsx (xlsx): Excel file containing the initial list of transcripts
        transcript_directory (str): Source of all transcripts
        transcript_selected (str): Destination for transcripts of interest
        transcript_processed (str): Directory for processed transcripts
        text_processed (str): Directory to store text file
        documents_text (str): Directory for documents.txt file 
        documents_ids_text (str): Directory to store document id list as a text file
        qa_num (list): List of page numbers denoting the start of question and answer sections
        company_ids_set (list): List of company names
        company_ids_order (list): List of numbers referencing number of file relating to one company

    Returns:
        documents_test_list (list): Returns a list of processed transcript document strings
        document_ids (list): List of document ids
        firm_id (list): List of firm ids
        output_df (df): Dataframe with document information
    """
    # Prepares the documentation
    # Get list of transcripts
    transcript_list, calltimes = get_transcipts(firm_score_xlsx, transcript_directory, transcript_selected)
    # Isolates Q&A sections while removing legal disclaimers
    remove_transcript_metadata(transcript_list,qa_num,transcript_selected,transcript_processed)
    # Creates supplementary identification (Changed here to remove files without text files)
    transcript_list, document_ids, firm_id, output_df = create_ids(transcript_list,qa_num, company_ids_set, company_ids_order, documents_ids_text, calltimes)
    # Creates the documents.txt file, documents ids, firm_ids, and dataframe of outputs
    documents_test_list = create_documents_text(transcript_list,transcript_processed, text_processed, documents_text)
    # Saves csv for comparison
    dataframe_file = Path('data/input/results.csv')
    output_df.to_csv(dataframe_file)
    return documents_test_list, document_ids, firm_id, output_df

def perform_stanford_nlp():
    """Executes Stanford NLP algorithm on processed documentation via 
    """
    print("Implementing Stanford NLP...")
    # Creates variables and directories in global options
    exec(open("global_options.py").read())
    # Step 1: Use `python parse.py` to use Stanford CoreNLP to parse the raw documents.
    exec(open("parse.py").read())
    # Step 2: Use `python clean_and_train.py` to clean, remove stopwords, and named entities in parsed `documents.txt` 
    exec(open("clean_and_train.py").read())
    # Step 3: Use `python create_dict.py` to create the expanded dictionary.
    exec(open("create_dict.py").read())
    # Step 4: Use `python score.py` to score the documents.
    exec(open("score.py").read())
    # Step 5: Use `python aggregate_firms.py` to aggregate the scores to the firm-time level.
    exec(open("aggregate_firms.py").read())
    return

def compare_results(results,output_scores):
    """Creates comparison excel sheets with helens results

    Args:
        results (str): Directory to the document id files
        output_scores (str): Directory for scoring sheets
    """
    # Load in the results
    output_df = pd.read_csv(results)
    # Set directories
    tf = 'firm_scores_TF.csv'
    tfidf = 'firm_scores_TFIDF.csv'
    wfidf = 'firm_scores_WFIDF.csv'
    helen_results = 'outputs/scores/firm_score_helen.xlsx'
    firm_scores_tf = Path(output_scores+'/'+tf)
    firm_scores_tfidf = Path(output_scores+'/'+tfidf)
    firm_scores_wfidf = Path(output_scores+'/'+wfidf)
    helen_results = Path(helen_results)
    # Read csv and excel files
    firm_scores_tf_df = pd.read_csv(firm_scores_tf)
    firm_scores_tfidf_df = pd.read_csv(firm_scores_tfidf)
    firm_scores_wfidf_df = pd.read_csv(firm_scores_wfidf)
    helen_results = pd.read_excel(helen_results)
    # Merge results with dataframes for comparison
    target_df = firm_scores_tfidf_df
    user_results_df = pd.merge(output_df,  target_df,how = 'left',on = output_df.index)
    comparison_df = pd.merge(user_results_df,helen_results,how = 'left',on = ['document_id'])
    print('Please enter a filename')
    filename = input()
    # Save comparison csv
    file_string = 'outputs/comparisons'+'/'+filename+'.xlsx'
    comparison_df.to_excel(file_string)
    return

# Inputs - established all the directories for the locations
# Inputs for processing
firm_score_xlsx = 'data/input/option-1/1.firm_score.xlsx'
transcript_directory = 'data/input/transcripts'
transcript_selected = 'data/raw/selected_transcripts'
transcript_processed = 'data/processed/processed_transcripts'
text_processed = 'data/processed/processed_text'
documents_text = 'data/input/documents.txt'
documents_ids_text = 'data/input/document_ids.txt'
# Creates array of pages numbers indicating the start of the Q&A section for each PDF
# Note: This is labourous but necessary. Values of 4 indicate no Q&A section in the document,
# starting at the presentation section
air_nz_num=[8,10,10,7,8,10,8,11,8,8,8]
aia_num = [4,4,12,12,12,9,10,15,11,10,10,10,10] # Changed to 4
anz_num = [14,6,10,11,11,13,11,13,11,10,11,13,13,8,7,8,7,10,11,12,13,11,12,10,11,11,13,12,11,12,8]
bql_num = [24,12,10,11,11,12,11,11,14,11,9,16,12,13,16,14,13,12,13,10,10,11,12,12,12,13,8]
bab_num = [4,10,10,12,11,10,10,10,14,15,10,10,10,12,10,10,12,12,15,15,9,10]
cba_num = [5,11,11,12,12,11,12,11,10,10,4,11,11,10,11,11,11,10,12,12,11,12,12,12,6,6,6,7,8] # Changed to 4 (29)
ce_num = [8,4] # Changed to 4
fph_num = [9,8,9,8,9,8,8,7,8,8,8]
fbu_num = [12,10,11,10,9,10,10,9,10,11,9]
gpt_num = [10,9]
il_num = [15,15,15,15,13,14,13,12,13,15,15,16,13]
kip_num = [11,10]
nab_num = [12,4,4,13,12,14,10,18,15,9,10,11,11,12,13,13,10,12,15,10,9,10,10,12,11,9,8,15,7,7,6] # Changed to 4 (31)
spk_num = [16,12,12]
tnz_num = [15,14,11,16,14,12,13,9,16]
vec_num = [9,12,9,9,10,9,9,8,12,11,10,9]
wpc_num = [12,19,12,14,14,13,13,11,12,11,11,12,11,11,16,14,12,12,12,11,11,12,10,11,7,8,7,7,7]
# Combines the arrays
qa_num = [air_nz_num,
            aia_num,
            anz_num,
            bql_num,
            bab_num,
            cba_num,
            ce_num,
            fph_num,
            fbu_num,
            gpt_num,
            il_num,
            kip_num,
            nab_num,
            spk_num,
            tnz_num,
            vec_num,
            wpc_num]

qa_num = air_nz_num+aia_num+anz_num+bql_num+bab_num+cba_num+ce_num+fph_num+fbu_num+gpt_num+il_num+kip_num+nab_num+spk_num+tnz_num+ vec_num + wpc_num
# Sets list for company ids
company_ids_set = ['Air New Zealand Limited','Auckland International Airport Limited','Australia New Zealand Banking Group Limited', 'Bank of Queensland Limited','Bendigo and Adelaide Bank Limited','Commonwealth Bank of Australia', 'Contact Energy Ltd','Fisher Paykel Healthcare Corporation Limited','Fletcher Building Ltd','Goodman Property Trust','Infratil Limited','Kiwi Income Property Trust','National Australia Bank Limited','Spark New Zealand Limited','Telecom Corp of New Zealand Ltd','Vector Limited','Westpac Banking Corporation']
company_ids_order = [11,24,55,82,104,133,135,146,157,159,172,174,205,208,217,229,258]
# Inputs for comparison
output_scores = 'outputs/scores'
results = 'data/input/results.csv'
output_word_contributions = 'outputs/scores/word_contributions'
firm_scores_tf = 'outputs/scores/firm_scores_TF.csv'
firm_scores_tfidf = 'outputs/scores/firm_scores_TFIDF.csv'
firm_scores_wfidf = 'outputs/scores/firm_scores_WFIDF.csv'
#####################################################
# Function Calls
# Set binary variables to control function calls
transcript_preparation = False
stanford_nlp_implementation = False
results_comparison = True
# Executes functions based on binary variables
if transcript_preparation == True:
    # Prepare the documents
    print("Preparing documents...")
    documents_test_list, document_ids, firm_id, output_df = prepare_documents(firm_score_xlsx, transcript_directory, transcript_selected, transcript_processed, text_processed, documents_text, documents_ids_text, qa_num, company_ids_set,company_ids_order)
if stanford_nlp_implementation == True:
    # Implements Stanford NLP
    perform_stanford_nlp()
if results_comparison == True:
    print('Comparing results...')
    compare_results(results,output_scores)
# Note: Australia and New Zealand Banking Group Limited - ShareholderAnalyst Call.pdf, Bank of Queensland Ltd. - ShareholderAnalyst Call.pdf
# Commonwealth Bank of Australia - ShareholderAnalyst Call.pdf, Infratil Limited - AnalystInvestor Day.pdf, Infratil Ltd. - AnalystInvestor Day.pdf
# National Australia Bank Limited - ShareholderAnalyst Call.pdf



