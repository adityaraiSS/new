# Dependency Imports #
# import os
import time
import numpy as np
import pandas as pd
import pymysql
from dateutil.relativedelta import relativedelta
from datetime import date
from PIL import Image
import boto3
import torch
# from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
# import joblib
import json
import pyarrow.parquet as pq
import pyarrow as pa
import pymongo
from urllib.parse import quote_plus
import warnings
warnings.filterwarnings("ignore")
import psutil
# import sys
import pickle_loader as pl

logger = None

# Fetching customer info from MySQL tables
## Age Calculator from run_date
def calculate_age(brth_date, run_date):
    year1, month1, day1 = map(int, run_date.split("-"))
    year0, month0, day0 = map(int, brth_date.split("-"))
    run_date = date(year1, month1, day1)
    brth_date = date(year0, month0, day0)
    age = relativedelta(run_date, brth_date)
    return age.years

## Cust Info Fetch: [customer_code, date_of_birth, image_path, execution_date, age, age_bin]
def cust_info_fetch(customer_code, mysql_con):
    try:
        try:
            # Creating a cursor object
            cursor = mysql_con.cursor()
            sql_query = f'''
            select customer_code, date_of_birth
            from tbl_subscribers
            where customer_code = {customer_code}
            '''
            cursor.execute(sql_query)
            # Fetch all the rows
            rows = cursor.fetchall()
            # Get column names from the cursor description
            column_names = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=column_names)
        finally:
            # Close the cursor and the connection
            cursor.close()
            mysql_con.close()
        
        run_date = date.today().isoformat()
        logger.info("----df---- %s", df)
        df['execution_date'] = run_date
        df['age'] = df.apply(lambda row: calculate_age(row['date_of_birth'], run_date), axis=1)
        df['age_bin'] = pd.cut(df['age'], bins=[0,24,34,100])
        return df
    except Exception as e:
        logger.error("-----Error1----- %s", e)
        return None

def load_text_descriptors_from_s3(s3_path, fallback_path, bucket_name):
    """
    Load text descriptors from S3 CSV file with fallback option
    Args:
        s3_path (str): Primary S3 path to CSV file
        fallback_path (str): Fallback S3 path to CSV file
        bucket_name (str): S3 bucket name
    Returns:
        tuple: (text_descriptions, high_thresholds, low_thresholds)
    """
    s3 = boto3.client('s3', region_name="ap-south-1")
    
    # Try primary path first
    try:
        response = s3.get_object(Bucket=bucket_name, Key=s3_path)
        csv_data = response['Body'].read().decode('utf-8')
        df = pd.read_csv(BytesIO(csv_data.encode()))
        logger.info("Text descriptors loaded from primary path: %s", s3_path)
    except (ClientError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.warning("Failed to load from primary path: %s, trying fallback", e)
        # Try fallback path
        try:
            response = s3.get_object(Bucket=bucket_name, Key=fallback_path)
            csv_data = response['Body'].read().decode('utf-8')
            df = pd.read_csv(BytesIO(csv_data.encode()))
            logger.info("Text descriptors loaded from fallback path: %s", fallback_path)
        except (ClientError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.error("Failed to load text descriptors from both paths: %s", e)
            raise RuntimeError(f"Unable to load text descriptors from both primary path '{s3_path}' and fallback path '{fallback_path}' in bucket '{bucket_name}'") from e
    
    # Extract text descriptors and thresholds
    text_descriptions = df['text_descriptor'].tolist()
    
    # Load thresholds if available, otherwise use 0
    high_thresholds = df.get('high_threshold', [0] * len(text_descriptions)).tolist()
    low_thresholds = df.get('low_threshold', [0] * len(text_descriptions)).tolist()
    
    import gc
    del response
    del csv_data
    del df
    gc.collect()
    
    return text_descriptions, high_thresholds, low_thresholds

# CLIP Model Inference on Image from S3
## Load the saved model and processor
# def load_model_and_processor(clip_model_dir):
#     try:
#         model = CLIPModel.from_pretrained(clip_model_dir)
#         processor = CLIPProcessor.from_pretrained(clip_model_dir)
#         return model, processor
#     except Exception as e:
#         logger.error("-----Error2----- %s", e)
#         return None, None

## Function to get CLIP Embedding Vector
def get_image_embedding(image_data):
    """
    Takes in image data, and returns the embedding vector.
    Args:
        - image_data: PIL Image or path to image file.
        - model: Loaded CLIP model.
        - processor: Loaded CLIP processor.
        - device: Device to run the model on, default is "cuda" if available.
    Returns:
        - Image embedding as a 1D numpy array.
    """
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        # Preprocess the image
        inputs = pl.processor(images=image, return_tensors="pt").to(pl.device)
        # Get image embedding
        with torch.no_grad():
            image_features = pl.model.get_image_features(**inputs)
        # Convert to numpy array and flatten
        image_embedding = image_features.cpu().numpy().flatten()
        
        import gc
        del image
        del inputs
        del image_features
        gc.collect()
        
        return image_embedding
    except Exception as e:
        logger.error("-----Error3----- %s", e)
        return None

def process_image_path(image_path, bucket_name):
    """
    Process a single image path to fetch the image, generate description, and handle errors if they occur.
    Args:
        image_path (str): Path to the image in the S3 bucket.
        bucket_name (str): S3 bucket name.
    Returns:
        pd.Series: A Series containing the results (or None if there's an error).
    """
    try:
        s3 = boto3.client('s3', region_name="ap-south-1")
        start_time = time.time()
        # Adjust image path
        selfie_img_path = image_path[1:] if image_path[0] == '/' else image_path
        logger.info("selfie_img_path - %s", selfie_img_path)
        # Get image from S3
        response = s3.get_object(Bucket=bucket_name, Key=selfie_img_path)
        image_data = response['Body'].read()
        # Process image and get embedding
        image_embedding = get_image_embedding(image_data)
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        import gc
        del response
        del image_data
        gc.collect()
        
        # Return the results as pd.Series
        return pd.Series({
            'embedding': image_embedding.tolist(),
            'processing_time': processing_time
        })
    except Exception as e:
        logger.error("-----Error4----- %s", e)
        # Return None values in case of an error
        return pd.Series({
            'embedding': None,
            'processing_time': None
        })

## S3 Parquet Table Storing: [customer_code, clip_embedding_vector]
def write_parquet_to_s3(bucket, key_prefix, df):
    try:
        dataframe = df[['customer_code', 'embedding']].copy()
        # Convert list to string for embedding vector
        if type(dataframe.embedding.iloc[0])==list:
            dataframe["embedding"] = dataframe["embedding"].apply(json.dumps)
        # Create the filename with path for s3
        filename = f"{key_prefix}/run_date={df.execution_date.iloc[0]}/{df.customer_code.iloc[0]}_{df.filename.iloc[0]}.parquet"
        # Create a PyArrow Table from the DataFrame
        table = pa.Table.from_pandas(dataframe)
        # Write to an in-memory buffer
        buffer = BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)
        # Upload the buffer to S3 using boto3
        s3_client = boto3.client("s3")
        s3_client.put_object(Bucket=bucket, Key=filename, Body=buffer.getvalue())
        logger.info(f"Parquet file successfully written to s3://{bucket}/{filename}")
        
        import gc
        del dataframe
        del filename
        del table
        del buffer
        gc.collect()
        
    except Exception as e:
        logger.error("-----write_parquet_to_s3----- %s", e)

# AgeSegmented Logistic Inference on Scaled Embedding Vector
## Function to parse list embeddings back to numpy arrays
def parse_embedding(list_embedding):
    try:
        return np.array(list_embedding)
    except Exception as e:
        logger.error("-----Error5----- %s", e)
        return None

def logistic_inference(customer_code, embedding, age_bin):
    try:
        embedding = parse_embedding(embedding)
        if str(age_bin)=='(0, 24]':
            embedding_scaled = pl.scaler1.transform(embedding.reshape(1, -1))
            logistic_probability = pl.best_logistic1.predict_proba(embedding_scaled)[:, 1]
            logistic_probability_bkt = pd.cut(logistic_probability, bins=pl.decile_trn_bins1, labels=False, include_lowest=True)
        elif str(age_bin)=='(24, 34]':
            embedding_scaled = pl.scaler2.transform(embedding.reshape(1, -1))
            logistic_probability = pl.best_logistic2.predict_proba(embedding_scaled)[:, 1]
            logistic_probability_bkt = pd.cut(logistic_probability, bins=pl.decile_trn_bins2, labels=False, include_lowest=True)
        elif str(age_bin)=='(34, 100]':
            embedding_scaled = pl.scaler3.transform(embedding.reshape(1, -1))
            logistic_probability = pl.best_logistic3.predict_proba(embedding_scaled)[:, 1]
            logistic_probability_bkt = pd.cut(logistic_probability, bins=pl.decile_trn_bins3, labels=False, include_lowest=True)
        else:
            seg = None
        
        import gc
        del embedding
        gc.collect()
        
        # Return the results as pd.Series
        return pd.Series({
            'logistic_prediction_probability': logistic_probability[0],
            'decile_bkt': logistic_probability_bkt[0]
        })
    except Exception as e:
        logger.error("-----Error6----- %s", e)
        return pd.Series({
            'logistic_prediction_probability': None,
            'decile_bkt': None
        })

# Feature Prompt Similarity Analysis on Embedding Vector
def get_text_embedding(text):
    """
    Generate text embedding for a given text using the CLIP model.
    Args:
        text (str): The input text to embed.
    Returns:
        np.ndarray: The text embedding as a numpy array.
    """
    try:
        inputs = pl.processor(text=[text], return_tensors="pt", padding=True).to(pl.device)
        with torch.no_grad():
            text_features = pl.model.get_text_features(**inputs)
        
        import gc
        del inputs
        gc.collect()
        
        return text_features.cpu().numpy().flatten()
    except Exception as e:
        logger.error("-----Error7----- %s", e)
        return None

def compute_similarity(image_embedding, text_embedding):
    """
    Compute cosine similarity between an image embedding and a text embedding.
    Args:
        image_embedding (np.ndarray): The image embedding.
        text_embedding (np.ndarray): The text embedding.
    Returns:
        float: The cosine similarity score.
    """
    return cosine_similarity(image_embedding.reshape(1, -1), text_embedding.reshape(1, -1))[0][0]

def add_text_similarity_columns(img_embedding, text_descriptions):
    """
    Add columns to the DataFrame with cosine similarity scores for given text descriptions.
    Args:
        img_embedding (list): 'embedding' column containing image embedding.
        text_descriptions (list): List of text descriptions to compare with.
    Returns:
        pd.Series: New similarity columns.
    """
    try:
        img_embedding = parse_embedding(img_embedding)
        col_names = []
        sim_values = []
        
        for text in text_descriptions:
            # Generate the text embedding for the given text
            text_embedding = get_text_embedding(text)
            # Compute similarity for each text
            similarity_value = compute_similarity(img_embedding, text_embedding)
            sim_values.append(similarity_value)
            # Add the similarity column
            col_names.append(f"sim_{text.replace(' ', '_')}")
        
        import gc
        del img_embedding
        del text_embedding
        del text_descriptions
        del similarity_value
        gc.collect()
        
        return pd.Series(sim_values, index=col_names)
    except Exception as e:
        logger.error("-----Error8----- %s", e)
        return None

def generate_flags_from_similarity(df, text_descriptions, high_thresholds, low_thresholds):
    """
    Generate flag columns based on similarity scores and thresholds
    Args:
        df (DataFrame): DataFrame containing similarity columns
        text_descriptions (list): List of text descriptions
        high_thresholds (list): List of high thresholds for each descriptor
        low_thresholds (list): List of low thresholds for each descriptor
    Returns:
        DataFrame: DataFrame with additional flag columns
    """
    try:
        df_flags = df.copy()
        
        for i, text in enumerate(text_descriptions):
            sim_col = f"sim_{text.replace(' ', '_')}"
            flag_col = f"flag_{text.replace(' ', '_')}"
            
            if sim_col in df_flags.columns:
                high_thresh = high_thresholds[i]
                low_thresh = low_thresholds[i]
                
                # Generate flags: 1 if similarity >= high_threshold, 0 if similarity <= low_threshold, otherwise maintain previous logic
                df_flags[flag_col] = df_flags[sim_col].apply(
                    lambda x: 1 if x >= high_thresh else (0 if x <= low_thresh else (1 if x > (high_thresh + low_thresh) / 2 else 0))
                )
        
        logger.info("Flag columns generated successfully")
        return df_flags
        
    except Exception as e:
        logger.error("-----generate_flags_from_similarity----- %s", e)
        return df

# MongoDB storing logic - Updated version
def save_to_mongodb(pandas_df, collection):
    try:
        # Prepare MongoDB insertion data
        json_doc = pandas_df.drop(columns=["embedding"]).iloc[0].to_dict()
        collection.insert_one(json_doc)
        logger.info("Data successfully stored in MongoDB")
        
        import gc
        del json_doc
        gc.collect()
        
    except Exception as e:
        logger.error("-----save_to_mongodb----- %s", e)

def save_flags_to_mongodb(pandas_df, collection, text_descriptions, high_thresholds, low_thresholds):
    """
    Save only flag data (0/1) to MongoDB collection using similarity scores and thresholds
    Args:
        pandas_df (DataFrame): DataFrame containing similarity columns
        collection: MongoDB collection object
        text_descriptions (list): List of text descriptions
        high_thresholds (list): List of high thresholds for each descriptor
        low_thresholds (list): List of low thresholds for each descriptor
    """
    try:
        # Prepare flag data for MongoDB - only basic info and flags
        flag_doc = {
            'customer_code': pandas_df.iloc[0]['customer_code'],
            'execution_date': pandas_df.iloc[0]['execution_date'],
            'filename': pandas_df.iloc[0]['filename'],
            'image_path': pandas_df.iloc[0]['image_path'],
            'age': pandas_df.iloc[0]['age'],
            'age_bin': pandas_df.iloc[0]['age_bin']
        }
        
        # Add only flag values (0/1) for each text descriptor
        for i, text in enumerate(text_descriptions):
            sim_col = f"sim_{text.replace(' ', '_')}"
            
            if sim_col in pandas_df.columns:
                similarity_score = pandas_df.iloc[0][sim_col]
                high_thresh = high_thresholds[i]
                low_thresh = low_thresholds[i]
                
                # Generate flag names and values
                descriptor_name = text.replace(' ', '_')
                
                # Flag for high threshold
                flag_high = 1 if similarity_score >= high_thresh else 0
                flag_doc[f"flag_high_{descriptor_name}"] = flag_high
                
                # Flag for low threshold  
                flag_low = 1 if similarity_score <= low_thresh else 0
                flag_doc[f"flag_low_{descriptor_name}"] = flag_low
        
        # Insert into MongoDB collection
        collection.insert_one(flag_doc)
        logger.info("Flag data successfully stored in MongoDB collection: selfie_clip_analytics_flags")
        
        import gc
        del flag_doc
        gc.collect()
        
    except Exception as e:
        logger.error("-----save_flags_to_mongodb----- %s", e)

def generate_flags_from_similarity_updated(df, text_descriptions, high_thresholds, low_thresholds):
    """
    Generate flag columns based on similarity scores and thresholds (Updated version)
    Args:
        df (DataFrame): DataFrame containing similarity columns
        text_descriptions (list): List of text descriptions
        high_thresholds (list): List of high thresholds for each descriptor
        low_thresholds (list): List of low thresholds for each descriptor
    Returns:
        DataFrame: DataFrame with additional flag columns
    """
    try:
        df_flags = df.copy()
        
        for i, text in enumerate(text_descriptions):
            sim_col = f"sim_{text.replace(' ', '_')}"
            flag_col = f"flag_{text.replace(' ', '_')}"
            
            if sim_col in df_flags.columns:
                high_thresh = high_thresholds[i]
                low_thresh = low_thresholds[i]
                
                # Generate flags using similarity scores directly
                df_flags[flag_col] = df_flags[sim_col].apply(
                    lambda x: 1 if x >= high_thresh else (0 if x <= low_thresh else (1 if x > (high_thresh + low_thresh) / 2 else 0))
                )
        
        logger.info("Flag columns generated successfully")
        return df_flags
        
    except Exception as e:
        logger.error("-----generate_flags_from_similarity_updated----- %s", e)
        return df

# Updated main function MongoDB section (replace the existing MongoDB section in your main function)
def updated_mongodb_section(df, content, text_descriptions, high_thresholds, low_thresholds):
    """
    Updated MongoDB section to handle both collections:
    - selfie_clip_analytics_resp: stores similarity scores and other analytics data
    - selfie_clip_analytics_flags: stores only flag values (0/1)
    """
    try:
        # Initialize MongoDB client Connection
        if content.get("request_type") == 'test':
            mongo_qa_user = quote_plus("aw_qa_user")
            mongo_qa_pwd = quote_plus("BJ9qK13X1M@1")
            client_analytics = pymongo.MongoClient("mongodb+srv://%s:%s@mongostage.earlysalarystaging.com/ES_Analytics?ssl=false"%(mongo_qa_user, mongo_qa_pwd))
        elif content.get('request_type') == 'prod':
            mongo_prod_user = quote_plus("facematch_user")
            mongo_prod_pwd = quote_plus("QDZR#qFdg5R")
            client_analytics = pymongo.MongoClient("mongodb+srv://%s:%s@mongoprod.internalearlysalary.com/ES_Analytics?ssl=false"%(mongo_prod_user, mongo_prod_pwd))
        
        es_analytics_db = client_analytics["ES_Analytics"]
        
        # Collection 1: Main analytics data with similarity scores
        selfie_clip_analytics_resp = es_analytics_db["selfie_clip_analytics_resp"]
        
        # Collection 2: Only flag data (0/1 values)
        selfie_clip_analytics_flags = es_analytics_db["selfie_clip_analytics_flags"]
        
        # Save main data (including similarity scores) to original collection
        save_to_mongodb(df, selfie_clip_analytics_resp)
        
        # Save only flag data (0/1) to new flags collection
        save_flags_to_mongodb(df, selfie_clip_analytics_flags, text_descriptions, high_thresholds, low_thresholds)
        
        if client_analytics:
            client_analytics.close()
        
        logger.info("Both MongoDB collections updated successfully")
        logger.info("- selfie_clip_analytics_resp: Contains similarity scores and analytics data")
        logger.info("- selfie_clip_analytics_flags: Contains only flag values (0/1)")
        
    except Exception as e:
        logger.error("-----updated_mongodb_section----- %s", e)

# ECS Resource Monitoring
def monitor_resources():
    try:
        # Get the memory usage in bytes
        memory_info = psutil.virtual_memory()
        total_memory = memory_info.total / (1024 ** 3)  # Total system memory in GB
        used_memory = memory_info.used / (1024 ** 3)  # Used memory in GB
        available_memory = memory_info.available / (1024 ** 3)  # Available memory in GB
        memory_percent = memory_info.percent  # Percentage of memory used
        
        logger.info(f"Total Memory: {total_memory:.2f} GB")
        logger.info(f"Used Memory: {used_memory:.2f} GB")
        logger.info(f"Available Memory: {available_memory:.2f} GB")
        logger.info(f"Memory Utilization: {memory_percent}%")
        
        # Get the CPU utilization as a percentage
        system_cpu_utilization = psutil.cpu_percent(interval=0.2)
        logger.info(f"System CPU Utilization: {system_cpu_utilization}%")
        
        import gc
        del memory_info
        del total_memory
        del used_memory
        del available_memory
        del memory_percent
        del system_cpu_utilization
        gc.collect()
        
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes", "r") as f:
            container_memory = int(f.read()) / (1024 ** 3)  # Convert to GB
        logger.info("Container Memory Usage: %s GB", container_memory)
        
    except Exception as e:
        logger.error("-----Resource_Monitoring_Error----- %s", e)

# Main function call on lambda trigger
def main(customer_code, image_path, bucket_name, passed_logger, content, text_descriptor_s3_path, fallback_text_descriptor_path):
    module_status = ""
    try:
        global logger
        logger = passed_logger
        
        # Fetch Customer Info
        mysql_con = pymysql.connect(
            host='es.c5cc4erpgekm.ap-south-1.rds.amazonaws.com',
            user='aw',
            password='Es@aw##$$1',
            db='earlysalary',
            port=7011)
        
        ## Customer to trigger
        module_status = "2_CustInfo_Error"
        df = cust_info_fetch(customer_code, mysql_con)
        df['image_path'] = image_path
        df["filename"] = df["image_path"].str.split('.').str[-2].str.split('/').str[-1]
        
        # Load text descriptors from S3
        module_status = "3_TextDescriptor_Error"
        text_descriptions, high_thresholds, low_thresholds = load_text_descriptors_from_s3(
            text_descriptor_s3_path, fallback_text_descriptor_path, bucket_name
        )
        
        ## Embedding Generation
        module_status = "4_EmbeddingGeneration_Error"
        df[['embedding', 'processing_time']] = df['image_path'].apply(process_image_path, bucket_name=bucket_name)
        
        ## S3 DeltaTable Storing: [customer_code, clip_embedding_vector]
        if content.get("request_type") == 'test':
            model_bucket_name = "analyticsdoc-test"
            key_prefix = "komal/selfie-clip-embeddings/data/embeddings"
        elif content.get('request_type') == 'prod':
            model_bucket_name = "analyticsdoc"
            key_prefix = "hardik/selfie_clip_embeddings/data/embeddings"
        
        ### Storing as list object
        module_status = "5_EmbeddingWriteS3_Error"
        write_parquet_to_s3(model_bucket_name, key_prefix, df)
        
        # CLIP Embedding Implementations
        ## AgeSegmented Logistic Inference on Scaled Embedding Vector
        module_status = "6_LogisticModel_Error"
        df[['logistic_prediction_probability', 'decile_bkt']] = df.apply(lambda row: logistic_inference(row.customer_code, row.embedding, row.age_bin), axis=1)
        df['age_bin'] = df['age_bin'].astype(str)
        
        ## Feature Prompt Similarity Analysis on Embedding Vector
        module_status = "7_FeatureSimilarity_Error"
        col_names = []
        for text in text_descriptions:
            col_names.append(f"sim_{text.replace(' ', '_')}")
        
        df[col_names] = df['embedding'].apply(add_text_similarity_columns, text_descriptions=text_descriptions)
        
        ## Generate Flag Columns
        module_status = "8_FlagGeneration_Error"
        df = generate_flags_from_similarity(df, text_descriptions, high_thresholds, low_thresholds)
        
        # MongoDB storing logic - Updated to handle both collections
        # NEW 
        module_status = "9_SelfieCLIPModule_Executed"
        updated_mongodb_section(df, content, text_descriptions, high_thresholds, low_thresholds)
        
        logger.info("-----ECS Resource Monitoring-----")
        monitor_resources()
        logger.info("Final df - %s: %s, %s: %s", df.columns[0], df.iloc[0, 0], df.columns[-1], df.iloc[0, -1])
        
        import gc
        del df
        try:
            torch.cuda.empty_cache()
            with torch.no_grad():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.error("-----Torch Err----- %s", e)
        gc.collect()
        logger.info("-----Garbage collection Completed !-----")
        
        return({"final_status":"success"})
        
    except Exception as e:
        logger.error("-----Main----- %s", e)
        logger.info("-----ECS Resource Monitoring-----")
        monitor_resources()
        return(module_status)
