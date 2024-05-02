import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

dbname = 'cldb'
user = 'root'
password = 'password'
host = 'localhost'
port = '5432'


engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')
Session = sessionmaker(bind=engine)

def update_data(tablename:str, df:pd.DataFrame):
    df.to_sql(tablename, con=engine, if_exists='append', index=False)

def query_data(query:str):
    return pd.read_sql(query, con=engine)