#%%
import json
from sqlalchemy import create_engine
import pandas as pd
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

USER = os.getenv('LOCAL_MYSQL_USER')
PASSWD = os.getenv('LOCAL_MYSQL_PASSWD')

SQLALCHEMY_CONNECTION_READ = f"mysql+mysqlconnector://{USER}:{PASSWD}@127.0.0.1:3306/code4bench"
engine_read = create_engine(SQLALCHEMY_CONNECTION_READ)

def get_source_table():
    query = """SELECT * FROM source
    WHERE languages_id IN (1,3,4,6) -- cpp
    AND verdicts_id = 1 -- accepted
    AND isduplicated = 0
    """
    source = pd.read_sql(query, engine_read)
    return source

def get_c_source_table():
    query = """SELECT * FROM source
    WHERE languages_id = 2
    AND verdicts_id = 1 -- accepted
    AND isduplicated = 0
    """
    source = pd.read_sql(query, engine_read)
    return source

def get_source_ids():
    query = """SELECT id FROM source
    WHERE languages_id IN (1,3,4,6) -- cpp
    AND verdicts_id = 1 -- accepted
    AND isduplicated = 0
    """
    source_ids = pd.read_sql(query, engine_read)
    return source_ids

def get_code_pairs(source, factor=1.5, same_factor=1.1, keep_single_subs=True):
    ixs_by_problem_author = source.groupby(['problem', 'author'])['submission'].agg(lambda x: list(x.index))
    pairs = []


    for (problem, author), ixs in tqdm(ixs_by_problem_author.iteritems(), total=len(ixs_by_problem_author)):
        #print(problem, author, ixs)
        df = source.loc[ixs]
        if (not keep_single_subs) and len(df) == 1:
            continue

        for i in range(len(df)):
            for j in range(len(df)):
                r1 = df.iloc[i]
                r2 = df.iloc[j]
                assert r1['author'] == author and r1['problem'] == problem
                assert r2['author'] == author and r2['problem'] == problem
                
                # id is id of submission in source
                # name is index in source df
                r = {
                    'before_source_ix': r1.name,
                    'before_id': r1['id'],
                    'after_source_ix': r2.name,
                    'after_id': r2['id'],
                    'problem': problem,
                    'author': author
                }
                if r1['time'] > factor * r2['time']:
                    r['label'] = 1 # first revision is slower, second revision is faster
                elif r2['time'] > factor * r1['time']:
                    r['label'] = -1 # first revision is faster, second revision is slower
                else:
                    if r1['time'] < same_factor * r2['time']:
                        r['label'] = 0 # none significantly faster
                    elif r2['time'] < same_factor * r1['time']:
                        r['label'] = 0 # none significantly faster
                    else:
                        r['label'] = None # "grey" zone

                if r['label'] is not None:
                    pairs.append(r)
                    
        
    pairs = pd.DataFrame(pairs)
    return pairs


def get_code_pair_times(source):
    ixs_by_problem_author = source.groupby(['problem', 'author'])['submission'].agg(lambda x: list(x.index))
    pairs = []


    for (problem, author), ixs in tqdm(ixs_by_problem_author.iteritems(), total=len(ixs_by_problem_author)):
        #print(problem, author, ixs)
        df = source.loc[ixs]

        for i in range(len(df)):
            for j in range(len(df)):
                r1 = df.iloc[i]
                r2 = df.iloc[j]
                assert r1['author'] == author and r1['problem'] == problem
                assert r2['author'] == author and r2['problem'] == problem
                
                # id is id of submission in source
                # name is index in source df
                r = {
                    'before_source_ix': r1.name,
                    'before_id': r1['id'],
                    'after_source_ix': r2.name,
                    'after_id': r2['id'],
                    'problem': problem,
                    'author': author
                }
                r['time_diff'] = r1['time'] - r2['time']
                r['time_rel'] = (r1['time'] - r2['time']) / (r1['time'] + 1)
                
                
                pairs.append(r)
                    
        
    pairs = pd.DataFrame(pairs)
    return pairs

#%%
lang = 'CPP' # 'C', 'CPP'
prefix = 'cf_c' if lang == 'C' else 'cf_cpp' 
lang, prefix

#%%
if lang == 'C':
    source = get_c_source_table()
else:
    source = get_source_table()
source = source.rename(columns={'problems_id': 'problem'})
len(source)

#%%
source_ids = source[['id']]
source_ids.to_csv(f'data/{prefix}_source_ids.csv')

#%%
(source.loc[source['countline'] < 300, ['id', 'sourceCode', 'author', 'problem']]
    .to_csv(f'data/{prefix}_code.csv')
)
#%%
(source.loc[source['countline'] < 300, ['id', 'author', 'problem', 'time']]
    .to_csv(f'data/{prefix}_times.csv')
)

#%%
factor=1.5
same_factor=1.1
keep_single_subs=False
pairs = get_code_pairs(source, factor, same_factor, keep_single_subs)
pairs.to_csv(f'data/{prefix}_pairs_{factor}_{same_factor}_{keep_single_subs}.csv', index=False)
pairs['label'].value_counts()

#%%
pairs = get_code_pair_times(source)
pairs.to_csv(f'data/{prefix}_pair_times.csv', index=False)

