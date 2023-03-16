
#%%
import pandas as pd
from gensim.models.word2vec import Word2Vec
from astnn_tree import ASTNode, SingleNode

#%%
def get_sequences(node, sequence):
    current = SingleNode(node)
    sequence.append(current.get_token())
    for _, child in node.children():
        get_sequences(child, sequence)
    if current.get_token().lower() == 'compound':
        sequence.append('End')

def dictionary_and_embedding(trees, output_file, size):
    def trans_to_sequences(ast):
        sequence = []
        get_sequences(ast, sequence)
        return sequence
    corpus = trees['ast'].apply(trans_to_sequences)
    
    str_corpus = corpus.apply(lambda c: ' '.join(c))
    trees['tokenized_ast'] = str_corpus

    w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3, seed=0)
    w2v.save(output_file)
    return w2v

# unnested list of statement trees but in order of occurence in file (by line)
def get_blocks(node, block_seq):
    children = node.children()
    name = node.__class__.__name__
    if name in ['FuncDef', 'If', 'For', 'While', 'DoWhile']:
        block_seq.append(ASTNode(node))
        if name != 'For':
            skip = 1
        else:
            skip = len(children) - 1

        for i in range(skip, len(children)):
            child = children[i][1]
            if child.__class__.__name__ not in ['FuncDef', 'If', 'For', 'While', 'DoWhile', 'Compound']:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
    elif name == 'Compound':
        block_seq.append(ASTNode(name))
        for _, child in node.children():
            if child.__class__.__name__ not in ['If', 'For', 'While', 'DoWhile']:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
        block_seq.append(ASTNode('End'))
    else:
        for _, child in node.children():
            get_blocks(child, block_seq)


def generate_block_seqs(w2v, asts):
    word2vec = w2v.wv
    vocab = word2vec.vocab # wv.vocab in gensim 4.x
    max_token = word2vec.syn0.shape[0] # self.wv.vectors

    # converts tokens to index id and tree structure to nested list structure
    def tree_to_index(node):
        token = node.token
        result = [vocab[token].index if token in vocab else max_token]
        children = node.children
        for child in children:
            result.append(tree_to_index(child))
        return result

    def trans2seq(r):
        blocks = []
        get_blocks(r, blocks) # list of ASTNode statement trees
        #print(blocks)
        tree = []
        for b in blocks:
            # print(b.token)
            btree = tree_to_index(b)
            tree.append(btree)
        return tree
        
    asts['blocks'] = asts['ast'].apply(trans2seq)


#%%
prefix = 'cf_c' # cf_c, cf_tle_c

#%%
asts = pd.read_pickle(f'data/{prefix}_asts.pkl')

# %%
w2v = dictionary_and_embedding(asts, f'data/{prefix}_w2v', 128)

#%%
generate_block_seqs(w2v, asts)

#%%
asts.to_pickle(f'data/{prefix}_asts.pkl')
# columns:
# id
# ast: pycparser object 
# tokenized_ast: str of tokens
# blocks: nested list of ints, input for model

# %%
def contains_minus_one(blocks):
    for b in blocks:
        if isinstance(b, list):
            if contains_minus_one(b):
                return True
        else:
            if b == -1:
                return True
    return False
    
asts['blocks'].apply(contains_minus_one).any()


# %%
