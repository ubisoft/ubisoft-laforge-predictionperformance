import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        # pretrained  embedding
        if pretrained_weight is not None:
            # initialise with word2vec embeddings
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    # node is list of nested lists of token ids, ST-trees at current depth
    # node itself cannot be ST-tree of form [root_token, [...]] but rather is list of them
    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            # each ST-tree has children, except if we set some nodes to -1
            assert False, 'None return'
            return None

        assert type(node[0]) is list

        # BC collects vector representations h of ST-trees
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))

        # group all children by teir position in parent 12-16
        index, children_index = [], []
        current_node, children = [], []
        # from Fig.4.
        # node = [ns_1, ns_2] = [ [root_token, [c_1, c_2, c_3]], [root_token', [c_1', c_2']] ]
        # C = children = [[c_1, c_1'], [c_1, c_2'], [c_3]]
        # CI = children_index = [[1,2], [1,2], [1]] 
        for i in range(size):
            # node[i] is ST-tree ns
            # node[i][0] is root node / block identifier for ST-tree
            if node[i][0] != -1:
                index.append(i) # position of ST-tree ns in batch
                current_node.append(node[i][0])
                temp = node[i][1:] # children of root node
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] != -1: # temp[j][0] is root node of sub ST-tree of ns
                        if len(children_index) <= j:
                            # current sub ST-tree temp[j] has more children then any previous subtree
                            children_index.append([i]) # start new group, record to which ST-tree node[i] ns_i sub-tree j belongs
                            children.append([temp[j]]) # same but instead of index, collect children in same structure
                        else:
                            # current sub ST-tree temp[j] has NOT more children then any previous subtree
                            children_index[j].append(i) # add to existing group, record to which ST-tree node[i] ns_i sub-tree j belongs
                            children[j].append(temp[j]) # same but instead of index, collect children in same structure
                  
            else:
                # does not happen, you could probably use this to ignore specific ST-trees in the AST
                assert False, '-1 token'
                batch_index[i] = -1
        
        # equation 1
        # is done implicitly in equation 2
        # when applied to ns = [root_token] (no subtrees (children) anymore)
        # -> next for loop is skipped
        # -> W_n^T We^T x_n + b_n is computed and returned as bath_current

        # equation 2
        # W_n^T v_n + b_n, v_n = W_e^T x_n
        # W_e is self.embedding, x_n one-hot encoding ~ just looks up the correct column
        # W_n is self.W_c, encoder embedding_dim -> encode_dim, independent of node
        # batch_current is output h
        batch_current = self.W_c(
            # out-of-place dim, index, tensor
            # Copies the elements of tensor into the self tensor by selecting the indices in the order given in index
            batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
            self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            # recursively get h_i for all children
            tree = self.traverse_mul(children[c], batch_children_index)
            # add to h (sum_{i in [1,C]} h_i)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
            else:
                assert False, 'None return'

        # batch_current = F.tanh(batch_current)
        batch_index = [i for i in batch_index if i != -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        # copy h to correct index in batch
        # index_copy returns tensor of shape self.batch_node
        # batch_node is pre-initialised to zeros of shape (self.batchsize, self.encode_dim)
        # so here we essentially bring write self.batch_current (h) to correct shape of self.batch_node (correct positions, 0 everywhere else)
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        # instead of appending here we could iteratively max to output to lower memory cost.
        return batch_current

    # is concatenation of blocks, so list of nested lists of token ids
    # bs is sum(lens) where lens are the lengths of blocks
    def forward(self, x, bs):
        # print("encoder", len(x), bs) # len(x) = bs
        # batch_size here is not number of programs in batch (64),
        # but the sum of the number of ('root') ST-Trees for each program
        # i.e. each program is sequence of encoded ST-trees h_i 
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        # print("batch_node", self.batch_node.shape)
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size))) # called recursively

        # print("node_list", len(self.node_list))
        # for n in self.node_list:
        #     print("\t", n.shape)

        #print(f'{len(self.node_list)=}')
        # max pooling
        if len(self.node_list) > 1000:
            # divide and conquer
            l = len(self.node_list)
            step = 1000
            ms = []
            for i in range(0, l, step):
                j = min(l, i+step)
                nodes = torch.stack(self.node_list[i:j])
                m = torch.max(nodes, 0)[0]
                #print('m', m.shape)
                ms.append(m)
            ms = torch.stack(ms)
            #print('ms', ms.shape)
            out = torch.max(ms, 0)
            out = out[0]
        else:
            # this would be huge tensor -> memory problems
            self.node_list = torch.stack(self.node_list) # shape is (len(node_list), self.batchsize, self.encode_dim)
            # print("node_list.shape", self.node_list.shape)
            out = torch.max(self.node_list, 0)
            out = out[0]
        
        # shape is (self.batch_size, self.encode_dim)
        #print("out", out.shape)
        return out

class ASTNN:
    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    # x is list of block (nested list of indices of tokens)
    # we have format x =[token, [token, [...], [...], ...], [token, ...], ...]
    # token is block identifier (root node) like 'FuncDef', 'Decl', binary operation etc
    # 'Compound' has no children
    def encode(self, x):
        self.hidden = self.init_hidden()
        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = [] # list of nested list of token ids
        for i in range(self.batch_size):
            for j in range(lens[i]): # iterate over block
                encodes.append(x[i][j]) # nested list of token ids (at least [0])
        
        # encodes = x[0] + x[1] + ... concatentation
        # sum(lens) = len(encodes)

        encodes = self.encoder(encodes, sum(lens))
        
        # print(lens, max_len)
        # pad sequences
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            seq.append(encodes[start:end]) # (len(item), encode_dim)
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i])) # (max_len-lens[i], encode_dim)
            start = end
        # seq: (len(item), encode_dim) followed by zeros of shape (max_len-lens[i], encode_dim)
        encodes = torch.cat(seq) # (len(x) * max_len, encode_dim)
        encodes = encodes.view(self.batch_size, max_len, -1)  # (len(x), max_len, encode_dim), with 0 padding on 'right'
        encodes = nn.utils.rnn.pack_padded_sequence(encodes, torch.LongTensor(lens), batch_first=True, enforce_sorted=False)

        # gru
        # batch first, self.hidden = zeros
        gru_out, _ = self.bigru(encodes, self.hidden) # (len(x), max_len, 2*hidden_dim)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, padding_value=-1e9)

        gru_out = torch.transpose(gru_out, 1, 2) # (len(x), 2*hidden_dim, max_len)

        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2) # (len(x), 2*hidden_dim)

        return gru_out


class BatchProgramClassifier(nn.Module, ASTNN):
    # def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
        super(BatchProgramClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size
        #class "BatchTreeEncoder"
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)

        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        self.hidden2domain = nn.Linear(self.hidden_dim * 2, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, alpha=None):
        gru_out = self.encode(x)

        y = self.hidden2label(gru_out)

        if alpha is None:
            return y

        reverse_features = ReverseLayerF.apply(gru_out, alpha)
        out_domain = self.hidden2domain(reverse_features)

        return y, out_domain



class BatchProgramComparator(nn.Module, ASTNN):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
        super(BatchProgramComparator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)
        
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        # if self.label_size == 2:
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        self.hidden2domain = nn.Linear(self.hidden_dim * 2, 2)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x1, x2, alpha=None):

        lvec, rvec = self.encode(x1), self.encode(x2)

        if self.label_size == 2:
            features = torch.abs(torch.add(lvec, -rvec))
            y = torch.sigmoid(self.hidden2label(features))
        else:
            # c = torch.cat((lvec, rvec), dim=-1)
            features = torch.add(lvec, -rvec)
            y = self.hidden2label(features)

        if alpha is None:
            return y

        reverse_features = ReverseLayerF.apply(features, alpha)
        out_domain = self.hidden2domain(reverse_features)
        
        return y, out_domain


class BatchProgramRegressor(nn.Module, ASTNN):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, batch_size, use_gpu=True, pretrained_weight=None):
        super(BatchProgramRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)

        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        self.hidden2val = nn.Linear(self.hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        gru_out = self.encode(x)
        
        y = self.hidden2val(gru_out)

        return y
