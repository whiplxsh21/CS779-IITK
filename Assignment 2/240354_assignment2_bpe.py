import argparse
import os
import re
from collections import Counter, defaultdict
import heapq

# NOTE: word end token is 256 here and whitesapce is '_'
RESERVED_TOKENS = ['<pad>', '<unk>', '<s>', '</s>']
EOW_MARKER = '256'

# defining the custom data structures used
class Node:
    """double linked list node (represents a token in a word)"""
    def __init__(self, token_index):
        self.token_index = token_index
        self.prev = None
        self.next = None
        
class SplitWord:
    """double linked list representing word's current split"""
    def __init__(self, tokens):
        self.head = None
        self.tail = None
        self.nodes = []
        prev = None
        for token in tokens:
            node = Node(token)
            node.prev = prev
            if prev:
                prev.next = node
            else:
                self.head = node
            prev = node
            self.nodes.append(node)
        self.tail = prev
    
    def tokens_list(self):
        """returns list of tokens"""
        toktokitoki = []
        node = self.head
        while node:
            toktokitoki.append(node.token_index)
            node = node.next
        return toktokitoki
    
    def merge_tokens(self, left, new_token_index):
        """merge left node with its next node, creating a new token"""
        right = left.next
        if not right:
            return None
        
        # create new merged node
        new_node = Node(new_token_index)
        new_node.prev = left.prev
        new_node.next = right.next
        
        if left.prev:
            left.prev.next = new_node
        else:
            self.head = new_node
        
        if right.next:
            right.next.prev = new_node
        else:
            self.tail = new_node
        
        # remove old nodes
        left.next = None; left.prev = None
        right.prev = None; right.next = None

        return new_node
              
### starting boilerplate code now

def load_training_data(train_path):
    """loads train data from a txt file"""
    with open(train_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def train_bpe_tokenizer(text, vocab_size):
    """trains bpe using linked list and heap"""
    words = []
    for word in re.findall(r'\S+|\s+', text):
        if not word.isspace():
            words.append(word)
    words_string = []
    for word in words:
        words_string.append(word)
    
    # encoding each word as a list of bytes + word end (256)
    corpus = []
    for word in words_string:
        encoded_word_list = SplitWord(list(word.encode('utf-8')) + [256])
        corpus.append(encoded_word_list)
    
    # initial vocab: reserved and bytes and EOW
    initial_vocab = RESERVED_TOKENS + [str(i) for i in range(257)]
    
    # count and map bigrams
    # default dict with key as (left_token_index, right_token_index) and value frequncy
    bigram_frequencies = defaultdict(int)
    # map it to a set containing (word_index, node)
    bigram_positions = defaultdict(set)
    
    for word_index, word in enumerate(corpus):
        node = word.head
        while node and node.next:
            bigram = (str(node.token_index),str(node.next.token_index))
            bigram_frequencies[bigram] += 1
            bigram_positions[bigram].add((word_index, node))
            node = node.next
            
    # heap to get bigram that is most 
    # the heap stores (-freq, merge_index, bigram)
    # merge_index breaks the ties in insertion order
    
    heap = []
    merge_index_count = 0
    for bigram, frequency in bigram_frequencies.items():
        heapq.heappush(heap, (-frequency, merge_index_count, bigram))
        merge_index_count += 1

    # track latest counts of bigrams for deletion later
    current_counts = dict(bigram_frequencies)
    
    merges = []
    merge_rules = {} # bigram -> new_token
    next_token_index = 257 # first new token
    
    # helper function (updates bigram frequencies after merge)
    def update_bigram_frequency_after_merge(word_index, merge_node):
        # remove the old connected bigrams and add new
        left = merge_node.prev
        right = merge_node.next
        
        # remove right if right exists
        if right:
            old_bigram = (str(merge_node.token_index), str(right.token_index))
            if old_bigram in bigram_positions:
                # hope this works
                try:
                    bigram_positions[old_bigram].remove((word_index, merge_node))
                    bigram_frequencies[old_bigram] -= 1
                except KeyError:
                    pass
                if bigram_frequencies[old_bigram] <= 0:
                    del bigram_frequencies[old_bigram]
                    del bigram_positions[old_bigram]
                current_counts[old_bigram] = bigram_frequencies.get(old_bigram, 0)
        
        # remove left if left exists
        if left:
            old_bigram = (str(left.token_index), str(merge_node.token_index))
            if old_bigram in bigram_positions:
                try:
                    bigram_positions[old_bigram].remove((word_index, left))
                    bigram_frequencies[old_bigram] -= 1
                except KeyError:
                    pass
                if bigram_frequencies[old_bigram] <= 0:
                    del bigram_frequencies[old_bigram]
                    del bigram_positions[old_bigram]
                current_counts[old_bigram] = bigram_frequencies.get(old_bigram, 0)
                
        # in case both left and merge exist, add a new bigram between them
        if left:
            new_bigram = (str(left.token_index), str(merge_node.token_index))
            bigram_frequencies[new_bigram] = bigram_frequencies.get(new_bigram, 0) + 1
            bigram_positions[new_bigram].add((word_index, left))
            current_counts[new_bigram] = bigram_frequencies[new_bigram]
            heapq.heappush(
                heap, 
                (-bigram_frequencies[new_bigram], merge_index_count + 1000000, new_bigram)
            )
            
        # in case both merge and right exist, add a new bigram 
        if right:
            new_bigram = (str(merge_node.token_index), str(right.token_index))
            bigram_frequencies[new_bigram] = bigram_frequencies.get(new_bigram, 0) + 1
            bigram_positions[new_bigram].add((word_index, merge_node))
            current_counts[new_bigram] = bigram_frequencies[new_bigram]
            heapq.heappush(
                heap, 
                (-bigram_frequencies[new_bigram], merge_index_count + 2000000, new_bigram)
            )
    
    vocab = initial_vocab
    # training loop
    while len(vocab) < vocab_size and heap:
        negative_count, _, bigram = heapq.heappop(heap)
        count = -negative_count
        
        # make sure count is current count and > 0
        if current_counts.get(bigram, 0) != count or count == 0:
            continue
        
        # skip reserved token
        if any(tok in RESERVED_TOKENS for tok in bigram):
            continue
            
        # new merged token (whitesoace is '_')
        new_token = bigram[0] + '_' + bigram[1]
        if new_token in vocab:
            continue
        
        vocab.append(new_token)
        merges.append(new_token)
        merge_rules[bigram] = new_token
        
        new_token_index = new_token # token id is a string for now
        
        # key optimization done here: gonna merge all occurences of the bigram here itself in the whole corpus
        bigram_instances = list(bigram_positions[bigram])
        # init again
        bigram_positions[bigram].clear()
        bigram_frequencies[bigram] = 0
        current_counts[bigram] = 0
        
        for word_index, node in bigram_instances:
            word = corpus[word_index]
            # if already merged then skip
            if node.next is None or node.token_index != int(bigram[0]) and node.token_index != int(bigram[1]):
                continue
            if (str(node.token_index), str(node.next.token_index)) != bigram:
                continue
            
            # merge
            merge_node = word.merge_tokens(node, new_token_index)
            update_bigram_frequency_after_merge(word_index, merge_node)
    tokenizer = {
        'vocab': set(vocab),
        'merges': merges,
        'merge_rules': merge_rules
    }
    return vocab, tokenizer


def tokenise_bpe(text, tokenizer):
    words = re.findall(r'\S+|\s+', text)
    tokens = []
    vocab = tokenizer['vocab']
    merges = tokenizer['merges']
    merge_rules = tokenizer['merge_rules']

    for item in words:
        if item.isspace():
            for char in item:
                tokens.append(str(ord(char)))
            continue

        byte_list = list(item.encode('utf-8')) + [256]
        word_array = [str(b) for b in byte_list]

        while True:
            pairs = [(word_array[i], word_array[i + 1]) for i in range(len(word_array) - 1)]
            pair_indices = {p: i for i, p in enumerate(pairs)}
            merged = None
            for merge_pair in merges:
                if merge_pair in pairs:
                    index = pair_indices[merge_pair]
                    merged = merge_pair
                    break
            if not merged:
                break
            new_token = merge_rules[merged]
            word_array = word_array[:index] + [new_token] + word_array[index + 2:]

        tokens.extend([token if token in vocab else '<unk>' for token in word_array])

    return tokens

def detokenize_bpe(tokens, tokenizer):
    vocab = tokenizer['vocab']
    merge_rules = tokenizer['merge_rules']

    def expand_token(token):
        if token.isdigit() and int(token) <= 256:
            return [token]
        elif token in RESERVED_TOKENS:
            return []
        elif '_' in token:
            left, right = token.split('_', 1)
            return expand_token(left) + expand_token(right)
        else:
            return []

    out_bytes = []
    word_bytes = []

    for token in tokens:
        if token in RESERVED_TOKENS:
            continue
        subs = expand_token(token)
        for subtoken in subs:
            if subtoken == EOW_MARKER:
                try:
                    out_bytes.append(
                        bytes([int(b) for b in word_bytes]).decode('utf-8', errors='replace')
                    )
                except Exception:
                    out_bytes.append('')
                word_bytes = []
            else:
                word_bytes.append(subtoken)
    if word_bytes:
        try:
            out_bytes.append(
                bytes([int(b) for b in word_bytes]).decode('utf-8', errors='replace')
            )
        except Exception:
            out_bytes.append('')
    return ''.join(out_bytes)
        
def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_bpe_vocab_{vocab_size}.txt"
    with open(fname, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(f'{token}\n')

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(f'{token}\n')

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(text)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, required=True)
    args = parser.parse_args()

    rollno = "240354"

    train_text = load_training_data(args.train)
    vocab, tokenizer = train_bpe_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, 'r', encoding='utf-8') as f:
        sample_text = (f.read())
    tokens = tokenise_bpe(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize_bpe(tokens, tokenizer)
    save_detokenized(detok_text, rollno)     

