import argparse
import re
import unicodedata
import heapq
from collections import defaultdict
import numpy as np

RESERVED_TOKENS = ['<pad>', '<unk>', '<s>', '</s>']
EOW_MARKER = '256'

def load_training_data(train_path):
    """Load full training text as string."""
    with open(train_path, 'r', encoding='utf-8') as f:
        return f.read()


def normalize_text(text):
    """Apply Unicode NFKC normalization and lowercasing."""
    return unicodedata.normalize("NFKC", text).lower()


def preprocess_text(text):
    """replace spaces with '▁' marker and collapse multiple spaces."""
    text = " ".join(text.split())
    return text.replace(" ", "▁")


# defining the custom data structures used
class Node:
    """double linked list node (represents a token in a word)"""
    def __init__(self, token):
        self.token = token
        self.prev = None
        self.next = None
        
class CorpusWord:
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
    
    def merge_pair(self, left, new_token_index):
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


def train_sp_tokenizer(text, vocab_size):
    text = normalize_text(text)
    text = preprocess_text(text)


    # split corpus wherever '_' is present (space marker)
    raw_words = [w for w in re.findall(r'▁[^▁]+', text)]
    if not raw_words:
        raw_words = [text]


    corpus = []
    for w in raw_words:
        byte_list = list(w.encode('utf-8')) + [256]  #(EOW token)
        corpus.append(CorpusWord([str(b) for b in byte_list]))


    vocab = RESERVED_TOKENS + [str(i) for i in range(257)]

    # bigram frequencies and positions
    bigram_frequencies = defaultdict(int)
    bigram_positions = defaultdict(set)

    for i, word in enumerate(corpus):
        node = word.head
        while node and node.next:
            bg = (node.token, node.next.token)
            bigram_frequencies[bg] += 1
            bigram_positions[bg].add((i, node))
            node = node.next

    # heap building
    heap = []
    merge_order = 0
    latest_counts = dict(bigram_frequencies)
    for bg, freq in bigram_frequencies.items():
        heapq.heappush(heap, (-freq, merge_order, bg))
        merge_order += 1

    merges = []
    merge_rules = {}
    next_token_id = 257  

    def update_bigram_counts(word_idx, merged_node):
        left_node = merged_node.prev
        right_node = merged_node.next

        # remove old bigrams
        if right_node:
            bg = (merged_node.token, right_node.token)
            if bg in bigram_positions:
                bigram_positions[bg].discard((word_idx, merged_node))
                bigram_frequencies[bg] -= 1
                if bigram_frequencies[bg] <= 0:
                    del bigram_frequencies[bg]
                    del bigram_positions[bg]
                latest_counts[bg] = bigram_frequencies.get(bg, 0)

        if left_node:
            bg = (left_node.token, merged_node.token)
            if bg in bigram_positions:
                bigram_positions[bg].discard((word_idx, left_node))
                bigram_frequencies[bg] -= 1
                if bigram_frequencies[bg] <= 0:
                    del bigram_frequencies[bg]
                    del bigram_positions[bg]
                latest_counts[bg] = bigram_frequencies.get(bg, 0)

        # add new bigrams
        if left_node:
            new_bg = (left_node.token, merged_node.token)
            bigram_frequencies[new_bg] = bigram_frequencies.get(new_bg, 0) + 1
            bigram_positions[new_bg].add((word_idx, left_node))
            latest_counts[new_bg] = bigram_frequencies[new_bg]
            heapq.heappush(heap, (-bigram_frequencies[new_bg], merge_order + 1000000, new_bg))
        if right_node:
            new_bg = (merged_node.token, right_node.token)
            bigram_frequencies[new_bg] = bigram_frequencies.get(new_bg, 0) + 1
            bigram_positions[new_bg].add((word_idx, merged_node))
            latest_counts[new_bg] = bigram_frequencies[new_bg]
            heapq.heappush(heap, (-bigram_frequencies[new_bg], merge_order + 2000000, new_bg))

    while len(vocab) < vocab_size and heap:
        neg_freq, _, bg = heapq.heappop(heap)
        freq = -neg_freq

        if freq == 0 or latest_counts.get(bg, 0) != freq:
            continue
        if bg[0] in vocab[:4] or bg[1] in vocab[:4]:
            continue

        new_token = bg[0] + '_' + bg[1]
        if new_token in vocab:
            continue

        vocab.append(new_token)
        merges.append(bg)
        merge_rules[bg] = new_token

        occurrences = list(bigram_positions[bg])
        bigram_positions[bg].clear()
        bigram_frequencies[bg] = 0
        latest_counts[bg] = 0

        for word_idx, left_node in occurrences:
            word = corpus[word_idx]
            if not left_node.next or (left_node.token, left_node.next.token) != bg:
                continue
            merged_node = word.merge_pair(left_node, new_token)
            update_bigram_counts(word_idx, merged_node)

    tokenizer = {
        'vocab': set(vocab),
        'merges': merges,
        'merge_rules': merge_rules,
    }
    return vocab, tokenizer


def tokenize(text, tokenizer, seed=42):

    text = preprocess_text(text)
    tokens_list = []

    words = [m.group(0) for m in re.finditer(r'▁[^▁]+', text)]
    if not words:
        words = [text]

    vocab = tokenizer['vocab']
    merges = tokenizer['merges']
    merge_rules = tokenizer['merge_rules']

    for w in words:
        byte_list = list(w.encode('utf-8')) + [256]
        word_tokens = [str(b) for b in byte_list]
        
        # merge greedily
        while True:
            pairs = [(word_tokens[i], word_tokens[i+1]) for i in range(len(word_tokens)-1)]
            pair_indices = {p: i for i, p in enumerate(pairs)}
            merged_key = None
            for merge_pair in merges:
                if merge_pair in pairs:
                    merged_key = merge_pair
                    break
            if not merged_key:
                break
            idx = pair_indices[merged_key]
            word_tokens = word_tokens[:idx] + [merge_rules[merged_key]] + word_tokens[idx+2:]
        
        tokens_list.extend(word_tokens)

    return tokens_list


def detokenize(tokens_list, tokenizer):
    vocab = tokenizer['vocab']

    def expand_token(token):
        if token.isdigit() and int(token) <= 256:
            return [token]
        if token in RESERVED_TOKENS:
            return []
        if '_' in token:
            left, right = token.split('_', 1)
            return expand_token(left) + expand_token(right)
        return []

    out_chars = []
    word_bytes = []

    for token in tokens_list:
        if token in RESERVED_TOKENS:
            continue
        subtokens = expand_token(token)
        for st in subtokens:
            if st == EOW_MARKER:
                try:
                    char = bytes([int(b) for b in word_bytes]).decode('utf-8', errors='replace')
                except Exception:
                    char = ''
                out_chars.append(char)
                word_bytes = []
            else:
                word_bytes.append(st)
    if word_bytes:
        try:
            char = bytes([int(b) for b in word_bytes]).decode('utf-8', errors='replace')
        except Exception:
            char = ''
        out_chars.append(char)

    text = ''.join(out_chars)
    text = text.replace('▁', ' ')
    return text



def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_sp_vocab_{vocab_size}.txt"
    with open(fname, 'w', encoding='utf-8') as f:
        for tkn in vocab:
            f.write(tkn + '\n')


def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_sp_tokens.txt"
    with open(fname, 'w', encoding='utf-8') as f:
        for tkn in tokens:
            f.write(tkn + '\n')


def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_sp_detokenized.txt"
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
    train_text = normalize_text(train_text)
    vocab, tokenizer = train_sp_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, 'r', encoding='utf-8') as f:
        sample_text = normalize_text(f.read())
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
