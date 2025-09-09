import argparse
import os
from collections import defaultdict
import re
import heapq
import math

RESERVED_TOKENS = ['<pad>', '<unk>', '<s>', '</s>']

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

def load_training_data(train_path):
    """loads training data and returns dict of word frequencies"""
    word_frequencies = defaultdict(int)
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.strip().split():
                word_frequencies[word] += 1
    return word_frequencies
    
def train_wp_tokenizer(word_frequencies, vocab_size):
    characters = set()
    for word in word_frequencies:
        characters.update(word)
    initial_vocab = sorted(characters)
    vocab = RESERVED_TOKENS + initial_vocab
    vocab = list(dict.fromkeys(vocab))  
    #print(f"initi vocab size: {len(vocab)}")  
    word_splits = {}
    for word in word_frequencies:
        word_splits[word] = SplitWord(list(word))

    pair_frequencies = defaultdict(int)
    pair_positions = defaultdict(set)

    for word, word_split in word_splits.items():
        node = word_split.head
        while node and node.next:
            pair = (node.token_index, node.next.token_index)
            pair_frequencies[pair] += word_frequencies[word]
            pair_positions[pair].add((word, node))
            node = node.next

    tokenwise_frequencies = defaultdict(int)
    for word, frequency in word_frequencies.items():
        for token in list(word):
            tokenwise_frequencies[token] += frequency

    def get_score(pair):
        freq_ab = pair_frequencies.get(pair, 0)
        if freq_ab == 0:
            return -math.inf
        freq_a = tokenwise_frequencies.get(pair[0], 0)
        freq_b = tokenwise_frequencies.get(pair[1], 0)
        if freq_a == 0 or freq_b == 0:
            return -math.inf
        return math.log(freq_ab) - math.log(freq_a) - math.log(freq_b)

    heap = []
    merge_count = 0
    for pair in list(pair_frequencies.keys()):
        score = get_score(pair)
        if score > -math.inf:
            heapq.heappush(heap, (-score, merge_count, pair))
            merge_count += 1

    while len(vocab) < vocab_size and heap:
        neg_score, _, pair = heapq.heappop(heap)
        if pair_frequencies.get(pair, 0) == 0:
            continue

        new_token = pair[0] + pair[1]
        if new_token in vocab:
            continue
        vocab.append(new_token)

        tokenwise_frequencies[new_token] = pair_frequencies[pair]

        pair_instances = list(pair_positions[pair])
        pair_positions[pair].clear()
        pair_frequencies[pair] = 0

        changed_pairs = set()
        for word, left_node in pair_instances:
            splt = word_splits[word]
            if not left_node.next or (left_node.token_index, left_node.next.token_index) != pair:
                continue
            prev_node = left_node.prev
            next_node = left_node.next.next

            if prev_node:
                old_pair = (prev_node.token_index, pair[0])
                if old_pair in pair_frequencies:
                    pair_frequencies[old_pair] -= word_frequencies[word]
                    pair_positions[old_pair].discard((word, prev_node))
                    changed_pairs.add(old_pair)

            if next_node:
                old_pair = (pair[1], next_node.token_index)
                if old_pair in pair_frequencies:
                    pair_frequencies[old_pair] -= word_frequencies[word]
                    pair_positions[old_pair].discard((word, left_node.next))
                    changed_pairs.add(old_pair)

            new_node = splt.merge_tokens(left_node, new_token)

            if new_node.prev:
                new_pair = (new_node.prev.token_index, new_token)
                pair_frequencies[new_pair] += word_frequencies[word]
                pair_positions[new_pair].add((word, new_node.prev))
                changed_pairs.add(new_pair)

            if new_node.next:
                new_pair = (new_token, new_node.next.token_index)
                pair_frequencies[new_pair] += word_frequencies[word]
                pair_positions[new_pair].add((word, new_node))
                changed_pairs.add(new_pair)

        for changed in changed_pairs:
            if pair_frequencies.get(changed, 0) > 0:
                score = get_score(changed)
                if score > -math.inf:
                    heapq.heappush(heap, (-score, merge_count, changed))
                    merge_count += 1

    #print(f"final vocav size: {len(vocab)}")
    tokenizer = {'vocab': set(vocab)}
    return set(vocab), tokenizer

        
def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_wp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def tokenize_wp(text, tokenizer):
    vocab_set = tokenizer['vocab']
    words = text.strip().split()
    all_tokens = []

    for word in words:
        if not word:
            continue
        
        current_tokens = []
        sub_word = word
        while sub_word:
            # find longest subword prefix in the vocav
            best_sub = ""
            for i in range(len(sub_word), 0, -1):
                if sub_word[:i] in vocab_set:
                    best_sub = sub_word[:i]
                    break
            
            if best_sub:
                current_tokens.append(best_sub)
                sub_word = sub_word[len(best_sub):]
            else:
                # if no prefix found, mark as unknown and stop
                all_tokens.append('<unk>')
                current_tokens = [] # clear all current tokens
                break
        
        if current_tokens:
            # add "##" to subtokens that are not the first piece of the word
            final_word_tokens = [current_tokens[0]]
            for tok in current_tokens[1:]:
                final_word_tokens.append("##" + tok)
            all_tokens.extend(final_word_tokens)

    return all_tokens


def detokenize_wp(tokens, tokenizer=None):
    text = ""
    for token in tokens:
        if token == '<unk>':
            if text:
                text += " "
            text += token
        elif token.startswith("##"):
            text += token[2:]
        else:
            if text:
                text += " "
            text += token
    return text

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_wp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_wp_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "240354"

    train_text = load_training_data(args.train)
    vocab, tokenizer = train_wp_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize_wp(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize_wp(tokens, tokenizer)
    save_detokenized(detok_text, rollno)



