import argparse
import os
from collections import defaultdict

def load_training_data(train_path):
    """loads training data and returns dict of word frequencies"""
    word_frequencies = defaultdict(int)
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.strip().split():
                word_frequencies[word] += 1
    return word_frequencies

    
def train_wordpiece_tokenizer(word_frequencies, vocab_size):
    """Learn WordPiece vocab with reserved tokens first."""
    
    # initialise vocab 
    vocab = set()
    for word in word_frequencies.keys():
        vocab.update(list(word))
    
    # sort for deterministic
    vocab = sorted(list(vocab))
    RESERVED_TOKENS = ['<pad>', '<unk>', '<s>', '</s>']
    vocab = RESERVED_TOKENS + vocab
    
    splits = {word: [char for char in word] for word in word_frequencies.keys()}
    
    while len(vocab) < vocab_size:
        
        # pair frequencies
        pair_frequencies = defaultdict(int)
        for word, frequency in word_frequencies.items():
            word_split = splits[word]
            for i in range(len(word_split)-1):
                pair = (word_split[i], word_split[i+1])
                pair_frequencies[pair] += frequency
        
        if not pair_frequencies:
            break
        
        # individual tokens frequencies
        token_frequenies = defaultdict(int)
        for word, frequency in word_frequencies.items():
            for token in splits[word]:
                token_frequenies[token] += frequency
                
        # find the merge_pair
        merge_pair = None
        max_score = -21
        for pair, frequency in pair_frequencies.items():
            # ensure tokens exist first
            if token_frequenies.get(pair[0], 0) == 0 or token_frequenies.get(pair[1], 0) == 0:
                continue
            
            # score = freq(a,b) / freq(a) * freq(b)
            score = frequency / (token_frequenies[pair[0]]) * (token_frequenies[pair[1]])
            if score > max_score or (score == max_score and pair < (merge_pair if merge_pair else ())):
                max_score = score
                merge_pair = pair
            
        if merge_pair is None:
            break
        
        # merge the merge pair
        new_token = str(merge_pair[0]) + str(merge_pair[1])
        vocab.append(new_token)
        
        # update splits with new token created
        new_splits = {}
        for word, split in splits.items():
            new_word_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i+1]) == merge_pair:
                    new_word_split.append(new_token)
                    i += 2
                else:
                    new_word_split.append(split[i])
                    i += 1
            new_splits[word] = new_word_split
        splits = new_splits

    # here tokenizer object is just the new vocab after merge
    tokenizer = {'vocab': set(vocab)}
    return vocab, tokenizer
        
        
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
    vocab, tokenizer = train_wordpiece_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize_wp(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize_wp(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
