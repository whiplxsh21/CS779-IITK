import argparse
import os
import re
from collections import Counter

# load training data in utf-8 from any txt file given its path and return it in text
def load_training_data(train_path):
    """Load and return raw text for training."""
    with open(train_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


# train the bpe tokenizer using data from train.txt
def train_bpe_tokenizer(text, vocab_size):
    """ 
    core bpe tokenizer training function.
    inputs: `text` and `vocab_size` 
    output: 
        `vocab`: ordered list of tokens (first reserved tokens, then other tokens, then merges)
        `tokenizer`: object (dict object) that holds all info for tokenization necessary.
    """
    # note: vocab_size includes the 4 reserved tokens.
    
    
    # split words for training
    words = re.findall(r'\S+|\s+', text)
    words = [w for w in words if not w.isspace()] # NOT USING SPACES FOR NOW, WILL USE A CUSTGOM TOKEN FOR THEM LATER
    
    # encoding words as list of byte lists + end marker token (256)
    corpus = [list(w.encode('utf-8')) + [256] for w in words] 
    
    # initial vocab: reserved tokens + 0-255 byte characters + end marker token 256
    vocab = ['<pad>', '<unk>', '<s>','</s>'] + [str(i) for i in range(257)]
    
    # merges is a list of tuples of form (a,b) when tokens a,b merge
    merges = []
    merge_rules = {} # this is the actual thing we're training: (a,b) -> new_token
    
    # training loop
    while len(vocab) < vocab_size:
        # bigram frequency Counter
        pair_counter = Counter()

        for word in corpus:
            for i in range(len(word)-1):
                pair = (str(word[i]), str(word[i+1]))
                pair_counter[pair] += 1
        if not pair_counter:
            break
        
        most_common_pair_frequency = max(pair_counter.values())
        # note: could be more than one pair with same frequency
        potential_merges = [pair for pair, frequency in pair_counter.items() if frequency == most_common_pair_frequency]

        best_merge = min(potential_merges, key=lambda token: (token[0], token[1]))
        
        # if this results in a reserved token, skip the merge
        if any(token in ['<pad>', '<unk>', '<s>','</s>'] for token in best_merge):
            break
        
        # making the new token after merge as (a,b) --> 'a_b'
        new_token = best_merge[0] + '_' + best_merge[1]
        
        # merge every occurence of best_merge in all pairs
        new_corpus = []
        for word in corpus:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (str(word[i]), str(word[i+1])) == best_merge:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(str(word[i]) if isinstance(word[i], int) else word[i])
                    i += 1
                
            new_corpus.append(new_word)
        corpus = new_corpus
        vocab.append(new_token)
        merges.append(best_merge)
        merge_rules[best_merge] = new_token
        
        
    tokenizer = {
        'vocab': set(vocab),
        'merges': merges,
        'merge_rules': merge_rules
    }
    
    return vocab, tokenizer

# run the bpe tokenizer trained using `sample-text.txt`
def tokenise_bpe(text, tokenizer):
    
    # load text and initialise tokeniser
    # loading words with whitespace, will handle them with a seperate token later
    words = re.findall(r'\S+|\s+', text)
    
    tokens = []
    vocab = tokenizer["vocab"]
    merges = tokenizer["merges"]
    merge_rules = tokenizer["merge_rules"]
    
    for item in words:
        # whitespace handling
        if item.isspace():
            for char in item:
                tokens.append(str(ord(char)))
                
        else:
            byte_list = list(item.encode("utf-8")) + [256]
            word_array = [str(byte) for byte in byte_list]
            
            # merge pairs (greedy)
            while True:
                pairs = [(word_array[i], word_array[i+1]) for i in range(len(word_array)-1)]
                pair_indices = {p: i for i, p in enumerate(pairs)}
                merged = None
                for byte_pair in merges:
                    if byte_pair in pairs:
                        index = pair_indices[byte_pair]
                        merged = byte_pair
                        break
                if not merged:
                    break
                new_token = merge_rules[merged]
                word_array = word_array[:index] + [new_token] + word_array[index+2:]
                
            # replace words not in current vocab with <unk>
            tokens.extend([token if token in vocab else "<unk>" for token in word_array])
    
    return tokens


# detokenize the text, given the tokens list right now and trained tokenizer
def detokenize_bpe(tokens, tokenizer):
    vocab = tokenizer["vocab"]
    merge_rules = tokenizer["merge_rules"]
    RESERVED_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
    WORD_END = "256" # token for word endings
    
    # helper func
    # recursively splits merged tokens back into main forms
    def expand_token(token):
        if token.isdigit() and int(token) <= 256:
            return [token]
        elif token in RESERVED_TOKENS:
            return []
        elif "_" in token:
            left, right = token.split("_", 1)
            return expand_token(left) + expand_token(right)
        else:
            return []    
    
    out_bytes = []
    word_bytes = []
    
    for token in tokens:
        if token in RESERVED_TOKENS:
            continue
        sub_tokens = expand_token(token)
        for subtoken in sub_tokens:
            if subtoken == WORD_END:
                try:
                    out_bytes.append(
                        bytes([int(b) for b in word_bytes]).decode("utf-8", errors="replace")
                    )
                except Exception:
                    out_bytes.append("")
                word_bytes = []
            else:
                word_bytes.append(subtoken)
    
    if word_bytes:
        try:
            out_bytes.append(
                bytes([int(b) for b in word_bytes]).decode('utf-8', errors='replace')
            )
        except Exception:
            out_bytes.append("")
    return "".join(out_bytes)
            

def save_vocab(vocab, rollno, vocab_size):
    """Save vocabulary file in required format."""
    fname = f"{rollno}_assignment2_bpe_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    # Replace with your actual roll number
    rollno = "240354"

    # Training
    train_text = load_training_data(args.train)
    vocab, tokenizer = train_bpe_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    # Tokenization
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenise_bpe(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    # Detokenization
    detok_text = detokenize_bpe(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
