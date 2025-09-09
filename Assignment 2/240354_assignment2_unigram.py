import argparse
import math
import os
from collections import defaultdict

def load_training_data(train_path):
    """loads training data and returns word frequencies"""
    word_frequencies = defaultdict(int)
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.strip().split():
                # using end of word markers here as well
                word_frequencies[word + '</w>'] += 1
    return word_frequencies

def train_unigram_tokenizer(word_frequencies, vocab_size):
    """Initialize, run EM training loop, prune to vocab_size."""
    
    # seed vocab initialisation
    seed_vocab = defaultdict(int)
    for word, frequency in word_frequencies.items():
        for i in range(len(word)):
            for j in range(i+1, len(word)+1):
                seed_vocab[word[i:j]] += frequency
                
    # have to make sure all single chars are in the vocab
    character_vocab = set()
    for word in word_frequencies:
        character_vocab.update(word)
    
    # start with vocab TWICE AS LONG AS TARGET VOCAB SIZE
    initial_vocab_size = int(vocab_size * 2)
    
    # start making new vocab
    vocab = ['<pad>', '<unk>', '<s>', '</s>'] + list(character_vocab)
    
    # sort subtrings based on frequency
    substrings_sort = sorted(seed_vocab.items(), key=lambda x: x[1], reverse=True)
    for substring, _ in substrings_sort:
        if len(vocab) >= initial_vocab_size:
            break
        if substring not in vocab:
            vocab.append(substring)
            
    # calculate probabilities
    tokenwise_prob = {token: 1.0 / len(vocab) for token in vocab}
    
    # expectation maximimization STARTS
    number_of_iterations = 10
    for _ in range(number_of_iterations):
        token_frequencies = defaultdict(int)
        total_token_count = 0

        for word, frequency in word_frequencies.items():
            _, segment = viterbi_segment(word, tokenwise_prob)
            for token in segment:
                if token == "<unk>": continue
                token_frequencies[token] += frequency
                total_token_count += frequency
            
        if total_token_count == 0: break
        
        for token in tokenwise_prob:
            if total_token_count > 0:
                tokenwise_prob[token] = token_frequencies[token] / total_token_count
    # ENDS
              
    # pruning the vocab
    while len(vocab) > vocab_size:
        token_losses = {}
        for token in vocab:
            if token in ['<pad>', '<unk>', '<s>', '</s>']: continue
            token_losses[token] = tokenwise_prob.get(token, 0)
        
        number_of_tokens_remove = max(1, int(len(vocab) * 0.1))
        if len(vocab) - number_of_tokens_remove < vocab_size:
            number_of_tokens_remove = len(vocab) - vocab_size
        if number_of_tokens_remove <= 0:
            break
            
        sorted_token_losses = sorted(token_losses.items(), key = lambda x: x[1])
        
        for i in range(number_of_tokens_remove):
            removing_token = sorted_token_losses[i][0]
            if removing_token in vocab:
                vocab.remove(removing_token)
                if removing_token in tokenwise_prob:
                    del tokenwise_prob[removing_token]
        
    final_vocab_probs = sorted([item for item in tokenwise_prob.items() if item[0] not in ['<pad>', '<unk>', '<s>', '</s>']], key=lambda x: x[1], reverse=True)
    final_vocab = ['<pad>', '<unk>', '<s>', '</s>'] + [item[0] for item in final_vocab_probs]
        
    tokenizer = {'tokenwise_probs' : tokenwise_prob}
    return final_vocab, tokenizer

def viterbi_segment(word, token_probs):
    """use viterbi to find best possible segmentation of a word"""
    best_scores = [-math.inf] * (len(word) + 1)
    best_scores[0] = 0
    backpointers = [None] * (len(word) + 1)

    for i in range(1, len(word) + 1):
        for j in range(i):
            token = word[j:i]
            prob = token_probs.get(token)
            if prob and prob > 0 and best_scores[j] > -math.inf:
                score = best_scores[j] + math.log(prob)
                if score > best_scores[i]:
                    best_scores[i] = score
                    backpointers[i] = j

    if best_scores[-1] == -math.inf:
        return -math.inf, ['<unk>']

    segmentation = []
    i = len(word)
    while i > 0:
        j = backpointers[i]
        if j is None:
            # this means this part cannoy be segmented
            # fallback to <unk> for whole word
            return -math.inf, ['<unk>']
        segmentation.insert(0, word[j:i])
        i = j
    
    return best_scores[-1], segmentation
            

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_unigram_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def tokenize(text, tokenizer):
    """tokenize using unigram model trained"""
    tokenwise_probs = tokenizer['tokenwise_probs']
    words = text.strip().split()
    tokens_list = []
    for word in words:
        if not word: continue
        _, segment = viterbi_segment(word + '</w>', tokenwise_probs)
        tokens_list.extend(segment)
    return tokens_list
    
def detokenize(tokens, tokenizer = None):
    """detokenize"""
    return "".join(tokens).replace('</w>', ' ').strip()

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_unigram_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_unigram_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, required=True)
    args = parser.parse_args()

    rollno = "240354"

    train_text = load_training_data(args.train)
    vocab, tokenizer = train_unigram_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, 'r', encoding='utf-8') as f:
        sample_text = (f.read())
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)  