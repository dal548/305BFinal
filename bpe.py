import collections
import re
import pickle
import sys

def build_intial_vocab(text):
    vocab = collections.Counter()
    tokens = re.split(r'(\s+)', text)
    for token in tokens:
        if token:
            # Represent the token as a tuple of characters (preserving whitespace)
            tokenized_token = tuple(token)
            vocab[tokenized_token] += 1
    return vocab

def get_pair_stats(vocab):
    pair_freqs = collections.Counter()
    for tokenized_word, freq in vocab.items():
        tokens = list(tokenized_word)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_vocab(pair, vocab):
    merged_token = "".join(pair)
    new_vocab = {}
    for tokenized_word, freq in vocab.items():
        tokens = list(tokenized_word)
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_vocab[tuple(new_tokens)] = freq
    return new_vocab

def bpe_tokenizer(text, desired_vocab_size, verbose = False):
    vocab = build_intial_vocab(text)
    merges = []
    
    while True:
        current_tokens = set()
        for tokenized_word in vocab:
            current_tokens.update(tokenized_word)

        if verbose:
            if len(current_tokens) % 100 == 0:
                print(len(current_tokens))
        
        if len(current_tokens) >= desired_vocab_size:
            break
        
        pair_stats = get_pair_stats(vocab)
        if not pair_stats:
            break
        
        best_pair = max(pair_stats, key=pair_stats.get)
        merges.append(best_pair)
        
        vocab = merge_vocab(best_pair, vocab)
    
    return merges, vocab, current_tokens

def apply_bpe(word, merges):
    tokens = list(word)
    for merge in merges:
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) == merge:
                tokens = tokens[:i] + ["".join(merge)] + tokens[i+2:]
                i = max(i-1, 0)
            else:
                i += 1
    return tokens

def bpe_tokenize(text, merges):
    pieces = re.split(r'(\s+)', text)
    pieces = [p for p in pieces if p]
    output = []
    for piece in pieces:
        if piece.isspace():
            output.append(piece)
        else:
            output.extend(apply_bpe(piece, merges))
    return output

if __name__ == "__main__":
    # read sys args
    if len(sys.argv) < 3:
        print("Usage: python bpe.py <input_file> <desired_vocab_size> [verbose]")
        sys.exit(1)

    input_file = sys.argv[1]
    desired_vocab_size = int(sys.argv[2])
    verbose = False
    if len(sys.argv) == 4:
        verbose = sys.argv[3] == "True"

    # read input file
    with open(f'data/{input_file}', 'r') as f:
        text = f.read()

    # tokenize
    merges, vocab, current_tokens = bpe_tokenizer(text, desired_vocab_size, verbose)

    # put in dict
    res = {"merges": merges, "vocab": vocab, "current_tokens": current_tokens}

    # save
    with open(f'models/bpe_tokenizer_{input_file[:-4]}_{desired_vocab_size}.pkl', 'wb') as f:
        pickle.dump(res, f)