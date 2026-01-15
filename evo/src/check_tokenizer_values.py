from transformers import AutoTokenizer

# ! denotes a cassette nucleotide
# - denotes a repeat nucleotide
# + denotes a spacer nucleotide
# ABCDGHKMNRSTVWY are the unique characters in the sequences of the dataset
# Found using the following code:
# unique_chars = set()
# for seq in dataset['sequence']:
#     unique_chars.update(set(seq))
# print(sorted(list(unique_chars)))

seq = "ABCDGHKMNRSTVWXY!-+"
print(f"Sequence: {seq}")

print("Load the tokenizer")
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/evo-1-8k-base", trust_remote_code=True)

print("Tokenize the sequence")
# Tokenize the sequence
tokens = tokenizer(seq, return_tensors="pt")

print(tokens)