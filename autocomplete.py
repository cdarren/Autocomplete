import doctest
from text_tokenize import tokenize_sentences


class PrefixTree:
    def __init__(self):
        self.value = None
        self.children = {}

    def __setitem__(self, key, value):
        """
        Add a key with the given value to the prefix tree,
        or reassign the associated value if it is already present.
        Raise a TypeError if the given key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError
        if not key:
            self.value = value
        else:
            start = key[0]
            rest = key[1:]
            if start not in self.children:
                self.children[start] = PrefixTree()
            self.children[start][rest] = value

    def __getitem__(self, key):
        """
        Return the value for the specified prefix.
        Raise a KeyError if the given key is not in the prefix tree.
        Raise a TypeError if the given key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError
        if not key:
            if self.value is None:
                raise KeyError
            return self.value

        start = key[0]
        rest = key[1:]
        if start not in self.children:
            raise KeyError
        return self.children[start][rest]

    def __delitem__(self, key):
        """
        Delete the given key from the prefix tree if it exists.
        Raise a KeyError if the given key is not in the prefix tree.
        Raise a TypeError if the given key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError
        if not key:
            if self.value is None:
                raise KeyError
            self.value = None
        else:
            start = key[0]
            rest = key[1:]
            if start not in self.children:
                raise KeyError
            del self.children[start][rest]
            if not self.children[start]:
                del self.children[start]

    def __contains__(self, key):
        """
        Is key a key in the prefix tree?  Return True or False.
        Raise a TypeError if the given key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError
        node = self
        for char in key:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.value is not None

    def __iter__(self):
        """
        Generator of (key, value) pairs for all keys/values in this prefix tree
        and its children.  Must be a generator!
        """
        for char, child in self.children.items():
            yield from ((char + key, value) for key, value in child)
            if child.value is not None:
                yield char, child.value


def word_frequencies(text):
    """
    Given a piece of text as a single string, create a prefix tree whose keys
    are the words in the text, and whose values are the number of times the
    associated word appears in the text.
    """
    # Tokenize the input text into sentences using the provided tokenize_sentences function
    sentences = tokenize_sentences(text)

    # Create an instance of the PrefixTree class
    tree = PrefixTree()

    # Iterate through the sentences and tokenize each sentence into words
    for sentence in sentences:
        words = sentence.lower().split()

        # Iterate through the words, adding them to the tree, and incrementing their frequency count
        for word in words:
            if word not in tree:
                tree[word] = 1
            else:
                tree[word] += 1

    return tree


def autocomplete(tree, prefix, max_count=None):
    """
    Return the list of the most-frequently occurring elements that start with
    the given prefix.  Include only the top max_count elements if max_count is
    specified, otherwise return all.

    Raise a TypeError if the given prefix is not a string.
    """
    if not isinstance(prefix, str):
        raise TypeError

    # Helper function to traverse the tree and collect keys with their frequency
    def collect_keys(node, current_key):
        if node.value is not None:
            yield current_key, node.value
        for char, child in node.children.items():
            yield from collect_keys(child, current_key + char)

    # Find keys that start with the given prefix
    node = tree
    for char in prefix:
        if char not in node.children:
            return []
        node = node.children[char]

    # Collect the keys and their frequencies
    keys_with_freqs = list(collect_keys(node, prefix))

    # Sort the keys based on their frequency
    sorted_keys = sorted(keys_with_freqs, key=lambda x: (-x[1], x[0]))

    # Return the top max_count keys if max_count is specified, otherwise return all keys
    return [key for key, _ in sorted_keys[:max_count]]


def edits(word, tree):
    """
    Generate valid edits for the given word that are present in the tree.
    """
    valid_edits = set()

    # Generate all possible edits with a single character deletion
    for i in range(len(word)):
        candidate = word[:i] + word[i + 1 :]
        if candidate in tree:
            valid_edits.add(candidate)

    # Generate all possible edits with a single character insertion
    for i in range(len(word) + 1):
        for char in "abcdefghijklmnopqrstuvwxyz":
            candidate = word[:i] + char + word[i:]
            if candidate in tree:
                valid_edits.add(candidate)

    # Generate all possible edits with a single character substitution
    for i in range(len(word)):
        for char in "abcdefghijklmnopqrstuvwxyz":
            if word[i] != char:
                candidate = word[:i] + char + word[i + 1 :]
                if candidate in tree:
                    valid_edits.add(candidate)

    # Generate all possible edits with a two-character transpose
    for i in range(len(word) - 1):
        candidate = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        if candidate in tree:
            valid_edits.add(candidate)

    return valid_edits


def autocorrect(tree, prefix, max_count=None):
    """
    Return the list of the most-frequent words that start with prefix or that
    are valid words that differ from prefix by a small edit.  Include up to
    max_count elements from the autocompletion.  If autocompletion produces
    fewer than max_count elements, include the most-frequently-occurring valid
    edits of the given word as well, up to max_count total elements.
    """
    autocompleted_words = set(autocomplete(tree, prefix, max_count))

    # Generate valid edits
    valid_edits = edits(prefix, tree)

    # Get the frequency of autocompleted words and valid edits and sort them by frequency
    autocompleted_words_with_freq = [(word, tree[word]) for word in autocompleted_words]
    valid_edits_with_freq = [(word, tree[word]) for word in valid_edits]

    sorted_autocompleted_words = sorted(
        autocompleted_words_with_freq, key=lambda x: (-x[1], x[0])
    )
    sorted_valid_edits = sorted(valid_edits_with_freq, key=lambda x: (-x[1], x[0]))

    if max_count is None:
        # Combine the sorted autocompleted words and valid edits
        combined_results = list(
            set(
                [word for word, freq in sorted_autocompleted_words]
                + [word for word, freq in sorted_valid_edits]
            )
        )

        # Sort the combined results by frequency
        combined_results_with_freq = [(word, tree[word]) for word in combined_results]
        sorted_combined_results = sorted(
            combined_results_with_freq, key=lambda x: (-x[1], x[0])
        )

        return [word for word, freq in sorted_combined_results]

    else:
        # Calculate the remaining slots for valid edits after autocompleted words
        remaining_slots = max_count - len(autocompleted_words)

        # Return the top max_count results
        return [word for word, _ in sorted_autocompleted_words] + [
            word for word, _ in sorted_valid_edits
        ][:remaining_slots]


def word_filter(tree, pattern):
    """
    Return list of (word, freq) for all words in the given prefix tree that
    match pattern.  pattern is a string, interpreted as explained below:
         * matches any sequence of zero or more characters,
         ? matches any single character,
         otherwise char in pattern char must equal char in word.
    """
    def match(node, current_key, pattern):
        if not pattern:
            if node.value is not None:
                return [(current_key, node.value)]
            else:
                return []

        if pattern[0] == '?':
            results = []
            for char, child in node.children.items():
                results.extend(match(child, current_key + char, pattern[1:]))
            return results
        elif pattern[0] == '*':
            idx = 0
            while idx < len(pattern) and pattern[idx] == '*':
                idx += 1
            results = match(node, current_key, pattern[idx:])
            for char, child in node.children.items():
                results.extend(match(child, current_key + char, pattern))
            return results
        else:
            if pattern[0] in node.children:
                return match(node.children[pattern[0]], current_key + pattern[0], pattern[1:])
            else:
                return []

    results = match(tree, "", pattern)
    return list(set(results))

# you can include test cases of your own in the block below.
if __name__ == "__main__":
    doctest.testmod()

    with open("alice.txt", encoding="utf-8") as f:
        alice = f.read()
    with open("two_cities.txt", encoding="utf-8") as f:
        tale = f.read()
    with open("pride.txt", encoding="utf-8") as f:
        pride = f.read()
    with open("drac.txt", encoding="utf-8") as f:
        drac = f.read()
    with open("meta.txt", encoding="utf-8") as f:
        meta = f.read()

    # Create the prefix tree structure
    alice_tree = word_frequencies(alice)

    # Find autocorrections for 'hear'
    autocorrections = autocorrect(alice_tree, 'hear')

    # Remove duplicates
    unique_autocorrections = list(set(autocorrections))

    # Sort the unique autocorrections by frequency
    unique_autocorrections_with_freq = [(word, alice_tree[word]) for word in unique_autocorrections]
    sorted_unique_autocorrections = sorted(unique_autocorrections_with_freq, key=lambda x: (-x[1], x[0]))

    # Get the top 12 autocorrections
    top_12_autocorrections = [word for word, freq in sorted_unique_autocorrections][:12]

    print(top_12_autocorrections)
