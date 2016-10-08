The input to the program is a file with lines containing the following information.
    A sequence of letters indicating sequence A
    A sequence of letters indicating sequence B
    An indication of whether local (1) or global (0) alignment is sought.
    A set of gap penalties for introducing gaps into A or B.
    The symbol alphabets to be used (e.g. ATGC for DNA strands and 21 single-letter abbreviations for the proteins, but there could be any alpha-numeric symbol (A-Z, a-z, 0-9))
    Lines showing the score between an element in A and one in B.
The output of the program is a file with lines containing the following information:
    score for the best alignment (rounded to the first decimal place i.e. 3.1415->3.1).
    All alignments which achieve the best score, with the format:
        empty line
        sequence A (with necessary gaps)
        sequence B (with necessary gaps) such that aligned characters from A and B are in the same column.
    Remember not to include the gaps at the very start and end of the alignment.
    Please use the underscore ("_") symbol for a gap as in the example output below.