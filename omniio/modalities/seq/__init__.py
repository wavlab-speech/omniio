"""Nucleotide sequences (reference / consensus genomes): one entry per record (chromosome),
2-bit packed with separate 1-bit N and soft-mask (lowercase) bitmaps, so a genomic interval
is three small byte-range reads. See ``common.py`` for the layout."""
