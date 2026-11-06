"""Sequences of discrete integers (VQ / RVQ codes, token ids, labels) — codec agnostic.

An entry is N *streams* of non-negative integers, each with its own length, alphabet
size and optional rate (units per second); RVQ is the case of N equal-length streams
sharing one rate. Nothing about codebooks, levels or models is stored — put that in the
free-form ``codec`` metadata. Payload: each stream bit-packed at ``ceil(log2(vocab))``
bits (the storage-optimal fixed-width encoding for near-uniform codes: 1.25 B/token at
1024-way), optionally zstd-wrapped; see ``common.py`` for the byte layout.
"""
