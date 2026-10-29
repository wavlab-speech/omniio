"""Genomic signal tracks (bigWig / bedGraph): one entry per chromosome, binned at a fixed
width at ingestion and stored as float16, so a genomic interval is one byte-range read of
``(end - start) / bin * 2`` bytes. See ``common.py``."""
