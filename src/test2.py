import pstats

s = pstats.Stats("outgp1.file")
s.sort_stats("tottime").print_stats(50)