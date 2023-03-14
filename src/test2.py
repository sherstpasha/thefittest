import pstats

s = pstats.Stats("outga_old2.file")
s.sort_stats("tottime").print_stats(50)