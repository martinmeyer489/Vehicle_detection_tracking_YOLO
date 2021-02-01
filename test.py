import config as cfg

print(cfg.WRITE_INTO_DB)
cfg.WRITE_INTO_DB = False
print(cfg.WRITE_INTO_DB)